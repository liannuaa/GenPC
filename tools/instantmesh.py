import os
import PIL
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import open3d as o3d
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl, save_ply, save_glb
from src.utils.infer_util import remove_background, resize_foreground, save_video


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)

    frames = torch.cat(frames, dim=1)[0]  # we suppose batch size is always 1
    return frames

diffusion_steps = 75
seed = 42
scale = 1.0
distance = 4.5
view = 6
no_rembg = True
export_texmap = False
seed_everything(seed)
###############################################################################
# Stage 0: Configuration.
###############################################################################
config = OmegaConf.load('configs/instant-mesh-base.yaml')
config_name = os.path.basename('configs/instant-mesh-base.yaml').replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config
IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
device = 'cuda'
# load diffusion model
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline='zero123plus',
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
# load custom white-background UNet
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin",
                                     repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)
pipeline = pipeline.to(device)
# load reconstruction model
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh",
                                      filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

def instantmesh(cfg, flag, image):
    # make output directories
    output_path = f'{cfg.output_path}/{flag}/'
    image_path = os.path.join(output_path)
    mesh_path = os.path.join(output_path)
    video_path = os.path.join(output_path)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    ###############################################################################
    # Stage 1: Multiview generation.
    ###############################################################################
    outputs = []
    # sampling
    output_image = pipeline(
        image,
        num_inference_steps=diffusion_steps,
    ).images[0]
    output_image.save(os.path.join(image_path, f'{flag}_zero123++.png'))
    print(f"  Image saved to {os.path.join(image_path, f'{flag}_zero123++.png')}")
    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)
    outputs.append({'name': flag, 'images': images})
    # delete pipeline to save memory
    # del pipeline

    ###############################################################################
    # Stage 2: Reconstruction.
    ###############################################################################
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * scale).to(device)
    chunk_size = 20 if IS_FLEXICUBES else 1
    for idx, sample in enumerate(outputs):
        name = sample['name']
        images = sample['images'].unsqueeze(0).to(device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
        if view == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(device)
            images = images[:, indices]
            input_cameras = input_cameras[:, indices]
        with torch.no_grad():
            # get triplane
            planes = model.forward_planes(images, input_cameras)
            # get mesh
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=export_texmap,
                **infer_config,
            )
            mesh_path_idx = os.path.join(mesh_path, f'{name}_instantmesh.obj')
            # mesh_path_idx = os.path.join(mesh_path, f'{name}_instantmesh.ply')
            if export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path_idx,
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                # save_obj(vertices, faces, vertex_colors, mesh_path_idx)
                # save_ply(vertices, vertex_colors, mesh_path_idx.replace('.obj', '.ply'))
                save_glb(vertices, faces, vertex_colors, mesh_path_idx.replace('.obj', '.glb'))
            print(f"  Mesh saved to {mesh_path_idx.replace('.obj', '.glb')}")

# configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video

def pcn_instantmesh(category, save_video_flag=True):
    diffusion_steps = 75
    seed = 0
    scale = 1.0
    distance = 4.5
    view = 6
    no_rembg = True
    save_video_flag = save_video_flag
    seed_everything(seed)
    ###############################################################################
    # Stage 0: Configuration.
    ###############################################################################
    config = OmegaConf.load('configs/instant-mesh-base.yaml')
    config_name = os.path.basename('configs/instant-mesh-base.yaml').replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
    device = torch.device('cuda')
    # load diffusion model
    print('Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # load custom white-background UNet
    print('Loading custom white-background unet ...')
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin",
                                         repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(device)

    # load reconstruction model
    print('Loading reconstruction model ...')
    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh",
                                          filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()

    # make output directories
    output_path = os.path.join("workspace",category)
    image_path = os.path.join(output_path)
    mesh_path = os.path.join(output_path)
    video_path = os.path.join(output_path)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * scale).to(device)
    file_number = len([f for f in os.listdir(output_path) if f.startswith(category)])
    i = 1
    for flag in os.listdir(output_path):
        if not flag.startswith(category):
            continue
        print(f'-------------{i}/{file_number}---{flag}--------')
        i += 1

        file_path = os.path.join(output_path, flag)
        image_path = os.path.join(output_path, flag, 'sam.png')
        image = PIL.Image.open(image_path)
        ###############################################################################
        # Stage 1: Multiview generation.
        ###############################################################################
        outputs = []
        # sampling
        output_image = pipeline(
            image,
            num_inference_steps=diffusion_steps,
        ).images[0]

        output_image.save(os.path.join(file_path, f'{flag}_zero123++.png'))
        print(f"Image saved to {os.path.join(file_path, f'{flag}_zero123++.png')}")

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)
        outputs.append({'name': flag, 'images': images})

        # delete pipeline to save memory
        # del pipeline

        ###############################################################################
        # Stage 2: Reconstruction.
        ###############################################################################

        chunk_size = 20 if IS_FLEXICUBES else 1

        for idx, sample in enumerate(outputs):
            name = sample['name']
            print(f'[{idx + 1}/{len(outputs)}] Creating {name} ...')

            images = sample['images'].unsqueeze(0).to(device)
            images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

            if view == 4:
                indices = torch.tensor([0, 2, 4, 5]).long().to(device)
                images = images[:, indices]
                input_cameras = input_cameras[:, indices]

            with torch.no_grad():
                # get triplane
                planes = model.forward_planes(images, input_cameras)
                # get mesh
                mesh_out = model.extract_mesh(
                    planes,
                    **infer_config,
                )
                mesh_path_idx = os.path.join(mesh_path, flag, f'{name}_instantmesh.obj')
                # mesh_path_idx = os.path.join(mesh_path, f'{name}_instantmesh.ply')
                vertices, faces, vertex_colors = mesh_out
                # save_obj(vertices, faces, vertex_colors, mesh_path_idx)
                # save_ply(vertices, vertex_colors, mesh_path_idx.replace('.obj', '.ply'))
                save_glb(vertices, faces, vertex_colors, mesh_path_idx.replace('.obj', '.glb'))
                print(f"Mesh saved to {mesh_path_idx.replace('.obj', '.glb')}")

                # get video
                if save_video_flag:
                    video_path_idx = os.path.join(video_path, flag, f'{name}.mp4')
                    render_size = infer_config.render_resolution
                    render_cameras = get_render_cameras(
                        batch_size=1,
                        M=120,
                        radius=distance,
                        elevation=20.0,
                        is_flexicubes=IS_FLEXICUBES,
                    ).to(device)
                    frames = render_frames(
                        model,
                        planes,
                        render_cameras=render_cameras,
                        render_size=render_size,
                        chunk_size=chunk_size,
                        is_flexicubes=IS_FLEXICUBES,
                    )
                    save_video(frames, video_path_idx, fps=30, )
                    print(f"Video saved to {video_path_idx}")




if __name__ == '__main__':
    flag = 'swivel chair'
    image = PIL.Image.open(f'workspace/{flag}/sam.png')
    # image = PIL.Image.open(f'workspace/{flag}/{flag}_gaussian.png')
    instantmesh(flag = flag,image=image,save_video_flag=True)
    # pcn_instantmesh(category='02691156', save_video_flag=True)
    # path = os.path.join('generative')
    # from utils.dataUtils import get_rotate_matrix ,normalize_numpy,numpy2o3d
    # for category in os.listdir(path):
    #     category_path = os.path.join(path, category)
    #     for file in os.listdir(category_path):
    #         file_file = os.path.join(category_path, file)
    #         # 在file_file文件夹下找到snapshotx.png
    #         for xx in os.listdir(file_file):
    #             if xx.startswith('snapshot'):
    #                 img_path = os.path.join(file_file, xx)
    #                 image = PIL.Image.open(img_path)
    #                 flag = category+'_'+file
    #                 # instantmesh(flag=flag, image=image, save_video_flag=True)
    #                 pcd = o3d.io.read_triangle_mesh(f'workspace/{flag}/{flag}_instantmesh.glb')
    #                 xyz = pcd.sample_points_uniformly(number_of_points=16384)
    #                 # o3d.visualization.draw_geometries([pcd])
    #                 x_rot_90 = get_rotate_matrix("x", 90)
    #                 y_rot_90 = get_rotate_matrix("y", 90)
    #                 np_xyz = np.asarray(xyz.points)
    #                 np_xyz = np.dot(np_xyz, x_rot_90.T)
    #                 np_xyz = np.dot(np_xyz, y_rot_90.T)
    #                 np_xyz = normalize_numpy(np_xyz,range=0.5)
    #                 # o3d.visualization.draw_geometries([numpy2o3d(np_xyz)])
    #                 # np_color = np.asarray(xyz.colors)
    #                 xyz = numpy2o3d(np_xyz)
    #                 o3d.io.write_point_cloud(f'{file_file}/instantmesh.pcd', xyz)
    # category = 'table_base'
    # file = '2'
    # img_path = os.path.join("generative",category,file,"snapshot12.png")
    # image = PIL.Image.open(img_path)
    # flag = category + '_' + file
    # instantmesh(flag=flag, image=image, save_video_flag=True)
    # pcd = o3d.io.read_triangle_mesh(f'workspace/{flag}/{flag}_instantmesh.glb')
    # xyz = pcd.sample_points_uniformly(number_of_points=16384)
    # o3d.io.write_point_cloud(f'generative/{category}/{file}/instantmesh.pcd', xyz)
