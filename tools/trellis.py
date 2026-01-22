import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import imageio
import warnings
from PIL import Image
import numpy as np
from models.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from models.TRELLIS.trellis.utils import render_utils, postprocessing_utils
warnings.filterwarnings("ignore")
# Load a pipeline from a model folder or a Hugging Face model hub.
trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("models/TRELLIS/TRELLIS-image-large")


def trellis(cfg, flag, img):
    trellis_pipeline.cuda()
    # Run the pipeline
    outputs = trellis_pipeline.run(
        img,
        seed=np.random.randint(0, 100000),
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
    # outputs is a dictionary containing generated 3D workspace in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    # video = render_utils.render_video(outputs['gaussian'][0])['color']
    # imageio.mimsave("sample_gs.mp4", video, fps=30)
    # video = render_utils.render_video(outputs['radiance_field'][0])['color']
    # imageio.mimsave("sample_rf.mp4", video, fps=30)
    # video = render_utils.render_video(outputs['mesh'][0])['normal']
    # imageio.mimsave(f"{cfg.output_path}/{flag}/trellis.mp4", video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(f"{cfg.output_path}/{flag}/{flag}_trellis.glb")
    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(f"{cfg.output_path}/{flag}/{flag}_trellis.ply")

def run(img, save_path):
    trellis_pipeline.cuda()
    # Run the pipeline
    os.makedirs(save_path, exist_ok=True)
    outputs = trellis_pipeline.run(
        img,
        preprocess_image=True,
        seed=42,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
    # outputs is a dictionary containing generated 3D workspace in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    # video = render_utils.render_video(outputs['gaussian'][0])['color']
    # imageio.mimsave("sample_gs.mp4", video, fps=30)
    # video = render_utils.render_video(outputs['radiance_field'][0])['color']
    # imageio.mimsave("sample_rf.mp4", video, fps=30)
    # video = render_utils.render_video(outputs['mesh'][0])['normal']
    # imageio.mimsave(f"{save_path}/trellis.mp4", video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(f"{save_path}/_trellis.glb")
    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(f"{save_path}/_trellis.ply")

if __name__ == '__main__':
    # flag = '3'
    # base_dir = f'output/test//{flag}/'
    # for fold in os.listdir(base_dir):
    #     fold_path = os.path.join(base_dir, fold)
    #     if os.path.isdir(fold_path) and 'moge' not in fold:
    #         img_path = os.path.join(base_dir, fold, f'crop.png')
    #         run(Image.open(img_path), save_path=fold_path)
    img = Image.open('data/moge.jpeg')
    run(img,'./')