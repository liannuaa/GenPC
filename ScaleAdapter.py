import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import open3d as o3d
from types import SimpleNamespace
from utils.dataUtils import save_ply_xyzrgb
from utils.runtime import data_dir, model_path, sample_dir, sample_file
import warnings
from reg_xyz import reg

warnings.filterwarnings("ignore")

class ScaleAdapter():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        if self.cfg.rembg_model == 'rembg':
            from rembg import remove
            self.rembg = remove
        elif self.cfg.rembg_model == 'RMBG':
            from tools.RMBG import RMBG_pred
            rmbg_model_path = model_path(self.cfg, "rmbg_model_path", "RMBG-2.0")
            self.rembg = lambda input_path, output_path: RMBG_pred(
                input_path, output_path, model_path=rmbg_model_path
            )
        else:
            raise NotImplementedError(f"Background model {self.cfg.rembg_model} not implemented.")

        if self.cfg.generative_model == "instantmesh":
            from tools.instantmesh import instantmesh
            self.generative = instantmesh
        elif self.cfg.generative_model in ("hunyuan2.0", "hunyuan2.1"):
            from tools.hunyuan3d_2 import hunyuan3d_2
            self.generative = hunyuan3d_2
        elif self.cfg.generative_model == "hunyuan3d_omni":
            from tools.hunyuan3d_omni import hunyuan3d_omni
            self.generative = hunyuan3d_omni
        elif self.cfg.generative_model == 'trellis':
            from trellis import trellis
            self.generative = trellis
        elif self.cfg.generative_model == 'trellis_2':
            from tools.trells_2 import trellis_2
            self.generative = trellis_2
        else:
            raise NotImplementedError(f"Generative model {self.cfg.generative_model} not implemented.")

    def remove_bg(self, flag, img_resource):
        input_path = sample_file(self.cfg, flag, "image.png" if img_resource == "obj" else "img.png")
        output_path = sample_file(self.cfg, flag, "img_sam.png")
        if bool(getattr(self.cfg, "skip_existing", False)) and output_path.exists():
            print(f" Skip background removal for {flag}: existing {output_path.name}.")
            return
        self.rembg(str(input_path), str(output_path))

    def colorPoint(self, flag, xyz, rgb, img_resource):
        point_uv = np.load(sample_file(self.cfg, flag, "point_uv.npy"))
        if img_resource == 'obj':
            save_ply_xyzrgb(
                xyz.detach().cpu().numpy(),
                rgb.detach().cpu().numpy(),
                str(sample_file(self.cfg, flag, "color_point.ply")),
            )
            return
        elif img_resource == 'depth':
            img = Image.open(sample_file(self.cfg, flag, "img.png"))
        else:
            raise ValueError(f"Unknown image resource: {img_resource}")
        # 如果point_uv是numpy则转换为torch
        if isinstance(point_uv, np.ndarray):
            point_uv = torch.tensor(point_uv).to(self.device)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_width, img_height = img.size
        img = transforms.ToTensor()(img).to(self.device)
        img_np = img.detach().cpu().numpy()
        point_pixel = torch.stack(
            (
                point_uv[:, 1] * (img_height - 1),
                point_uv[:, 0] * (img_width - 1),
            ),
            dim=-1,
        ).long()
        point_pixel[:, 0] = point_pixel[:, 0].clip(0, img_height - 1)
        point_pixel[:, 1] = point_pixel[:, 1].clip(0, img_width - 1)
        colors = np.zeros_like(xyz.detach().cpu().numpy())
        for i, (x, y) in enumerate(point_pixel.detach().cpu().numpy()):
            colors[i] = img_np[:, x, y]
        save_ply_xyzrgb(
            xyz.detach().cpu().numpy(),
            colors,
            str(sample_file(self.cfg, flag, "color_point.ply")),
        )

    def img2shape(self, flag):
        ply_path = sample_file(self.cfg, flag, f"{flag}_{self.cfg.generative_model}.ply")
        glb_path = sample_file(self.cfg, flag, f"{flag}_{self.cfg.generative_model}.glb")
        if bool(getattr(self.cfg, "skip_existing", False)) and ply_path.exists():
            print(f" Skip shape generation for {flag}: existing {ply_path.name}.")
            return
        img = Image.open(sample_file(self.cfg, flag, "img_sam.png"))
        self.generative(self.cfg, flag, img)

    def scaleReg(self, flag):
        if str(getattr(self.cfg, "reg_backend", "")).lower() == "render_to_moge_sim3":
            self.render_to_moge_sim3_reg(flag)
            return

        sample_overrides = getattr(self.cfg, "reg_sample_overrides", {}) or {}
        override = sample_overrides.get(str(flag), {})
        cd_inv_weight = float(getattr(self.cfg, "reg_cd_inv_weight", 0.5))
        diff_init = bool(getattr(self.cfg, "reg_diff_init", True))
        reg_fine_xyz = bool(getattr(self.cfg, "reg_fine_xyz", True))
        if override:
            cd_inv_weight = float(override.get("cd_inv_weight", cd_inv_weight))
            diff_init = bool(override.get("diff_init", diff_init))
            reg_fine_xyz = bool(override.get("reg_fine_xyz", reg_fine_xyz))
        reg(
            self.cfg,
            flag,
            cd_inv_weight=cd_inv_weight,
            diff_init=diff_init,
            reg_fine_xyz=reg_fine_xyz,
        )

    def resolve_partial_path(self, flag):
        input_paths = getattr(self.cfg, "input_paths", {}) or {}
        if str(flag) in input_paths:
            return str(input_paths[str(flag)])

        direct_path = sample_dir(self.cfg, flag) / str(flag)
        if direct_path.exists():
            return str(direct_path)

        for suffix in (".ply", ".pcd"):
            candidate = data_dir(self.cfg) / f"{flag}{suffix}"
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(f"Input point cloud for '{flag}' not found under {data_dir(self.cfg)}.")

    def render_to_moge_sim3_reg(self, flag):
        from scripts.run_moge_to_raw_partial_from_camera import run as run_moge_to_raw_partial
        from scripts.run_render_to_moge_sim3 import run as run_render_to_moge_sim3

        flag = str(flag)
        out_sample_dir = sample_dir(self.cfg, flag)
        final_path = sample_file(self.cfg, flag, f"{flag}_fused.ply")
        info_path = sample_file(self.cfg, flag, f"{flag}_render_to_moge_sim3_info.json")
        if bool(getattr(self.cfg, "skip_existing", False)) and final_path.exists() and info_path.exists():
            print(f" Skip render-to-MoGe Sim3 registration for {flag}: existing {final_path.name}.")
            return

        partial_path = self.resolve_partial_path(flag)
        moge_model = model_path(self.cfg, "moge_model_path", "moge-2-vitl")
        rmbg_model = model_path(self.cfg, "rmbg_model_path", "RMBG-2.0")
        image_size = int(getattr(self.cfg, "generate_res", getattr(self.cfg, "res", 512)))

        bridge_transform = sample_file(
            self.cfg,
            flag,
            f"{flag}_moge_to_raw_partial_moge_to_raw_partial_transform.npy",
        )
        bridge_mask = sample_file(self.cfg, flag, f"{flag}_moge_to_raw_partial_object_mask.png")
        bridge_info = sample_file(self.cfg, flag, f"{flag}_moge_to_raw_partial_info.json")
        if not (
            bool(getattr(self.cfg, "skip_existing", False))
            and bridge_transform.exists()
            and bridge_mask.exists()
            and bridge_info.exists()
        ):
            run_moge_to_raw_partial(
                SimpleNamespace(
                    sample_dir=str(out_sample_dir),
                    partial_path=partial_path,
                    image_name="img.png",
                    camera_name="camera.pth",
                    moge_model=str(moge_model),
                    rmbg_model=str(rmbg_model),
                    device=self.cfg.device,
                    fp16=bool(getattr(self.cfg, "moge_fp16", True)),
                    image_size=image_size,
                    padding=float(getattr(self.cfg, "padding", 0.15)),
                    max_pixel_distance=float(getattr(self.cfg, "moge_bridge_max_pixel_distance", 2.0)),
                    object_alpha_threshold=int(getattr(self.cfg, "object_alpha_threshold", 128)),
                    object_mask_erode_pixels=int(getattr(self.cfg, "object_mask_erode_pixels", 2)),
                    ransac_iterations=int(getattr(self.cfg, "moge_bridge_ransac_iterations", 5000)),
                    ransac_threshold=float(getattr(self.cfg, "moge_bridge_ransac_threshold", 0.08)),
                    max_correspondences=int(getattr(self.cfg, "moge_bridge_max_correspondences", 4000)),
                    seed=int(getattr(self.cfg, "reg_geotransformer_seed", 7351)),
                    output_prefix=f"{flag}_moge_to_raw_partial",
                )
            )
        else:
            print(f" Skip MoGe-to-raw-partial bridge for {flag}: existing transform and mask.")

        run_render_to_moge_sim3(
            SimpleNamespace(
                flag=flag,
                sample_root=str(out_sample_dir.parent),
                sample_dir=str(out_sample_dir),
                out_root=str(out_sample_dir.parent),
                out_dir=str(out_sample_dir),
                image_name="img.png",
                complete_name=f"{flag}_{self.cfg.generative_model}.ply",
                object_mask_name=f"{flag}_moge_to_raw_partial_object_mask.png",
                moge_to_partial_name=f"{flag}_moge_to_raw_partial_moge_to_raw_partial_transform.npy",
                partial_path=partial_path,
                moge_model=str(moge_model),
                device=self.cfg.device,
                fp16=bool(getattr(self.cfg, "moge_fp16", True)),
                object_alpha_threshold=int(getattr(self.cfg, "object_alpha_threshold", 128)),
                object_mask_erode_pixels=int(getattr(self.cfg, "render_moge_object_mask_erode_pixels", 0)),
                eval_points=int(getattr(self.cfg, "render_sim3_eval_points", 60000)),
                seed=int(getattr(self.cfg, "render_sim3_seed", 6145)),
                splat_radius=int(getattr(self.cfg, "render_sim3_splat_radius", 1)),
                scale_multipliers=getattr(self.cfg, "render_sim3_scale_multipliers", "0.55,0.7,0.85,1.0,1.15,1.3"),
                translation_steps=getattr(self.cfg, "render_sim3_translation_steps", "0.12,0.06,0.03,0.015"),
                rotation_steps_deg=getattr(self.cfg, "render_sim3_rotation_steps_deg", "12,6,3"),
                scale_steps=getattr(self.cfg, "render_sim3_scale_steps", "1.12,1.06,1.03"),
                refine_rounds=int(getattr(self.cfg, "render_sim3_refine_rounds", 1)),
                visible_icp_iterations=int(getattr(self.cfg, "render_sim3_visible_icp_iterations", 3)),
                icp_trim_quantile=float(getattr(self.cfg, "render_sim3_icp_trim_quantile", 0.7)),
                icp_max_pairs=int(getattr(self.cfg, "render_sim3_icp_max_pairs", 20000)),
                icp_rollback_on_score_drop=bool(getattr(self.cfg, "render_sim3_icp_rollback_on_score_drop", True)),
                icp_min_score_gain=float(getattr(self.cfg, "render_sim3_icp_min_score_gain", 0.0)),
                final_name=f"{flag}_fused.ply",
            )
        )
        

    def scaleAdapter(self, xyz, flag, rgb=None):
        print("Stage 2 : .....")
        if rgb is not None:
            img_resource = 'obj' # 使用点云自身颜色
        else:
            img_resource = 'depth' # 使用controlnet输出的rgb图片
        self.remove_bg(flag, img_resource=img_resource)
        if str(getattr(self.cfg, "reg_backend", "")).lower() == "render_to_moge_sim3":
            self.img2shape(flag)
            return
        self.colorPoint(flag, xyz, rgb, img_resource=img_resource)
        self.img2shape(flag)
