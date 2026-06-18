import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import open3d as o3d
from utils.dataUtils import normalize_numpy, save_ply_xyzrgb
from utils.runtime import model_path, sample_file
import warnings
from reg_xyz import reg, load_generated_point_cloud

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
        img = Image.open(sample_file(self.cfg, flag, "img_sam.png"))
        self.generative(self.cfg, flag, img)

    def scaleReg(self, flag):
        sample_overrides = getattr(self.cfg, "reg_sample_overrides", {}) or {}
        override = sample_overrides.get(str(flag), {})
        cd_inv_weight = float(getattr(self.cfg, "reg_cd_inv_weight", 0.5))
        diff_init = bool(getattr(self.cfg, "reg_diff_init", True))
        reg_fine_xyz = bool(getattr(self.cfg, "reg_fine_xyz", True))
        if override:
            cd_inv_weight = float(override.get("cd_inv_weight", cd_inv_weight))
            diff_init = bool(override.get("diff_init", diff_init))
            reg_fine_xyz = bool(override.get("reg_fine_xyz", reg_fine_xyz))
        source_pcd = o3d.io.read_point_cloud(str(sample_file(self.cfg, flag, "color_point.ply")))
        source_xyz = np.asarray(source_pcd.points)
        source_extent = source_xyz.max(axis=0) - source_xyz.min(axis=0)
        direct_fallback = False
        if bool(getattr(self.cfg, "reg_pre_fallback", True)):
            raw_target_pcd = load_generated_point_cloud(
                self.cfg.output_path,
                flag,
                self.cfg.generative_model,
            )
            raw_target_xyz = np.asarray(raw_target_pcd.points)
            raw_target_xyz, _, _ = normalize_numpy(raw_target_xyz, range=0.5)
            raw_target_extent = raw_target_xyz.max(axis=0) - raw_target_xyz.min(axis=0)
            raw_extent_ratio = np.divide(
                raw_target_extent,
                source_extent,
                out=np.zeros_like(raw_target_extent),
                where=source_extent > 1e-9,
            )
            pre_fallback_ratio = float(getattr(self.cfg, "reg_pre_fallback_extent_ratio", 1.6))
            pre_fallback_min_ratio = float(getattr(self.cfg, "reg_pre_fallback_min_extent_ratio", 0.8))
            direct_fallback = (
                float(raw_extent_ratio.max()) > pre_fallback_ratio
                and float(raw_extent_ratio.min()) > pre_fallback_min_ratio
            )
            if direct_fallback:
                print(
                    f"Direct registration fallback for {flag}: "
                    f"raw_extent_ratio={raw_extent_ratio.round(3).tolist()}"
                )
        if direct_fallback:
            reg(
                self.cfg,
                flag,
                cd_inv_weight=float(override.get("cd_inv_weight", getattr(self.cfg, "reg_fallback_cd_inv_weight", 0.0))),
                diff_init=bool(override.get("diff_init", getattr(self.cfg, "reg_fallback_diff_init", False))),
                reg_fine_xyz=reg_fine_xyz,
            )
            return
        reg(
            self.cfg,
            flag,
            cd_inv_weight=cd_inv_weight,
            diff_init=diff_init,
            reg_fine_xyz=reg_fine_xyz,
        )
        adaptive_fallback = bool(override.get("adaptive_fallback", getattr(self.cfg, "reg_adaptive_fallback", True)))
        if adaptive_fallback:
            target_pcd = o3d.io.read_point_cloud(str(sample_file(self.cfg, flag, f"{flag}_registered_gen.ply")))
            target_xyz = np.asarray(target_pcd.points)
            target_extent = target_xyz.max(axis=0) - target_xyz.min(axis=0)
            extent_ratio = np.divide(
                target_extent,
                source_extent,
                out=np.zeros_like(target_extent),
                where=source_extent > 1e-9,
            )
            fallback_ratio = float(getattr(self.cfg, "reg_fallback_extent_ratio", 1.45))
            if float(extent_ratio.max()) > fallback_ratio:
                print(
                    f"Adaptive registration fallback for {flag}: "
                    f"extent_ratio={extent_ratio.round(3).tolist()}"
                )
                reg(
                    self.cfg,
                    flag,
                    cd_inv_weight=float(getattr(self.cfg, "reg_fallback_cd_inv_weight", 0.0)),
                    diff_init=bool(getattr(self.cfg, "reg_fallback_diff_init", False)),
                    reg_fine_xyz=reg_fine_xyz,
                )
        

    def scaleAdapter(self, xyz, flag, rgb=None):
        print("Stage 2 : .....")
        if rgb is not None:
            img_resource = 'obj' # 使用点云自身颜色
        else:
            img_resource = 'depth' # 使用controlnet输出的rgb图片
        self.remove_bg(flag, img_resource=img_resource)
        self.colorPoint(flag, xyz, rgb, img_resource=img_resource)
        self.img2shape(flag)
