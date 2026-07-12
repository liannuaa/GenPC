import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import open3d as o3d
from utils.dataUtils import save_ply_xyzrgb
from utils.runtime import model_path, sample_file
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
        if bool(getattr(self.cfg, "skip_existing", False)) and ply_path.exists() and glb_path.exists():
            print(f" Skip shape generation for {flag}: existing {ply_path.name} and {glb_path.name}.")
            return
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
        reg(
            self.cfg,
            flag,
            cd_inv_weight=cd_inv_weight,
            diff_init=diff_init,
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
