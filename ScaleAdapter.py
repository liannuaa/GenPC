import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import yaml
from munch import Munch
import open3d as o3d
from utils.dataUtils import *
import warnings
import io
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
            self.rembg = RMBG_pred
        if self.cfg.generative_model == "instantmesh":
            from tools.instantmesh import instantmesh
            self.generative = instantmesh
        elif self.cfg.generative_model == 'sf3d':
            from SF3D import sf3d
            self.generative = sf3d
        elif self.cfg.generative_model == 'trellis':
            from trellis import trellis
            self.generative = trellis
        elif self.cfg.generative_model == 'trellis_2':
            from tools.trells_2 import trellis_2
            self.generative = trellis_2

    def remove_bg(self, flag, img_resource):
        if img_resource == 'obj':
            img = Image.open(f'{self.cfg.output_path}/{flag}/image.png')
        elif img_resource == 'depth':
            img = Image.open(f'{self.cfg.output_path}/{flag}/img.png')
        output_path = self.rembg(f'{self.cfg.output_path}/{flag}/img.png', f'{self.cfg.output_path}/{flag}/img_sam.png')

    def colorPoint(self, flag, xyz, gt, rgb, img_resource):
        cam = torch.load(f'{self.cfg.output_path}/{flag}/camera.pth', weights_only=False)
        point_uv = np.load(f'{self.cfg.output_path}/{flag}/point_uv.npy')
        if img_resource == 'obj':
            save_ply_xyzrgb(xyz.detach().cpu().numpy(), rgb.detach().cpu().numpy(), f'{self.cfg.output_path}/{flag}/color_point.ply')
            return
        elif img_resource == 'depth':
            img = Image.open(f'{self.cfg.output_path}/{flag}/img.png')
        # 如果point_uv是numpy则转换为torch
        if isinstance(point_uv, np.ndarray):
            point_uv = torch.tensor(point_uv).to(self.device)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = transforms.ToTensor()(img).to(self.device)
        img_np = img.detach().cpu().numpy()
        point_pixel = point_uv * 1024  # [piont_num,2]
        point_pixel = point_pixel.long()
        point_pixel = torch.cat((point_pixel[:, 1].unsqueeze(-1), point_pixel[:, 0].unsqueeze(-1)), dim=-1)
        point_pixel = point_pixel.clip(0, 1024 - 1)
        colors = np.zeros_like(xyz.detach().cpu().numpy())
        for i, (x, y) in enumerate(point_pixel.detach().cpu().numpy()):
            colors[i] = img_np[:, x, y]
        # o3d.io.write_point_cloud(f"workspace/{flag}/GT.ply", numpy2o3d(gt))
        save_ply_xyzrgb(xyz.detach().cpu().numpy(), colors, f'{self.cfg.output_path}/{flag}/color_point.ply')

    def img2shape(self, flag):
        img = Image.open(f'{self.cfg.output_path}/{flag}/img_sam.png')
        self.generative(self.cfg, flag, img)

    def scaleReg(self, flag):
        reg(self.cfg, flag, cd_inv_weight=0.5, diff_init=True, reg_fine_xyz=True)
        

    def scaleAdapter(self, xyz, flag, rgb=None):
        print("Stage 2 : .....")
        if rgb is not None:
            img_resource = 'obj' # 使用点云自身颜色
        else:
            img_resource = 'depth' # 使用controlnet输出的rgb图片
        self.remove_bg(flag, img_resource=img_resource)
        self.colorPoint(flag, xyz, xyz, rgb, img_resource=img_resource)
        self.img2shape(flag)

if __name__ == "__main__":
    cfg_txt = open('./configs/config.yaml', "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    sa = ScaleAdapter(cfg)
    flag = '09639'
    xyz = torch.tensor(load_xyz(f'./data/{flag}.ply')).to(cfg.device)
    gt = torch.tensor(load_xyz(f'./data/GT/{flag}.ply')).to(cfg.device)
    sa.colorPoint(flag, xyz, gt)
    sa.img2shape(flag)
    sa.scaleAdapter(xyz, flag)
