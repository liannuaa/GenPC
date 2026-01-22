import yaml
import torch
from munch import Munch
from utils.dataUtils import *
from DepthPrompting import DepthPrompting
from ScaleAdapter import *
from utils.loss_util import Completionloss
from fpsample import fps_sampling
import gc
from pathlib import Path

def free_memory():
    """释放GPU显存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_file_names_from_dir(directory):
    """
    获取目录下所有文件的文件名（不包含后缀）
    
    Args:
        directory: 目录路径，可以是字符串或Path对象
        
    Returns:
        list: 文件名列表（不包含后缀）
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"目录不存在: {directory}")
        return []
    
    if not directory.is_dir():
        print(f"不是目录: {directory}")
        return []
    
    file_names = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            # 获取文件名（不包含后缀）
            file_name = file_path.stem
            file_names.append(file_name)
    
    return sorted(file_names)  # 排序返回


def main(cfg):
    """主函数：处理多个样本"""
    category = 'PED'
    flags = get_file_names_from_dir(f'./data/waymo/{category}')

    ## 第一阶段：深度提示
    # print("=== Stage 1: Depth Prompting ===")
    # dp = DepthPrompting(cfg)
    # for flag in flags:
    #     print(f'Processing {flag}...')
    #     # 加载点云数据
    #     xyz, rgb = torch.tensor(load_xyz(f'./data/waymo/{category}/{flag}.ply')).to(cfg.device)
    #     # 深度提示阶段
    #     dp.getImage(xyz, flag, depth_gen=True, img_gen=True)
    #     # 释放当前循环的显存
    #     del xyz, rgb
    #     free_memory()
    
    # ## 完全释放第一阶段的模型
    # del dp
    # free_memory()
    # print(f"Stage 1 完成，显存已释放")
    # print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 第二阶段：尺度适配
    print("\n=== Stage 2: Scale Adapter ===")
    sa = ScaleAdapter(cfg)
    for flag in flags:
        print(f'Processing {flag}...')
        # 加载点云数据
        xyz, rgb = torch.tensor(load_xyz(f'./data/waymo/{category}/{flag}.ply')).to(cfg.device)
        # 尺度适配阶段
        sa.scaleAdapter(xyz, flag)
        del xyz, rgb
        free_memory()
    
    # 完全释放第二阶段的模型
    del sa
    free_memory()
    for flag in flags:
        xyz, rgb = torch.tensor(load_xyz(f'./data/waymo/{category}/{flag}.ply')).to(cfg.device)
        reg(cfg, flag, cd_inv_weight=0.5, diff_init=True, reg_fine_xyz=True)
    
if __name__ == '__main__':
    cfg_txt = open('./configs/config_lidar_ped.yaml', "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    main(cfg)
