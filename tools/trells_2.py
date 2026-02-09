import os
import sys

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（tools的上一级）
project_root = os.path.dirname(current_dir)
# TRELLIS.2 路径
trellis_path = os.path.join(project_root, 'models', 'TRELLIS.2')

sys.path.insert(0, trellis_path)
sys.path.insert(0, project_root)

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
import warnings
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import time
import numpy as np
from pathlib import Path
from utils.dataUtils import glb2point
warnings.filterwarnings("ignore")

trellis2_pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
trellis2_pipeline.cuda()

def load_trellis2_pipeline(ckpt_path):
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(ckpt_path)
    return pipeline


def trellis_2(cfg, flag, img):
    """仿照 trellis 函数，为 TRELLIS.2 创建的函数
    
    Args:
        cfg: 配置对象，需要有 output_path 属性
        flag: 标识符
        img: PIL Image 对象或图片路径
        trellis2_pipeline: TRELLIS.2 管道对象（如果为 None 则加载）
    """
    # 如果 img 是路径字符串，加载图片
    if isinstance(img, str):
        img = Image.open(img)
    
    print(f"正在运行 TRELLIS.2 管道...")
    # 运行管道
    mesh = trellis2_pipeline.run(img)[0]
    mesh.simplify(16777216)  # nvdiffrast limit
    
    # 创建输出目录
    os.makedirs(f"{cfg.output_path}/{flag}", exist_ok=True)
    
    # 导出到 GLB
    print(f"正在生成 GLB 文件...")
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   1000000,
        texture_size        =   4096,
        remesh              =   True,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True
    )
    glb.export(f"{cfg.output_path}/{flag}/{flag}_trellis_2.glb")
    print(f"✓ GLB 已保存: {cfg.output_path}/{flag}/{flag}_trellis_2.glb")
    
    # 转换为 PLY 点云
    ply_path = f"{cfg.output_path}/{flag}/{flag}_trellis_2.ply"
    glb_to_ply(
        f"{cfg.output_path}/{flag}/{flag}_trellis_2.glb",
        ply_path,
        num_points=100000,
        verbose=True
    )


def glb_to_ply(glb_path, ply_path, num_points=16384, verbose=True):
    """将 GLB 文件转换为 PLY 点云文件
    
    Args:
        glb_path (str): 输入 GLB 文件路径
        ply_path (str): 输出 PLY 文件路径
        num_points (int): 采样点数，默认为 100000
        verbose (bool): 是否打印日志
        
    Returns:
        bool: 转换是否成功
    """
    try:
        import open3d as o3d
        
        if verbose:
            print(f"正在从 GLB 转换: {glb_path}")
        
        # 使用 glb2point 从 GLB 文件生成点云
        pcd = glb2point(glb_path, down_sample=None, num_points=num_points)
        
        if pcd is None or len(pcd.points) == 0:
            if verbose:
                print(f"✗ 生成的点云为空")
            return False
        
        # 保存为 PLY 文件
        o3d.io.write_point_cloud(ply_path, pcd)
        
        if verbose:
            print(f"✓ PLY 点云已保存: {ply_path}")
            print(f"  点数: {len(pcd.points)}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ PLY 转换失败: {str(e)}")
        return False



if __name__ == '__main__':
    pass