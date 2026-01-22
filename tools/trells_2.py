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

def load_trellis2_pipeline():
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
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


def process_rgb_images_to_shapes(flag, base_dir="/home/lian/Desktop/genpc_base/data", 
                                 trellis2_pipeline=None):
    """从 RGB 图片生成对应的 shape 文件
    
    Args:
        flag: 数据集标识符（如 'table3'）
        base_dir: 基础目录
        trellis2_pipeline: TRELLIS.2 管道对象（如果为 None 则加载）
        
    Returns:
        dict: 处理结果 {image_name: {status, output_path}}
    """
    # 加载管道
    if trellis2_pipeline is None:
        print("正在加载 TRELLIS.2 管道...")
        trellis2_pipeline = load_trellis2_pipeline()
    
    # 定义路径
    rgb_dir = os.path.join(base_dir, 'scene', 'rgb', flag)
    shape_dir = os.path.join(base_dir, 'scene', 'shape', flag)
    
    # 检查 RGB 目录是否存在
    if not os.path.exists(rgb_dir):
        print(f"✗ RGB 目录不存在: {rgb_dir}")
        return {}
    
    # 创建 shape 输出目录
    os.makedirs(shape_dir, exist_ok=True)
    
    print(f"\n" + "="*70)
    print(f"从 RGB 图片生成 Shape 文件 (flag: {flag})")
    print("="*70)
    print(f"\n输入目录: {rgb_dir}")
    print(f"输出目录: {shape_dir}\n")
    
    # 获取所有 RGB 图片
    image_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = [f for f in os.listdir(rgb_dir) 
                   if f.lower().endswith(image_formats)]
    
    if not image_files:
        print(f"✗ 在 {rgb_dir} 中未找到图片文件")
        return {}
    
    print(f"找到 {len(image_files)} 个图片文件:")
    for img_file in image_files:
        print(f"  - {img_file}")
    
    results = {}
    
    # 处理每个图片
    for idx, image_file in enumerate(image_files, 1):
        image_name = os.path.splitext(image_file)[0]  # 去掉扩展名
        image_path = os.path.join(rgb_dir, image_file)
        
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_file}")
        print("-" * 70)
        
        try:
            # 读取图片
            img = Image.open(image_path)
            print(f"✓ 已读取图片: {image_path}")
            print(f"  尺寸: {img.size}")
            
            # 运行 TRELLIS.2 管道
            print(f"正在运行 TRELLIS.2 管道...")
            trellis2_pipeline.cuda()
            
            start_time = time.time()
            
            # 运行管道
            mesh = trellis2_pipeline.run(img)[0]
            mesh.simplify(16777216)  # nvdiffrast limit
            
            run_time = time.time()
            print(f"✓ 管道运行完成: {run_time - start_time:.2f} 秒")
            
            # 导出到 GLB
            print(f"正在生成 GLB 文件...")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=500000,
                texture_size=2048,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False
            )
            
            export_time = time.time()
            print(f"✓ GLB 生成完成: {export_time - run_time:.2f} 秒")
            
            # 保存 GLB 文件（文件名与输入图片一致）
            output_path = os.path.join(shape_dir, f"{image_name}.glb")
            glb.export(output_path, extension_webp=True)
            
            print(f"✓ 已保存: {output_path}")
            
            # 将 GLB 转换为 PLY 点云
            print(f"正在转换为 PLY 点云...")
            ply_path = os.path.join(shape_dir, f"{image_name}.ply")
            success = glb_to_ply(output_path, ply_path, num_points=100000, verbose=True)
            
            if success:
                results[image_file] = {
                    'status': 'success',
                    'output_path': output_path,
                    'ply_path': ply_path,
                    'process_time': export_time - start_time
                }
            else:
                results[image_file] = {
                    'status': 'success',
                    'output_path': output_path,
                    'ply_path': None,
                    'process_time': export_time - start_time
                }
            
        except Exception as e:
            print(f"✗ 处理失败: {str(e)}")
            results[image_file] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 显示统计信息
    print(f"\n" + "="*70)
    print(f"处理完成")
    print("="*70)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"\n✓ 成功: {success_count}/{len(results)}")
    if failed_count > 0:
        print(f"✗ 失败: {failed_count}/{len(results)}")
    
    if success_count > 0:
        print(f"\n✓ 文件已保存到: {shape_dir}")
        print(f"\n生成的文件:")
        for img_file, result in results.items():
            if result['status'] == 'success':
                print(f"  - {os.path.basename(result['output_path'])} (GLB)")
                if result.get('ply_path'):
                    print(f"    {os.path.basename(result['ply_path'])} (PLY - 100000 点)")
    
    print("\n" + "="*70 + "\n")
    
    return results

if __name__ == '__main__':
    
    flag = 'table3'
    base_dir = 'data'
    
    # 加载 TRELLIS.2 管道
    print("加载 TRELLIS.2 管道...")
    trellis2_pipeline = load_trellis2_pipeline()
    
    # 从 RGB 图片生成 Shape 文件
    results = process_rgb_images_to_shapes(flag, base_dir, trellis2_pipeline)