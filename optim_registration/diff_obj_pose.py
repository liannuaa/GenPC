import os
import sys
import math
import time
import cv2
import numpy as np
import imageio
sys.path.append('../')
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.transforms import rotation_6d_to_matrix
from torch import nn, optim
from tqdm import tqdm

from utils.dataUtils import load_xyz, glb2point
from utils.loss_util import *

sys.path.append('..')
"""
    7-DoF 物体姿态优化 (旋转 + 平移 + 尺度)
    优化 complete (CAD) 点云的姿态, 使其在固定相机下渲染结果匹配 partial 点云渲染。
"""


def load_ref_img(img_path, target_size=512, object_scale=0.8, device='cpu'):
    """
    输入一个RGBA图片，使用alpha通道作为mask，返回mask后的图像，并将物体缩放到target_size的指定比例。

    Args:
        img_path (str): 输入图片路径。
        target_size (int): 目标尺寸，默认为512。
        object_scale (float): 物体相对于target_size的缩放比例，默认为0.8（80%）。
        device (str): 设备，默认为'cpu'。

    Returns:
        tuple: (torch.Tensor, torch.Tensor) Masked并resize后的图像和对应的mask。
    """
    # 读取图片
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    # 检查是否为RGBA图像
    if img.shape[2] == 4:  # 如果是RGBA图像
        alpha = img[:, :, 3]  # 提取alpha通道
        mask = (alpha > 0).astype(np.float32)  # 创建mask
        img = img[:, :, :3]  # 去掉alpha通道
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img * mask[:, :, None]  # 应用mask到RGB图像
    else:
        raise ValueError("Input image must be RGBA format.")

    # 找到前景区域的边界框
    mask_coords = np.where(mask > 0)
    if len(mask_coords[0]) == 0:
        raise ValueError("No foreground object found in the image")
    
    min_y, max_y = mask_coords[0].min(), mask_coords[0].max()
    min_x, max_x = mask_coords[1].min(), mask_coords[1].max()
    
    # 裁剪出前景区域
    foreground_img = img[min_y:max_y+1, min_x:max_x+1]
    foreground_mask = mask[min_y:max_y+1, min_x:max_x+1]
    
    # 获取前景区域尺寸
    fg_h, fg_w = foreground_img.shape[:2]
    
    # 计算目标尺寸（target_size的object_scale倍）
    target_object_size = int(target_size * object_scale)
    
    # 计算等比例缩放尺寸，使物体适应target_object_size
    scale = min(target_object_size / fg_w, target_object_size / fg_h)
    new_w = int(fg_w * scale)
    new_h = int(fg_h * scale)
    
    # Resize前景图像和mask
    resized_img = cv2.resize(foreground_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(foreground_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 创建目标大小的黑色背景和空白mask
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    mask_canvas = np.zeros((target_size, target_size), dtype=np.float32)
    
    # 将resize后的图像和mask放置在中心
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img
    mask_canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_mask
    
    # 转换为张量
    canvas_tensor = torch.from_numpy(canvas).float() / 255.0
    mask_tensor = torch.from_numpy(mask_canvas).float()
    
    return canvas_tensor.to(device), mask_tensor.to(device)

def render_reference_image(vert_pos, vert_col, radius, render_size, device):
    """渲染 partial 参考图像和 mask (固定相机)"""
    R, T = look_at_view_transform(eye=torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, device=device), device=device)
    cameras = PerspectiveCameras(
        focal_length=(4.0,),
        R=R,
        T=T,
        image_size=((render_size, render_size),),
        device=device
    )
    raster_settings = PointsRasterizationSettings(
        image_size=(render_size, render_size),
        radius=torch.tensor([radius], dtype=torch.float32, device=device).expand(vert_pos.shape[0]),
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PulsarPointsRenderer(rasterizer=rasterizer).to(device)
    pcl = Pointclouds(points=vert_pos[None, ...], features=vert_col[None, ...])
    result = renderer(
        pcl,
        gamma=(1e-2,),
        zfar=(5.0,),
        znear=(1e-4,),
        radius_world=True,
        bg_col=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
    )[0]
    ref_mask = compute_mask_from_rendering(result)
    return result, ref_mask, R, T

def load_point_cloud(point_path, device, radius=0.05, num_points=5000):
    """
    加载点云数据

    Args:
        point_path (str): 点云文件路径
        device (torch.device): 设备
        radius (float): 下采样半径
        num_points (int): 点云中点的数量

    Returns:
        tuple: 点的位置和颜色 (N,3) 张量
    """
    if point_path.endswith('.ply'):
        xyz, color = load_xyz(point_path, down_sample=radius)
    elif point_path.endswith('.glb'):
        pcd = glb2point(point_path, down_sample=radius, num_points=num_points)
        xyz, color = np.asarray(pcd.points), np.asarray(pcd.colors)
    else:
        raise ValueError('Unsupported point cloud format')
    n_points = xyz.shape[0]
    vert_pos = torch.tensor(xyz, dtype=torch.float32, device=device)
    if color is None:
        vert_col = torch.ones(n_points, 3, dtype=torch.float32, device=device)
    else:
        vert_col = torch.tensor(color, dtype=torch.float32, device=device)
        if vert_col.max() > 1.0:
            vert_col = vert_col / 255.0
    return vert_pos, vert_col

def compute_mask_from_rendering(rendered_img, threshold=0.1, method='luminance'):
    """从渲染结果生成mask (硬阈值, 不可导)
    NOTE: 仅用于参考/可视化, 不应用于需要梯度的 IoU 计算。
    """
    # 若已有 alpha 通道
    if rendered_img.shape[-1] == 4:
        alpha = rendered_img[..., 3]
        print('Alpha channel detected')
        return (alpha > 0).float()
    if method == 'occupancy':
        return (rendered_img.sum(-1) > threshold).float()
    luminance = 0.299 * rendered_img[:, :, 0] + 0.587 * rendered_img[:, :, 1] + 0.114 * rendered_img[:, :, 2]
    return (luminance > threshold).float()

# 新增: 可微 soft mask + soft IoU
def soft_iou_loss(result_img, ref_mask, threshold=0.1, tau=0.05, method='luminance', smooth=1e-6):
    """计算可微分的 soft IoU 损失。
    result_img: (H,W,3)
    ref_mask:  (H,W) 0/1 (不需要梯度)
    通过对亮度做 (lum - threshold)/tau 的 Sigmoid 平滑近似 occupancy。
    tau 越小越接近硬阈值, 但梯度会更尖锐。
    返回: (loss, iou_value, pred_soft_mask)
    """
    if method == 'occupancy':
        occ_raw = result_img.sum(-1)
    else:  # luminance
        occ_raw = 0.299 * result_img[:, :, 0] + 0.587 * result_img[:, :, 1] + 0.114 * result_img[:, :, 2]
    # 平滑近似: sigmoid((x - th)/tau)
    pred_soft = torch.sigmoid((occ_raw - threshold) / tau)
    intersection = (pred_soft * ref_mask).sum()
    union = pred_soft.sum() + ref_mask.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    loss = 1.0 - iou
    return loss, iou, pred_soft

def normalize_images(ref_img, result_img, method='statistical'):
    """标准化图像以减少颜色差异
    Args:
        ref_img: 参考图像 [H, W, C]
        result_img: 结果图像 [H, W, C]  
        method: 标准化方法 'statistical' 或 'histogram'
    """
    if method == 'statistical':
        # 统计标准化 - 让result图像的均值和方差匹配ref图像
        ref_mean = torch.mean(ref_img, dim=(0, 1), keepdim=True)
        ref_std = torch.std(ref_img, dim=(0, 1), keepdim=True) + 1e-6
        
        result_mean = torch.mean(result_img, dim=(0, 1), keepdim=True)
        result_std = torch.std(result_img, dim=(0, 1), keepdim=True) + 1e-6
        
        # 标准化result图像使其分布匹配ref图像
        result_normalized = (result_img - result_mean) / result_std * ref_std + ref_mean
        result_normalized = torch.clamp(result_normalized, 0.0, 1.0)
        
    elif method == 'histogram':
        # 简化的直方图匹配 - 仅匹配亮度通道
        ref_gray = 0.299 * ref_img[:, :, 0] + 0.587 * ref_img[:, :, 1] + 0.114 * ref_img[:, :, 2]
        result_gray = 0.299 * result_img[:, :, 0] + 0.587 * result_img[:, :, 1] + 0.114 * result_img[:, :, 2]
        
        ref_mean = torch.mean(ref_gray)
        ref_std = torch.std(ref_gray) + 1e-6
        result_mean = torch.mean(result_gray)
        result_std = torch.std(result_gray) + 1e-6
        
        # 对每个通道应用亮度匹配
        result_normalized = (result_img - result_mean) / result_std * ref_std + ref_mean
        result_normalized = torch.clamp(result_normalized, 0.0, 1.0)
    else:
        result_normalized = result_img
    
    return ref_img, result_normalized


def dice_loss(pred_mask, target_mask, smooth=1e-6):
    """计算可微分的Dice损失
    Args:
        pred_mask: 预测mask [H, W]
        target_mask: 目标mask [H, W]
        smooth: 平滑项
    Returns:
        dice_loss: Dice损失值
    """
    # 展平为1D张量
    pred_flat = pred_mask.view(-1)
    target_flat = target_mask.view(-1)
    
    # 计算交集和并集
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice_coeff

def compute_soft_mask(rendered_img, threshold=0.1, tau=0.05, method='luminance'):
    """计算可微分的soft mask
    Args:
        rendered_img: 渲染图像 [H, W, C]
        threshold: 阈值
        tau: 平滑参数，越小越接近硬阈值
        method: 计算方法
    Returns:
        soft_mask: 可微分的soft mask [H, W]
    """
    if method == 'occupancy':
        occ_raw = rendered_img.sum(-1)
    else:  # luminance
        occ_raw = 0.299 * rendered_img[:, :, 0] + 0.587 * rendered_img[:, :, 1] + 0.114 * rendered_img[:, :, 2]
    
    # 使用sigmoid创建soft mask，保持梯度
    soft_mask = torch.sigmoid((occ_raw - threshold) / tau)
    return soft_mask

def edge_loss(ref_img, result):
    """计算边缘损失 (MSE)"""
    ref_edges_h = torch.abs(ref_img[:, :-1, :] - ref_img[:, 1:, :])
    ref_edges_v = torch.abs(ref_img[:-1, :, :] - ref_img[1:, :, :])
    result_edges_h = torch.abs(result[:, :-1, :] - result[:, 1:, :])
    result_edges_v = torch.abs(result[:-1, :, :] - result[1:, :, :])
    edge_loss_value = F.mse_loss(ref_edges_h, result_edges_h) + F.mse_loss(ref_edges_v, result_edges_v)
    return edge_loss_value

def compute_loss_function(ref_img, result, ref_mask=None, ref_points=None, result_points=None, cdloss=None):
    """计算组合损失函数 (包含可微 soft IoU + Chamfer Distance)。"""
    # 图像标准化
    ref_img_norm, result_norm = normalize_images(ref_img, result)
    
    # 使用可微分的soft mask而不是硬阈值mask
    mask_result = compute_soft_mask(result_norm, threshold=0.1, tau=0.05)
    mask_ref = compute_soft_mask(ref_img_norm, threshold=0.1, tau=0.05)
    
    # 保存mask用于可视化 (使用硬阈值版本)
    # imageio.imsave('mask_ref.png', (compute_mask_from_rendering(ref_img_norm).detach().cpu().numpy() * 255).astype(np.uint8))
    # imageio.imsave('mask_result.png', (compute_mask_from_rendering(result_norm).detach().cpu().numpy() * 255).astype(np.uint8))

    # MSE损失 (使用标准化后的图像)
    mse_loss = F.mse_loss(ref_img_norm, result_norm)

    # 可微分的mask损失 - 使用多种损失函数
    mask_loss = F.mse_loss(mask_result, mask_ref) * 30 + F.binary_cross_entropy(mask_result, mask_ref)

    # Dice损失 (对形状更敏感)
    mask_dice_loss = dice_loss(mask_result, mask_ref)
    
    # 组合mask损失 - 可以调整权重
    mask_loss = mask_loss * 1 + mask_dice_loss * 10

    # 边缘损失
    edge_loss_value = torch.tensor(0.0, device=result.device)
    # edge_loss_value = edge_loss(ref_img_norm, result_norm)
    
    # IoU损失
    iou_loss_value = torch.tensor(0.0, device=result.device)
    iou_value = torch.tensor(1.0, device=result.device)
    if ref_mask is not None:
        # 使用可微 soft IoU (不再用硬阈值)
        iou_loss_value, iou_value, _ = soft_iou_loss(result_norm, ref_mask, threshold=0.1, tau=0.05)

    # Chamfer Distance损失 (3D点云)
    cd_loss_value = torch.tensor(0.0, device=result.device)
    if ref_points is not None and result_points is not None:
        # cd_loss_value = chamfer_distance_loss(ref_points, result_points)
        cd_loss_value = + cdloss.chamfer_partial_l1(result_points.unsqueeze(0), ref_points.unsqueeze(0)) + \
             0.5 * cdloss.chamfer_partial_l1(ref_points.unsqueeze(0), result_points.unsqueeze(0)) 

    # 权重: 可根据需要调节
    total_loss = (mse_loss * 0 + 
                  edge_loss_value * 0 + 
                  mask_loss * 1 +
                  iou_loss_value * .0 + 
                  cd_loss_value * 3.0)  # CD损失权重
    
    return total_loss, mse_loss, edge_loss_value, iou_loss_value, iou_value, cd_loss_value, mask_loss


class ObjectPoseOptim(nn.Module):
    """7-DoF 物体姿态优化模块 (旋转(6D)+平移+尺度)

    改进:
      1. 仅在 __init__ 中构建 cameras / rasterizer / renderer, 减少重复开销。
      2. 分离局部坐标 (去质心) + 旋转 + 尺度 + 复位 + 平移 的步骤, 便于调试。
      3. 提供 optional 正交化(投影) 以防数值漂移 (project_every > 0 时生效)。
      4. 提供 get_transform / get_current_RT 便于外部访问当前变换。
      5. 可选择返回变换后的点 (return_pts)。
    """
    def __init__(self, vert_pos, vert_col, radius, render_size, device, R_cam, T_cam, init_rot,
                 focal=4.0, project_every=0):
        super().__init__()
        self.device = device
        self.render_size = render_size
        self.gamma = 1e-2
        self.project_every = project_every  # 每多少步做一次正交化(0=不做)

        # 固定几何数据
        self.register_parameter('vert_pos', nn.Parameter(vert_pos, requires_grad=False))
        self.register_parameter('vert_col', nn.Parameter(vert_col, requires_grad=False))
        self.register_parameter('vert_rad', nn.Parameter(
            torch.full((vert_pos.shape[0],), radius, dtype=torch.float32, device=device), requires_grad=False))
        self.register_buffer('center', vert_pos.mean(0))  # 质心

        # 姿态参数
        self.rot_6d = nn.Parameter(init_rot)
        self.trans = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
        self.log_scale = nn.Parameter(torch.tensor([math.log(0.75)], dtype=torch.float32, device=device))  # scale = exp(log_scale) = 0.75

        # 相机固定缓存
        self.register_buffer('R_cam', R_cam)
        self.register_buffer('T_cam', T_cam)
        self.register_buffer('focal_length', torch.tensor([focal], dtype=torch.float32, device=device))

        self.cameras = PerspectiveCameras(
            focal_length=(self.focal_length,),
            R=self.R_cam, T=self.T_cam,
            image_size=((self.render_size, self.render_size),),
            device=self.device
        )
        raster_settings = PointsRasterizationSettings(
            image_size=(self.render_size, self.render_size),
            radius=self.vert_rad * 1.1
        )
        self.rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.renderer = PulsarPointsRenderer(self.rasterizer).to(self.device)

        # 预缓存常量
        self.register_buffer('bg_col', torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device))
        self.register_buffer('znear', torch.tensor([1e-4], dtype=torch.float32, device=self.device))
        self.register_buffer('zfar', torch.tensor([5.0], dtype=torch.float32, device=self.device))
        self.register_buffer('gamma_buf', torch.tensor([self.gamma], dtype=torch.float32, device=self.device))
        self._step = 0


    def get_current_RT(self):
        R = rotation_6d_to_matrix(self.rot_6d[None])[0].detach()
        s = torch.exp(self.log_scale.detach())[0]
        t = self.trans.detach()
        return R, t, s

    def get_transform(self):
        R, t, s = self.get_current_RT()
        T = torch.eye(4, device=R.device)
        T[:3, :3] = R * s
        T[:3, 3] = t
        return T

    def forward(self, return_pts=False, project_now=False):
        # 可选正交化
        if project_now or (self.project_every > 0 and self._step % self.project_every == 0 and self._step > 0):
            with torch.no_grad():
                R_tmp = rotation_6d_to_matrix(self.rot_6d[None])[0]
                R_proj = self._project_to_rotation(R_tmp)
                # 使用官方转换函数保持列优先 (col0|col1)
                self.rot_6d.data = matrix_to_rotation_6d(R_proj[None])[0].data
        self._step += 1

        # 旋转 + 尺度 + 平移
        R_obj = rotation_6d_to_matrix(self.rot_6d[None])[0]
        scale = torch.exp(self.log_scale)[0]
        local = (self.vert_pos - self.center) * scale
        local = (R_obj @ local.T).T
        pts = local + self.center + self.trans

        pcl = Pointclouds(points=pts[None], features=self.vert_col[None])
        img = self.renderer(
            pcl,
            gamma=self.gamma_buf,
            zfar=self.zfar,
            znear=self.znear,
            radius_world=True,
            bg_col=self.bg_col,
        )[0]
        if return_pts:
            return img, R_obj, scale, pts
        return img, R_obj, scale

def create_opencv_visualization_obj(ref_img, result, model, i, loss,  iou_value=None, vis=True):
    result_im = (result.detach().cpu().numpy() * 255).astype(np.uint8)
    ref_im = (ref_img.detach().cpu().numpy() * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_im, cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(ref_im, cv2.COLOR_RGB2BGR)
    overlay_img = np.ascontiguousarray(((result * 0.5 + ref_img * 0.5).detach().cpu().numpy() * 255).astype(np.uint8))
    overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    h, w = ref_bgr.shape[:2]
    result_bgr = cv2.resize(result_bgr, (w, h))
    overlay_bgr = cv2.resize(overlay_bgr, (w, h))
    combined = np.hstack([ref_bgr, result_bgr, overlay_bgr])
    title_height = 40
    title_img = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
    for idx, name in enumerate(["Reference", "Optim", "Overlay"]):
        cv2.putText(title_img, name, (w*idx + w//2 - 50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    final_img = np.vstack([title_img, combined])
    txt = f"Step {i} Loss {loss:.4f} Scale {torch.exp(model.log_scale).item():.3f}"
    cv2.putText(final_img, txt, (10, title_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    if iou_value is not None:
        cv2.putText(final_img, f"IoU {iou_value:.3f}", (10, title_height + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    if vis:
        cv2.imshow("Object Pose Optimization", final_img)
        key = cv2.waitKey(1) & 0xFF
        return key == 27, final_img
    return None, final_img

def build_transform(R_obj, t, scale):
    T = torch.eye(4, device=R_obj.device)
    T[:3, :3] = R_obj * scale  # 将尺度吸收进旋转块 (各向同性)
    T[:3, 3] = t
    return T

def get_init_rot(axis, angle_deg, device):
    """根据轴(axis)与角度(度)生成对应的6D旋转表示 (列拼接col0|col1).
    用法: rot6d = get_init_rot('x', 90, device) -> shape (6,)
    """
    if isinstance(axis, str):
        axis_l = axis.lower()
        mapping = {
            'x': torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            'y': torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            'z': torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        }
        if axis_l not in mapping:
            raise ValueError(f"未知轴: {axis}. 允许 'x','y','z' 或 3D 向量")
        axis_vec = mapping[axis_l].to(device)
    else:
        axis_vec = torch.as_tensor(axis, dtype=torch.float32, device=device)
        if axis_vec.numel() != 3:
            raise ValueError("自定义轴必须是长度为3的向量")
        axis_vec = axis_vec / (axis_vec.norm() + 1e-8)
    angle_rad = math.radians(angle_deg)
    rot_vec = axis_vec * angle_rad
    R = axis_angle_to_matrix(rot_vec[None])[0]  # (3,3)
    rot6d = matrix_to_rotation_6d(R[None])[0]  # 正确列排列 [r00,r10,r20,r01,r11,r21]
    return rot6d


def object_pose_optimization(glb_path, point_path, radius=0.005, lr=0.005, iters=300, render_size=224, vis=False, save_path=None, device=None, cam_bias_num=4):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cdloss = Completionloss(loss_func='cd_l1')

    # 加载 partial & complete
    partial_xyz, partial_col = load_point_cloud(point_path=point_path, device=device, radius=radius, num_points=8000)

    complete_xyz, complete_col = load_point_cloud(point_path=glb_path, device=device, radius=radius, num_points=120000)
    print(f"Complete 点云: {complete_xyz.shape[0]} points, Partial 点云: {partial_xyz.shape[0]} points")

    ref_img, ref_mask, R_cam, T_cam = render_reference_image(partial_xyz, partial_col, radius, render_size, device)
    imageio.imsave("partial.png", (ref_img.detach().cpu() * 255).to(torch.uint8).numpy())
    imageio.imsave("partial_mask.png", (ref_mask.detach().cpu().numpy() * 255).astype(np.uint8))
    
    # 保存标准化前后的对比
    print("图像统计信息:")
    print(f"Partial图像 - 均值: {torch.mean(ref_img, dim=(0,1))}, 标准差: {torch.std(ref_img, dim=(0,1))}")
    if save_path is not None:
        writer = imageio.get_writer("pose.gif", mode='I', duration=0.3, loop=0)
    best_loss = float('inf')
    best_state = None
    for start in range(cam_bias_num):
        # print(f"\n=== Multi-start {start+1}/{cam_bias_num} ===")
        #把complete_xyz旋转90, 180, 270
        rot_init = get_init_rot('y', start * 90, device)  # 可选 'x',90 | 'y',180 | 'z',270
        print(f"初始旋转 (6D): {rot_init.tolist()}")
        model = ObjectPoseOptim(complete_xyz, complete_col, radius, render_size, device, R_cam, T_cam, rot_init).to(device)
        optimizer = optim.Adam([
            {"params": [model.rot_6d], "lr": lr},
            {"params": [model.trans], "lr": lr * 0.2},
            {"params": [model.log_scale], "lr": lr * 0.1},
        ])
        pbar = tqdm(range(iters+1), desc=f"Start {start+1}")
        patience = 300
        patience_counter = 0
        local_best = float('inf')
        for i in pbar:
            optimizer.zero_grad()
            result, R_obj, scale, transformed_pts = model(return_pts=True)
            
            # 计算损失函数 (包含点云Chamfer Distance)
            total_loss, mse_loss, edge_loss, iou_loss_value, iou_value, cd_loss_value, mask_loss = compute_loss_function(
                ref_img, result, ref_mask, partial_xyz, transformed_pts, cdloss
            )
            
            # 正交性正则 (R R^T ~= I)
            ortho_err = torch.norm(R_obj @ R_obj.T - torch.eye(3, device=device))
            rot_reg = 0.001 * ortho_err
            loss = total_loss + rot_reg  # 仅使用旋转正则
            loss.backward()
            optimizer.step()
            cur_loss = loss.item()
            if cur_loss < local_best:
                local_best = cur_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > patience:
                print(f"早停: {patience} 无改进")
                break
            pbar.set_postfix({
                'loss': f'{cur_loss:.4f}',
                'mse': f'{mse_loss.item():.4f}',
                'mask': f'{mask_loss.item():.4f}',
                'cd': f'{cd_loss_value.item():.4f}',
                'iou': f'{iou_value.item():.3f}',
                's': f'{scale.item():.3f}'
            })
            if save_path is not None and i % 10 == 0:
                _, stacked_img = create_opencv_visualization_obj(ref_img, result, model, i, cur_loss, iou_value.item(), vis=vis)
                writer.append_data(stacked_img)

        print(f"Start {start+1} best loss: {local_best:.6f}")
        if local_best < best_loss:
            best_loss = local_best
            best_state = {
                'rot_6d': model.rot_6d.detach().clone(),
                'trans': model.trans.detach().clone(),
                'log_scale': model.log_scale.detach().clone()
            }
            # print(f"*** 全局最佳更新 -> {best_loss:.6f}")
    if save_path is not None:
        if vis:
            cv2.destroyAllWindows()
        writer.close()
    # print("\n=== 最终结果 ===")
    # print(f"最佳损失: {best_loss:.6f}")
    # 重建最终变换矩阵
    rot_6d = best_state['rot_6d']
    trans = best_state['trans']
    scale = torch.exp(best_state['log_scale'])[0]
    R_obj = rotation_6d_to_matrix(rot_6d[None])[0]
    T_final = build_transform(R_obj, trans, scale)
    final_transform = T_final.detach().cpu().numpy()
    np.save('final_transform.npy', final_transform)
    print("最终 4x4 变换矩阵 (complete -> partial 相机坐标系):")
    print(final_transform)
    return final_transform

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    final_transform = object_pose_optimization(
        glb_path=args.glb_path,
        point_path=args.point_path,
        radius=args.radius,
        lr=args.lr,
        iters=args.iters,
        render_size=args.render_size,
        vis=args.vis,
        save_path=args.save_path,
        device=device,
        cam_bias_num=args.cam_bias_num
    )
    if args.debug:
        import open3d as o3d
        pcd_full = glb2point(args.glb_path)
        pcd_partial = o3d.io.read_point_cloud(f"../data/{args.flag}.ply")
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
        o3d.visualization.draw_geometries([pcd_full, pcd_partial, coord], window_name='Before')
        pcd_full.transform(final_transform)  # 将 complete 变换以匹配 partial
        o3d.visualization.draw_geometries([pcd_full, pcd_partial, coord], window_name='After')

if __name__ == "__main__":
    """
    python diff_obj_pose.py --flag 09639 --iters 150 --render_size 224 --radius 0.03  --lr 0.01 --debug --vis
    """
    import argparse
    parser = argparse.ArgumentParser(description="7-DoF object pose (R,t,scale) optimization with Pulsar renderer")
    parser.add_argument("--flag", type=str, default='01184')
    parser.add_argument("--glb_path", type=str, default=None)
    parser.add_argument("--point_path", type=str, default=None)
    parser.add_argument("--radius", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--render_size", type=int, default=224)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--save_path", type=str, default=None, help="保存路径")
    parser.add_argument("--cam_bias_num", type=int, default=4, help="相机偏置数量")
    args = parser.parse_args()
    
    if args.glb_path is None:
        args.glb_path = f"/root/shared-nvme/genpc_open/workspace/{args.flag}/{args.flag}_trellis_2.glb"
    if args.point_path is None:
        args.point_path = f"/root/shared-nvme/genpc_open/workspace/{args.flag}/color_point.ply"
    main(args)
