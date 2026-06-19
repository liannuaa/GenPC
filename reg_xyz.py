from pathlib import Path

import open3d as o3d
import numpy as np
import torch
from utils.dataUtils import (
    get_rotate_matrix,
    glb2point,
    normalize_numpy,
    numpy2o3d,
    remove_noise_from_point_cloud,
)
from copy import deepcopy
from utils.loss_util import Completionloss
from optim_registration.diff_obj_pose import object_pose_optimization
from fpsample import fps_sampling
from utils.runtime import sample_file, save_intermediates


def load_generated_point_cloud(path, flag, model_name, fallback_points=163840):
    sample_path = Path(path) / str(flag)
    ply_path = sample_path / f"{flag}_{model_name}.ply"
    if ply_path.exists():
        return o3d.io.read_point_cloud(str(ply_path))
    return glb2point(str(sample_path / f"{flag}_{model_name}.glb"), num_points=fallback_points)


def icp_with_scaling_xyz(source, target, scales, max_correspondence_distance=0.05, init_transform=np.eye(4)):
    reg_p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # 构建缩放矩阵
    scaling_matrix = np.eye(4)
    scaling_matrix[0, 0] = scales[0]
    scaling_matrix[1, 1] = scales[1]
    scaling_matrix[2, 2] = scales[2]
    # Refine registration with scaling
    source.transform(scaling_matrix)
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, init_transform, reg_p2p
    )
    return result_icp


def icp_with_scaling(source, target,scale, max_correspondence_distance=0.05, init_transform=np.eye(4)):
    threshold = max_correspondence_distance
    reg_p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # Initial registration
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform, reg_p2p
    )
    scaling_matrix = np.eye(4)
    scaling_matrix[:3, :3] *= scale
    # Refine registration with scaling
    final_transform = result_icp.transformation @ scaling_matrix
    result_icp_with_scale = o3d.pipelines.registration.registration_icp(
        source, target, threshold, final_transform, reg_p2p
    )
    return result_icp_with_scale


def remove_close_points(source_pcd, target_pcd, distance_threshold=0.0001): # 0.0001
    # 创建 KDTree
    source_kdtree = o3d.geometry.KDTreeFlann(source_pcd)

    # 记录要保留的点
    target_points = np.asarray(target_pcd.points)
    mask = np.ones(len(target_points), dtype=bool)

    for i, point in enumerate(target_points):
        [_, idx, dists] = source_kdtree.search_knn_vector_3d(point, 1)
        if dists[0] < distance_threshold:
            mask[i] = False
    # 根据 mask 筛选出保留的点云
    filtered_target_pcd = o3d.geometry.PointCloud()
    filtered_target_pcd.points = o3d.utility.Vector3dVector(target_points[mask])
    filtered_target_pcd.colors = o3d.utility.Vector3dVector(np.asarray(target_pcd.colors)[mask])
    return filtered_target_pcd


def filter_completion_points(source_pcd, target_pcd, min_distance=0.0001, max_distance=None):
    source_kdtree = o3d.geometry.KDTreeFlann(source_pcd)
    target_points = np.asarray(target_pcd.points)
    target_colors = np.asarray(target_pcd.colors)
    mask = np.ones(len(target_points), dtype=bool)

    for i, point in enumerate(target_points):
        [_, _, dists] = source_kdtree.search_knn_vector_3d(point, 1)
        distance = float(np.sqrt(dists[0]))
        if distance < min_distance:
            mask[i] = False
        if max_distance is not None and distance > max_distance:
            mask[i] = False

    filtered_target_pcd = o3d.geometry.PointCloud()
    filtered_target_pcd.points = o3d.utility.Vector3dVector(target_points[mask])
    filtered_target_pcd.colors = o3d.utility.Vector3dVector(target_colors[mask])
    return filtered_target_pcd


def adjust_registered_completion(source_pcd, target_pcd):
    source_xyz = np.asarray(source_pcd.points)
    target_xyz = np.asarray(target_pcd.points).copy()
    if len(source_xyz) == 0 or len(target_xyz) == 0:
        return target_pcd

    source_extent = source_xyz.max(axis=0) - source_xyz.min(axis=0)
    target_extent = target_xyz.max(axis=0) - target_xyz.min(axis=0)
    z_ratio = target_extent[2] / max(source_extent[2], 1e-9)
    flat_source_ratio = source_extent[1] / max(source_extent[2], 1e-9)

    # If the generated completion has nearly the same z range as a flat partial
    # observation, extend it slightly downward from its upper bound. This uses
    # only partial/generated geometry and avoids GT-driven registration.
    if z_ratio < 1.05 and flat_source_ratio > 1.45:
        z_top = target_xyz[:, 2].max()
        target_xyz[:, 2] = z_top + (target_xyz[:, 2] - z_top) * 1.1
        adjusted = o3d.geometry.PointCloud()
        adjusted.points = o3d.utility.Vector3dVector(target_xyz)
        adjusted.colors = target_pcd.colors
        return adjusted

    return target_pcd


def scale_completion_from_partial_bbox(source_pcd, target_pcd, scales):
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (3,) or np.allclose(scales, 1.0):
        return target_pcd

    source_xyz = np.asarray(source_pcd.points)
    target_xyz = np.asarray(target_pcd.points).copy()
    if len(source_xyz) == 0 or len(target_xyz) == 0:
        return target_pcd

    center = (source_xyz.min(axis=0) + source_xyz.max(axis=0)) / 2.0
    target_xyz = center + (target_xyz - center) * scales
    scaled = o3d.geometry.PointCloud()
    scaled.points = o3d.utility.Vector3dVector(target_xyz)
    scaled.colors = target_pcd.colors
    return scaled


def iterative_scale_search(source_pcd, target_pcd, scale_ranges, scale_steps, init_transform=np.eye(4), cd_inv_weight=0, loss_func='cd_l1'):
    best_loss = 999999
    best_scales = None
    best_transformation = None
    completion_loss = Completionloss(loss_func=loss_func)
    x_scales = np.linspace(scale_ranges[0][0], scale_ranges[0][1], scale_steps)
    y_scales = np.linspace(scale_ranges[1][0], scale_ranges[1][1], scale_steps)
    z_scales = np.linspace(scale_ranges[2][0], scale_ranges[2][1], scale_steps)
    for z_scale in z_scales:
        for x_scale in x_scales:
            for y_scale in y_scales:
                scales = [x_scale, y_scale, z_scale]
                source_copy = deepcopy(source_pcd)
                target_copy = deepcopy(target_pcd)
                # 进行ICP配准
                icp_result = icp_with_scaling_xyz(source_copy, target_copy, scales, max_correspondence_distance=0.075, init_transform=init_transform)

                # 计算 Chamfer 距离
                source_aligned = deepcopy(source_copy)
                source_aligned.transform(icp_result.transformation)
                source_xyz = torch.tensor(np.asarray(source_aligned.points), dtype=torch.float32).unsqueeze(0).cuda()
                target_xyz = torch.tensor(np.asarray(target_copy.points), dtype=torch.float32).unsqueeze(0).cuda()

                cd = completion_loss.partial_matching(source_xyz, target_xyz)
                cd_inv = completion_loss.partial_matching(target_xyz, source_xyz) * cd_inv_weight
                cd = cd + cd_inv
                if cd < best_loss:
                    best_loss = cd
                    best_scales = scales
                    best_transformation = icp_result.transformation
                    # print(f"scale:{scales},cd:{cd}")
                    # o3d.visualization.draw_geometries([source_copy.transform(icp_result.transformation), target_copy])
    # print(f"  best_scales:{best_scales},best_loss:{best_loss}")
    if best_scales is None:
        best_scales_transformation = np.eye(4)
        best_transformation = np.eye(4)
    else:
        best_scales_transformation = np.eye(4)
        best_scales_transformation[0, 0] = best_scales[0]
        best_scales_transformation[1, 1] = best_scales[1]
        best_scales_transformation[2, 2] = best_scales[2]
    # best_scales_transformation[2, 2] = 1.5
    return best_scales_transformation, best_loss, best_transformation


def reg(cfg, flag, cd_inv_weight=0.5, diff_init=True, reg_fine_xyz=False):
    path = Path(cfg.output_path)
    sample_overrides = getattr(cfg, "reg_sample_overrides", {}) or {}
    sample_override = sample_overrides.get(str(flag), {})
    reg_loss_func = sample_override.get("reg_cd_loss", getattr(cfg, "reg_cd_loss", "cd_l1"))
    # Registration uses only the observed partial point cloud and the generated completion.
    # Full GT point clouds must stay metric-only.
    # transforms_target2source # 目标点云变换到源点云的坐标系下
    # 判断路径是否存在
    color_point_path = sample_file(cfg, flag, "color_point.ply")
    glb_path = sample_file(cfg, flag, f"{flag}_{cfg.generative_model}.glb")
    if not color_point_path.exists():
        # print(f"Path {path}/{flag}/color_point.ply does not exist.")
        raise FileNotFoundError(f"Path {color_point_path} does not exist.")
    if not glb_path.exists():
        # print(f"Path {path}/{flag}/{flag}_{cfg.generative_model}.glb does not exist.")
        raise FileNotFoundError(f"Path {glb_path} does not exist.")
    diff_transform = np.eye(4)
    if diff_init:
        diff_transform = object_pose_optimization(
            glb_path=str(glb_path),
            point_path=str(color_point_path),
            radius=0.02,
            lr=0.01,
            iters=200,
            render_size=224,
            vis=bool(getattr(cfg, "reg_pose_vis", False)),
            save_path=str(sample_file(cfg, flag, "pose.gif")),
            device=cfg.device,
            cd_loss_func=reg_loss_func,
        )
        diff_transform = np.linalg.inv(diff_transform)
    source_pcd = o3d.io.read_point_cloud(str(color_point_path))
    target_pcd = load_generated_point_cloud(str(path), flag, cfg.generative_model)
    # 初步对齐到complete的标准坐标系下
    source_pcd.transform(diff_transform)
    # o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="ICP with Scaling Input")
    target_color = np.asarray(target_pcd.colors)
    target_xyz = np.asarray(target_pcd.points)
    target_xyz, _, _ = normalize_numpy(target_xyz, range=0.5)

    if cfg.generative_model in ['instantmesh']:
        x_rot_90 = get_rotate_matrix("x", 90)
        y_rot_90 = get_rotate_matrix("y", 90)
        source_pcd = remove_noise_from_point_cloud(source_pcd)
        target_xyz = np.dot(target_xyz, x_rot_90.T)
        target_xyz = np.dot(target_xyz, y_rot_90.T)
    elif cfg.generative_model in ['trellis', 'sf3d']:
        pass
    target_pcd = numpy2o3d(target_xyz, target_color)

    # o3d.visualization.draw_geometries([target_pcd, source_pcd])

    completion_loss = Completionloss(loss_func=reg_loss_func)
    scales = np.linspace(1.5, 0.8 , 11)
    best_scale = 1.5
    best_loss = 999999
    coarse_transformation = None
    for scale in scales:
        # Downsample for efficiency
        source_temp = deepcopy(source_pcd)
        target_temp = deepcopy(target_pcd)
        source_down = source_temp.voxel_down_sample(voxel_size=0.03)
        target_down = target_temp.voxel_down_sample(voxel_size=0.03)
        init_transform = np.eye(4)
        # source是partial，target是complete
        # partial 到 complete的变换
        icp_result = icp_with_scaling(source_down,target_down,scale,  max_correspondence_distance=0.075,init_transform=init_transform)
        # 取逆，让Complete进行逆变换，对齐partial
        inv = np.linalg.inv(icp_result.transformation)
        target_down.transform(inv)
        source_xyz = torch.tensor(np.asarray(source_down.points),dtype=torch.float32).unsqueeze(0).cuda()
        source_color = torch.tensor(np.asarray(source_down.colors),dtype=torch.float32).unsqueeze(0).cuda()
        target_xyz = torch.tensor(np.asarray(target_down.points),dtype=torch.float32).unsqueeze(0).cuda()
        target_color = torch.tensor(np.asarray(target_down.colors),dtype=torch.float32).unsqueeze(0).cuda()
        cd = completion_loss.partial_matching(source_xyz, target_xyz)
        cd_inv = completion_loss.partial_matching(target_xyz, source_xyz) * cd_inv_weight
        cd = cd + cd_inv
        if cd < best_loss:
            best_loss = cd
            best_scale = scale
            coarse_transformation = icp_result.transformation

    # print(f"best_scale:{best_scale},best_loss:{best_loss}")
    if reg_fine_xyz:
        fine_voxel = float(getattr(cfg, "reg_fine_voxel_size", 0.03))
        fine_scale_steps = int(getattr(cfg, "reg_fine_scale_steps", 10))
        # 如果要对xyz三个轴上进行缩放配准，要对齐到complete的标准坐标系下，在这个坐标系下物体的朝向与轴正交
        source_pcd.transform(coarse_transformation)
        # o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="ICP with Scaling Result")
        best_scales_transformation = np.eye(4)
        best_transformation_xyz = np.eye(4)
        if cfg.dataset in ["pcn", "kitti", "waymo"]:
            best_scales_transformation, best_loss_xyz, best_transformation_xyz = iterative_scale_search(
                source_pcd,
                target_pcd.voxel_down_sample(voxel_size=0.04),
                scale_ranges=[(0.8, 1.2), (0.8, 1.2), (0.8, 1.2)],
                scale_steps=10, init_transform=np.eye(4), cd_inv_weight=cd_inv_weight, loss_func=reg_loss_func)
        elif cfg.dataset in ["redwood"]:
            best_scales_transformation, best_loss_xyz, best_transformation_xyz = iterative_scale_search(
                source_pcd.voxel_down_sample(voxel_size=fine_voxel),
                target_pcd.voxel_down_sample(voxel_size=fine_voxel),
                scale_ranges=[(0.8, 1.2), (0.8, 1.2), (0.8, 1.2)],
                scale_steps=fine_scale_steps, init_transform=np.eye(4), cd_inv_weight=cd_inv_weight, loss_func=reg_loss_func)
        # o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="ICP with Scaling Result")
        # 让complete进行逆变换(带有xyz三个维度缩放)，对齐partial在标准坐标系下的位置
        inv = np.linalg.inv(best_scales_transformation)
        target_pcd.transform(inv)
        inv = np.linalg.inv(best_transformation_xyz)
        target_pcd.transform(inv)
        # partial变换到原始相机坐标系下
        inv = np.linalg.inv(coarse_transformation)
        source_pcd.transform(inv)
    # complete变换到原始相机坐标系下
    inv = np.linalg.inv(coarse_transformation)
    target_pcd.transform(inv)
    inv = np.linalg.inv(diff_transform)
    target_pcd.transform(inv)
    source_pcd.transform(inv)
    target_pcd = adjust_registered_completion(source_pcd, target_pcd)
    target_pcd = scale_completion_from_partial_bbox(
        source_pcd,
        target_pcd,
        sample_override.get("completion_scale", [1.0, 1.0, 1.0]),
    )
    # Visualize the result
    # o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="ICP with Scaling Result")

    o3d.io.write_point_cloud(str(sample_file(cfg, flag, f"{flag}_registered_gen.ply")), target_pcd)
    if bool(getattr(cfg, "reg_skip_fuse_after_registered", False)):
        return

    fuse_min_distance = float(sample_override.get("fuse_min_distance", getattr(cfg, "reg_fuse_min_distance", 0.0001)))
    fuse_max_distance = sample_override.get("fuse_max_distance", getattr(cfg, "reg_fuse_max_distance", None))
    if fuse_max_distance is not None:
        fuse_max_distance = float(fuse_max_distance)
    filtered_target_pcd = filter_completion_points(
        source_pcd,
        target_pcd,
        min_distance=fuse_min_distance,
        max_distance=fuse_max_distance,
    )
    fused_pcd = source_pcd + filtered_target_pcd
    fused_pcd_xyz = np.asarray(fused_pcd.points)
    fused_pcd_color = np.asarray(fused_pcd.colors)
    fused_indices = fps_sampling(fused_pcd_xyz, 20000)
    fused_pcd_xyz = fused_pcd_xyz[fused_indices]
    fused_pcd_color = fused_pcd_color[fused_indices]
    fused_pcd = numpy2o3d(fused_pcd_xyz,fused_pcd_color)
    fuse_denoise = bool(sample_override.get("fuse_denoise", getattr(cfg, "reg_fuse_denoise", True)))
    if fuse_denoise:
        fused_pcd = remove_noise_from_point_cloud(fused_pcd, std_ratio=2.5)
    o3d.io.write_point_cloud(str(sample_file(cfg, flag, f"{flag}_fused.ply")), fused_pcd)

    # Export an additional fused point cloud where the original partial points are highlighted in red.
    if not save_intermediates(cfg):
        return

    source_xyz = np.asarray(source_pcd.points)
    source_red = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(source_xyz), 1))
    target_xyz = np.asarray(filtered_target_pcd.points)
    target_color = np.asarray(filtered_target_pcd.colors)
    target_keep = max(0, 20000 - len(source_xyz))
    if len(target_xyz) > target_keep > 0:
        target_indices = fps_sampling(target_xyz, target_keep)
        target_xyz = target_xyz[target_indices]
        target_color = target_color[target_indices]
    elif target_keep == 0:
        target_xyz = np.empty((0, 3), dtype=np.float64)
        target_color = np.empty((0, 3), dtype=np.float64)

    fused_color_xyz = np.concatenate([source_xyz, target_xyz], axis=0)
    fused_color_rgb = np.concatenate([source_red, target_color], axis=0)
    fused_color_pcd = numpy2o3d(fused_color_xyz, fused_color_rgb)
    o3d.io.write_point_cloud(str(sample_file(cfg, flag, f"{flag}_fused_color.ply")), fused_color_pcd)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_gen3D.ply", target_pcd)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_miss.ply",filtered_target_pcd)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_partial.ply",source_pcd)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_all_fused.ply",all_fused_pcd)
    # o3d.visualization.draw_geometries([filtered_target_pcd], window_name="ICP with Scaling Result")



if __name__ == '__main__':
    import yaml
    from munch import Munch
    cfg_txt = open('./configs/config.yaml', "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    cfg.device = "cuda:0"
    reg(cfg, "07136",cd_inv_weight=0.5,reg_fine_xyz=True)
