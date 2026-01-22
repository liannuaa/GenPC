import open3d as o3d
import numpy as np
from utils.dataUtils import *
from copy import deepcopy
from utils.loss_util import *
from optim_registration.diff_obj_pose import object_pose_optimization
from fpsample import fps_sampling

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


def iterative_scale_search(source_pcd, target_pcd, scale_ranges, scale_steps, init_transform=np.eye(4), cd_inv_weight=0):
    best_loss = 999999
    best_scales = None
    best_transformation = None
    completion_loss = Completionloss(loss_func='cd_l1')
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
                source_xyz = torch.tensor(np.asarray(source_copy.points), dtype=torch.float32).unsqueeze(0).cuda()
                target_xyz = torch.tensor(np.asarray(target_copy.points), dtype=torch.float32).unsqueeze(0).cuda()

                cd = completion_loss.chamfer_partial_l1(source_xyz, target_xyz)
                cd_inv = completion_loss.chamfer_partial_l1(target_xyz, source_xyz) * cd_inv_weight
                cd = cd + cd_inv
                if cd < best_loss:
                    best_loss = cd
                    best_scales = scales
                    best_transformation = icp_result.transformation
                    # print(f"scale:{scales},cd:{cd}")
                    # o3d.visualization.draw_geometries([source_copy.transform(icp_result.transformation), target_copy])
    print(f"  best_scales:{best_scales},best_loss:{best_loss}")
    best_scales_transformation = np.eye(4)
    best_scales_transformation[0, 0] = best_scales[0]
    best_scales_transformation[1, 1] = best_scales[1]
    best_scales_transformation[2, 2] = best_scales[2]
    # best_scales_transformation[2, 2] = 1.5
    return best_scales_transformation, best_loss, best_transformation


def reg(cfg, flag, cd_inv_weight=0.5, diff_init=True, reg_fine_xyz=False):
    path = cfg.output_path
    # transforms_target2source # 目标点云变换到源点云的坐标系下
    # 判断路径是否存在
    if not os.path.exists(f"{path}/{flag}/color_point.ply"):
        print(f"Path {path}/{flag}/color_point.ply does not exist.")
        raise FileNotFoundError(f"Path {path}/{flag}/color_point.ply does not exist.")
    if not os.path.exists(f"{path}/{flag}/{flag}_{cfg.generative_model}.glb"):
        print(f"Path {path}/{flag}/{flag}_{cfg.generative_model}.glb does not exist.")
        raise FileNotFoundError(f"Path {path}/{flag}/{flag}_{cfg.generative_model}.glb does not exist.")
    if diff_init:
        # 使用diff_camera_pose优化相机位姿
        diff_transform = object_pose_optimization(
            glb_path=f"{path}/{flag}/{flag}_{cfg.generative_model}.glb",
            point_path=f"{path}/{flag}/color_point.ply",
            # ref_img_path=f"{path}/{flag}/img_sam.png",
            radius=0.02,
            lr=0.01,
            iters=200,
            render_size=224,
            vis=True,
            device=cfg.device
        )
        diff_transform = np.linalg.inv(diff_transform)
        # print(f"transform from diff_camera_pose: {diff_transform}")
    source_pcd = o3d.io.read_point_cloud(f"{path}/{flag}/color_point.ply")
    target_pcd = glb2point(f"{path}/{flag}/{flag}_{cfg.generative_model}.glb", num_points=163840)
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

    completion_loss = Completionloss(loss_func='cd_l1')
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
        cd = completion_loss.chamfer_partial_l1(source_xyz, target_xyz)
        cd_inv = completion_loss.chamfer_partial_l1(target_xyz, source_xyz) * cd_inv_weight
        cd = cd + cd_inv
        if cd < best_loss:
            best_loss = cd
            best_scale = scale
            coarse_transformation = icp_result.transformation

    # print(f"best_scale:{best_scale},best_loss:{best_loss}")
    if reg_fine_xyz:
        # 如果要对xyz三个轴上进行缩放配准，要对齐到complete的标准坐标系下，在这个坐标系下物体的朝向与轴正交
        source_pcd.transform(coarse_transformation)
        # o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="ICP with Scaling Result")
        if cfg.dataset in ["pcn","kitti"]:
            best_scales_transformation, best_loss_xyz, best_transformation_xyz = iterative_scale_search(
                source_pcd,
                target_pcd.voxel_down_sample(voxel_size=0.04),
                scale_ranges=[(0.8, 1.2), (0.8, 1.2), (0.8, 1.2)],
                scale_steps=10, init_transform=np.eye(4), cd_inv_weight=cd_inv_weight)
        elif cfg.dataset in ["redwood"]:
            best_scales_transformation, best_loss_xyz, best_transformation_xyz = iterative_scale_search(
                source_pcd.voxel_down_sample(voxel_size=0.03),
                target_pcd.voxel_down_sample(voxel_size=0.03),
                scale_ranges=[(0.8, 1.2), (0.8, 1.2), (0.8, 1.2)],
                scale_steps=10, init_transform=np.eye(4), cd_inv_weight=cd_inv_weight)
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
    # Visualize the result
    # o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="ICP with Scaling Result")

    filtered_target_pcd = remove_close_points(source_pcd, target_pcd, distance_threshold=0.0001)
    fused_pcd = source_pcd + filtered_target_pcd
    all_fused_pcd = source_pcd + target_pcd
    fused_pcd_xyz = np.asarray(fused_pcd.points)
    fused_pcd_color = np.asarray(fused_pcd.colors)
    fused_indices = fps_sampling(fused_pcd_xyz, 20000)
    fused_pcd_xyz = fused_pcd_xyz[fused_indices]
    fused_pcd_color = fused_pcd_color[fused_indices]
    fused_pcd = numpy2o3d(fused_pcd_xyz,fused_pcd_color)
    fused_pcd = remove_noise_from_point_cloud(fused_pcd, std_ratio=2.5)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_gen3D.ply", target_pcd)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_miss.ply",filtered_target_pcd)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_partial.ply",source_pcd)
    o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_fused.ply",fused_pcd)
    # o3d.io.write_point_cloud(f"{path}/{flag}/{flag}_all_fused.ply",all_fused_pcd)
    # o3d.visualization.draw_geometries([filtered_target_pcd], window_name="ICP with Scaling Result")



if __name__ == '__main__':
    import yaml
    from munch import Munch
    cfg_txt = open('./configs/config.yaml', "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    cfg.device = "cuda:0"
    reg(cfg, "01184",cd_inv_weight=0.5,reg_fine_xyz=True)