import open3d as o3d
import numpy as np
import torch

from utils.dataUtils import *
from utils.loss_util import *



def metric_my_redwood(flag, cd_l1=True):
    x_180 = get_rotate_matrix("x",180)
    gt_pcd = o3d.io.read_point_cloud(f"data/GT/{flag}.ply")
    gt_pcd.rotate(x_180,center=(0,0,0))
    gt_xyz = np.asarray(gt_pcd.points)
    completion_emd = Completionloss(loss_func='emd')
    completion_cd = Completionloss(loss_func='cd_l1')
    cd =99999
    emd =99999
    if cd_l1:
        gt_torch = torch.from_numpy(gt_xyz).unsqueeze(0).float().cuda()
        # gt_torch = torch.from_numpy(gt_xyz).unsqueeze(0).float().cuda()
        # gt_torch = fps_subsample(gt_torch, 16384).float().cuda()
    else:
        gt_torch = torch.from_numpy(gt_xyz).unsqueeze(0).float().cuda()
        gt_torch = fps_subsample(gt_torch, 16384).float().cuda()
    path = os.path.join(f"workspace/{flag}/")
    for file in os.listdir(path):
        if (
                ".ply" in file and "optim" in file or
                ".ply" in file and "fused" in file or
                ".ply" in file and "mesh" in file or
                ".ply" in file and "denoise" in file
        ):
            pcd = o3d.io.read_point_cloud(path+file)
            pcd_xyz = np.asarray(pcd.points)
            pcd_torch = torch.from_numpy(pcd_xyz).unsqueeze(0).float().cuda()
            pcd_torch = fps_subsample(pcd_torch,16384).float().cuda()
            if cd_l1:
                losscd = completion_cd.get_loss(gen=pcd_torch, gt=gt_torch)
                print(f"{file} {getID(flag)} : {(losscd * 100):.2f}")
            else:
                losscd = completion_cd.get_loss(gen=pcd_torch, gt=gt_torch)
                lossemd = completion_emd.get_loss(gen=pcd_torch, gt=gt_torch)
                print(f"{file} {getID(flag)} : {(losscd * 100):.2f}/{(lossemd * 100):.2f}")
                if(lossemd<emd):
                    emd = lossemd
                    cd = losscd
    print(f"{getID(flag)} {(cd*100):.2f}/{(emd*100):.2f}")
def metric_sds_redwood(flag):
    gt_mesh = o3d.io.read_triangle_mesh(f"data/GT_big/{getID(flag)}.ply")
    gt_bbox = gt_mesh.get_axis_aligned_bounding_box()
    gt_center = gt_bbox.get_center()
    gt_extent = gt_bbox.get_extent()
    # 计算GT网格的归一化尺度因子
    gt_scale_factor = 1.0 / max(gt_extent)
    gt_mesh.translate(-gt_center)  # 平移到原点
    gt_mesh.scale(gt_scale_factor, center=np.zeros(3))  # 缩放到单位立方体

    est_mesh = o3d.io.read_triangle_mesh(f"data/other_compare/redwood_sds/{getID(flag)}/validation/df_ep2000__surface.obj")
    est_mesh.translate(-gt_center)  # 平移到与GT网格对齐
    est_mesh.scale(gt_scale_factor, center=np.zeros(3))  # 使用与GT相同的缩放因子
    # o3d.visualization.draw_geometries([gt_mesh,est_mesh])

    gt_sample = gt_mesh.sample_points_uniformly(number_of_points=40000)
    est_sample = est_mesh.sample_points_uniformly(number_of_points=40000)
    vis_sample = est_mesh.sample_points_uniformly(number_of_points=16384)
    x_rot = get_rotate_matrix("x", 180)
    vis_sample.rotate(x_rot, center=(0, 0, 0))
    # 保存
    world_plane = np.load(f"data/world_planes/{getID(flag)}.npy")
    a, b, c, d = world_plane[:4]
    normal = np.array([a, b, c])
    y_axis = np.array([0, 1, 0])
    rotation_axis = np.cross(normal, y_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cos_theta = np.dot(normal, y_axis) / (np.linalg.norm(normal) * np.linalg.norm(y_axis))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    elevation = float(180 - np.degrees(theta))
    vis_sample.rotate(get_rotate_matrix("x", -elevation), center=(0, 0, 0))
    # o3d.visualization.draw_geometries([vis_sample])
    o3d.io.write_point_cloud(f"data/other_compare/redwood_sds/{getID(flag)}.ply", vis_sample)
    gt_xyz = np.asarray(gt_sample.points)
    gt_torch = torch.from_numpy(gt_xyz).unsqueeze(0).float().cuda()
    gt_torch = fps_subsample(gt_torch, 16384).float().cuda()
    est_xyz = np.asarray(est_sample.points)
    est_torch = torch.from_numpy(est_xyz).unsqueeze(0).float().cuda()
    est_torch = fps_subsample(est_torch,16384).float().cuda()
    # o3d.visualization.draw_geometries([gt_sample,est_sample])
    completion_cd = Completionloss(loss_func='cd_l1')
    completion_emd = Completionloss(loss_func='emd')
    cdloss = completion_cd.get_loss(gen=est_torch, gt=gt_torch)
    emdloss = completion_emd.get_loss(gen=est_torch, gt=gt_torch)
    # print(f"{flag} cdloss: {(cdloss*1000):.2f}, emdloss: {(emdloss*1000):.2f}")
    print(f"{getID(flag)} {flag} {(cdloss*100):.2f}/{(emdloss*100):.2f}")

def metric_pcn_deeplearning(category):
    GT_path = f"data/pcn/complete/{category}/"


import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist


def UHD(partial_path, complete_path):
    """
    计算单向 Hausdorff 距离，从 partial 点云到 complete 点云
    使用 open3d 读取点云并计算欧氏距离
    """
    # 加载点云数据
    partial_pcd = o3d.io.read_point_cloud(partial_path)
    complete_pcd = o3d.io.read_point_cloud(complete_path)

    # 获取点云的顶点
    partial_points = np.asarray(partial_pcd.points)
    complete_points = np.asarray(complete_pcd.points)
    if len(partial_points) >=20000:
        partial_points = fps_subsample(torch.from_numpy(partial_points).unsqueeze(0).float().cuda(), 10000).float().cuda()
        partial_points = partial_points.squeeze(0).cpu().numpy()
    if len(complete_points) >=20000:
        complete_points = fps_subsample(torch.from_numpy(complete_points).unsqueeze(0).float().cuda(), 20000).float().cuda()
        complete_points = complete_points.squeeze(0).cpu().numpy()
    # 计算两个点集之间的距离矩阵
    dist_matrix = cdist(partial_points, complete_points, metric='euclidean')

    # 对于 partial 点云中的每个点，找到它到 complete 点云的最近距离
    min_dist_pcd1 = np.min(dist_matrix, axis=1)  # partial 点云中每个点到 complete 点云的最小距离

    # 取这些最小距离中的最大值，作为从 partial 到 complete 的 Hausdorff 距离
    hd = np.max(min_dist_pcd1)

    return hd


def cd_emd(pcdpath1,pcdpath2):
    pcd1 = o3d.io.read_point_cloud(pcdpath1)
    pcd2 = o3d.io.read_point_cloud(pcdpath2)
    completion_cd = Completionloss(loss_func='cd_l1')
    completion_emd = Completionloss(loss_func='emd')
    pcd1_xyz = np.asarray(pcd1.points)
    pcd2_xyz = np.asarray(pcd2.points)
    pcd1_torch = torch.from_numpy(pcd1_xyz).unsqueeze(0).float().cuda()
    pcd2_torch = torch.from_numpy(pcd2_xyz).unsqueeze(0).float().cuda()
    pcd1_torch = fps_subsample(pcd1_torch, 16384).float().cuda()
    pcd2_torch = fps_subsample(pcd2_torch, 16384).float().cuda()
    cdloss = completion_cd.get_loss(gen=pcd2_torch, gt=pcd1_torch)
    emdloss = completion_emd.emd_loss(p1=pcd2_torch, p2=pcd1_torch)
    return cdloss, emdloss
def metrci_deep_redwood_emd(method):
    path = os.path.join(f'data/other_compare/redwood_{method}/')
    # x_180 = get_rotate_matrix("x", 180)

    completion_cd = Completionloss(loss_func='cd_l1')

    completion_emd = Completionloss(loss_func='emd')
    for file in os.listdir(path):
        id = file.split('.')[0]
        gt_pcd = o3d.io.read_point_cloud(f"data/redwood/test/complete/000/{id}.pcd")

        # gt_pcd.rotate(x_180, center=(0, 0, 0))
        gt_xyz = np.asarray(gt_pcd.points)
        gt_torch = torch.from_numpy(gt_xyz).unsqueeze(0).float().cuda()
        gt_torch = fps_subsample(gt_torch, 16384).float().cuda()
        pcd = o3d.io.read_point_cloud(path + file)
        pcd_xyz = np.asarray(pcd.points)
        pcd_torch = torch.from_numpy(pcd_xyz).unsqueeze(0).float().cuda()
        # o3d.visualization.draw_geometries([gt_pcd,pcd])
        # pcd_torch = FPS(pcd_torch, 16384).unsqueeze(0).float().cuda()
        cdloss = completion_cd.get_loss(gen=pcd_torch, gt=gt_torch)
        emdloss = completion_emd.get_loss(gen=pcd_torch, gt=gt_torch)
        print(f"{id}-cd-loss: {(cdloss * 100):.2f}, emd-loss: {(emdloss * 100):.2f}")

if __name__ == "__main__":
    ok_list = [ "swivel chair", "Plant vases", "armchair", "trash can", "sofa", "vespa", "Kid tricycle", "table_base", "chair", "Wheelie Bin", ]
    # ok_list = ["Plant vases", "armchair", "trash can", "sofa", "vespa", "Kid tricycle", "table_base", "chair",  "Wheelie Bin", ]
    # metric("vespa")
    metric_my_redwood("01184")
    # metric_my_redwood("trash can")
    # metric_my_redwood("vespa")
    # metric_sds_redwood("Kid tricycle")
    # metrci_deep_redwood_emd('adapointr')
    uhd = 0
    # shapeformer
    # for i in range(10):
    #     partial_path = f'workspace/zshapeformer/base/{i}_data_c_mesh.ply'
    #     complete_path = f'workspace/zshapeformer/base/{i}_s0_mesh.ply'
    #     tt = UHD(partial_path, complete_path)
    #     print(f"oneUHD: {tt*1000:.2f}")
    #     uhd += tt
    # for flag in ok_list:
    #     partial_path = f'workspace/{flag}/{flag}_partial.ply'
    #     complete_path = f'workspace/{flag}/{flag}_fused.ply'
    #     uhd += UHD(partial_path, complete_path)
    # uhd = uhd / len(ok_list)
    # print(f"UHD: {uhd*100:.2f}")
    # metric_sds_redwood("Wheelie Bin")