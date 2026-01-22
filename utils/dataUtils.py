import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import torch
import PIL
import trimesh
import os
from torchvision import transforms

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]
def visualize_point_cloud(coords):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.visualization.draw_geometries([pcd])

def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5

def get_elevation_angle(flag,dataset="redwood"):
    elevation = 0
    if dataset == "redwood":
        id = getID(flag)
        world_plane = np.load(f"data/world_planes/{id}.npy")
        a, b, c, d = world_plane[:4]
        normal = np.array([a, b, c])
        y_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(normal, y_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_theta = np.dot(normal, y_axis) / (np.linalg.norm(normal) * np.linalg.norm(y_axis))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        elevation = float(180 - np.degrees(theta))
    elif dataset in ["pcn","kitti"]:
        view = np.load(f'workspace/{flag}/view.npy')
        x, y, z = view[0], view[1], view[2]
        radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta_rad = np.arcsin(y / radius)
        theta_deg = np.degrees(theta_rad)
        elevation = float(theta_deg)
    return elevation

def group_image6(photos, ids, save_flag="group_depth"):
    """
    合并6张图片并保存为一个图像文件。

    参数:
    - photos: numpy数组，形状为[6, 3, 256, 256]，数值范围为0-1。
    - ids: 每张图片的标签，长度为6的字符串列表。
    - save_path: 保存的文件路径，默认为"group_image.png"。
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 创建2行3列的子图
    fig.suptitle("Grouped Images", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        image = photos[i].transpose(1, 2, 0)  # 转置到[256, 256, 3]以便显示
        ax.imshow(image)
        ax.set_title(ids[i])  # 设置标题为对应id
        ax.axis("off")  # 隐藏坐标轴
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以容纳标题
    plt.savefig(f'workspace/{save_flag}_{ids[0]}')

# 4：给初始点云增加点
def random_add_points(coords):
    pcd_by3d = o3d.geometry.PointCloud()
    pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
    bbox = pcd_by3d.get_axis_aligned_bounding_box()
    np.random.seed(0)
    num_points = 100000
    points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))
    kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)
    points_inside = []
    for point in points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
        if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
            points_inside.append(point)

    all_coords = np.array(points_inside)
    all_coords = np.concatenate([all_coords,coords],axis=0)
    return all_coords


# 生成插值点（可以根据需要生成新的插值点位置）
def generate_interpolation_points(points, num_points=10000):
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    new_points = np.random.uniform(min_coords, max_coords, (num_points, 3))
    return new_points


# 线性插值函数
def linear_interpolation(points, new_points, k=2):
    tree = KDTree(points)
    distances, indices = tree.query(new_points, k=k)
    weights = 1 / (distances + 1e-8)
    weights /= weights.sum(axis=1)[:, np.newaxis]
    interpolated_points = np.sum(points[indices] * weights[:, :, np.newaxis], axis=1)
    return interpolated_points


def xyz2xyzrgb(partial_path, add_point_type='random', add_num_points=5000):
    point_cloud = o3d.io.read_point_cloud(partial_path)
    print(len(point_cloud.points))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    # visualize_point_cloud(points)
    if add_point_type == 'random':
        coords= random_add_points(points)
    elif add_point_type == 'linear':
        add_points = generate_interpolation_points(points,num_points=add_num_points)
        coords = linear_interpolation(points,add_points,k=5)
    else:
        coords = points
    # 点数量
    # visualize_point_cloud(coords)
    num_pts = len(coords)
    print(num_pts)
    shs = np.random.random((num_pts, 3)) / 255.0
    rgb = SH2RGB(shs)
    return coords,rgb

def getRandomColor(num_points):
    shs = np.random.random((num_points, 3)) / 255.0
    rgb = SH2RGB(shs)
    return rgb

def save_ply_xyzrgb(coords,rgb,save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(save_path, pcd)

def save_ply_xyz(coords,save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.io.write_point_cloud(save_path, pcd)


def load_xyz(path, down_sample=None):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float32)
    # 降采样（如果需要）
    if down_sample:
        pcd = pcd.voxel_down_sample(voxel_size=down_sample)
        points = np.asarray(pcd.points, dtype=np.float32)
    # 颜色处理
    has_valid_color = pcd.has_colors() and not np.allclose(pcd.colors, 0)
    if has_valid_color:
        colors = np.asarray(pcd.colors, dtype=np.float32)
    else:
        colors = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0) + 1e-8)
        colors = np.clip(colors, 0, 1)
        range_vals = points.max(axis=0) - points.min(axis=0)
    return points, colors

def glb2obj(glb_path, obj_path=None):
    import trimesh
    

def glb2ply(glb_path, ply_path=None):
    import trimesh
    """
    Convert a GLB file to PLY format using trimesh.
    Parameters:
    glb_path (str): Path to the input GLB file.
    ply_path (str): Path to save the output PLY file.
    """
    mesh = trimesh.load_mesh(glb_path)

    # If it's a Scene, combine all meshes into one
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Convert texture to vertex colors (bake the texture)
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        mesh.visual = mesh.visual.to_color()
    # Export the mesh to PLY format
    if ply_path is not None:
        mesh.export(ply_path, file_type='ply')
        print(f"Exported {glb_path} to {ply_path}")
    return mesh

def glb2point(glb_path, down_sample=None, num_points=16384):
    print(glb_path)
    mesh = trimesh.load(glb_path, file_type='glb')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Convert texture to vertex colors (bake the texture)
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        mesh.visual = mesh.visual.to_color()
    # Sample points from the mesh surface, 带颜色
    pts, face_idx = mesh.sample(num_points, return_index=True)

    # 获取顶点颜色
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # 转换为0-1范围
    else:
        # 如果没有颜色信息，使用默认颜色
        vertex_colors = np.ones((len(mesh.vertices), 3)) * 0.5
    
    # 通过面索引获取每个采样点对应的三角形的顶点颜色
    face_colors = vertex_colors[mesh.faces[face_idx]]  # (n_pts, 3, 3)
    
    # 计算重心坐标
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_idx], pts)
    
    # 使用重心坐标插值颜色
    color = np.sum(face_colors * bary[:, :, np.newaxis], axis=1)  # (n_pts, 3)
    color = np.clip(color, 0, 1)  # 确保颜色值在有效范围内
    # 返回o3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(color)
    if down_sample:
        pcd = pcd.voxel_down_sample(voxel_size=down_sample)
    return pcd


def smooth_depth(flag):
    # 读取深度图
    depth_image = cv2.imread(f'workspace/{flag}/depth.png', cv2.IMREAD_GRAYSCALE)
    # 使用高斯滤波器
    smoothed_depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
    # 保存平滑后的深度图
    cv2.imwrite(f'workspace/{flag}/smoothed_depth.png', smoothed_depth_image)

def convex():
    pcd = o3d.io.read_point_cloud("../data/car.ply")  # 读取ply或者pcd文件
    # # 裁剪点云
    # vol = o3d.visualization.read_selection_polygon_volume(
    #     "../test_data/Crop/cropped.json")
    # chair = vol.crop_point_cloud(pcd)

    # 计算点云的凸包
    hull, _ = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))  # 凸包的颜色
    o3d.visualization.draw_geometries([pcd, hull_ls])


def registration(source,target):
    # 可视化初始点云
    o3d.visualization.draw_geometries([source, target], window_name="Initial Alignment", point_show_normal=False)
    # 设置初始变换矩阵为单位矩阵
    trans_init = np.eye(4)
    # 设置 ICP 算法的距离阈值
    threshold = 10
    # 使用点对点 ICP 进行配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    # 输出变换矩阵
    print("Transformation Matrix:")
    print(reg_p2p.transformation)
    # 将源点云应用变换矩阵
    source.transform(reg_p2p.transformation)
    # 可视化对齐后的点云
    o3d.visualization.draw_geometries([source, target], window_name="Aligned Point Clouds", point_show_normal=False)

def numpy2o3d(points,colors=None):
    '''
    points: numpy array
    return: open3d.geometry.PointCloud
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def o3d2numpy(pcd):
    '''
    pcd: open3d.geometry.PointCloud
    return: numpy array
    '''
    return np.asarray(pcd.points)

def get_view_points(xyz, view_point, threshold=10000):
    """
    xyz:numpy array
    view_point: numpy array
    Returns:
    """
    # 判断类型,如果是o3d则转换为numpy
    if isinstance(xyz, o3d.geometry.PointCloud):
        xyz = np.asarray(xyz.points)
    # 如果是torch则转换为numpy
    elif isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    point_visibility = np.zeros(xyz.shape[0], dtype=bool)
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz))
    o3d_camera = np.array(view_point)
    _, pt_map = pcd.hidden_point_removal(o3d_camera, threshold)
    visible_point_ids = np.array(pt_map)
    point_visibility[visible_point_ids] = True
    return point_visibility


def fibonacci_sphere(samples, radius=2):
    """
    使用Fibonacci球生成均匀分布在球面上的视角点。

    Args:
    samples (int): 要生成的视角数量。
    radius (float): 视角与物体中心的距离（球的半径）。

    Returns:
    numpy array: 视角坐标数组，每一行是一个视角点 (x, y, z)。
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_y = math.sqrt(1 - y*y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius_y * radius
        z = math.sin(theta) * radius_y * radius
        y = y * radius

        points.append((x, y, z))

    return np.array(points)

def get_surface_point(xyz, threshold=500, radius=2, samples=256):
    """
    从多个视角获取物体表面的可见点的并集。
    Args:
    xyz (numpy array or Open3D PointCloud): 点云数据。
    threshold (float): 可见性检查的阈值。
    radius (float): 视角与物体中心的距离。
    samples (int): 生成的视角数量。

    Returns:
    numpy array: 布尔数组，表示每个点在多个视角下是否可见。
    """
    # 使用Fibonacci球生成视角
    view_points = fibonacci_sphere(samples, radius)

    # 判断类型，如果是Open3D点云则转换为numpy
    if isinstance(xyz, o3d.geometry.PointCloud):
        xyz = np.asarray(xyz.points)
    elif isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()

    # 初始化可见性布尔数组
    point_visibility = np.zeros(xyz.shape[0], dtype=bool)

    # 调用get_view_points来计算每个视角下的可见点并取并集
    for view_point in view_points:
        visible_points = get_view_points(xyz, view_point, threshold)
        point_visibility |= visible_points  # 将每个视角的可见点并入总结果
    return point_visibility


def color_point(flag,img,point_uv,res=1024,dataset="redwood",view=None):
    id = getID(flag) if dataset=='redwood' else flag
    if dataset == "redwood":
        xyz = load_xyz(f"data/{id}.ply")
        GT_xyz = load_xyz(f"data/GT/{id}.ply")
    elif dataset == "kitti":
        cam = torch.load(f'workspace/{flag}/cam.pth')
        # print(cam)
        xyz = load_xyz(f"data/kitti/{flag}.pcd")
        pcd = numpy2o3d(xyz)
        center = pcd.get_center()
        xyz = cam.extrinsics.transform(torch.tensor(xyz).cuda())
        xyz = xyz.squeeze(0).detach().cpu().numpy()
        # 移到原点
        transformed_pcd = numpy2o3d(xyz)
        transformed_center = transformed_pcd.get_center()
        center_detla = center - transformed_center
        xyz = xyz + center_detla
    elif dataset == "pcn":
        cam = torch.load(f'workspace/{flag}/cam.pth')
        category = flag.split("_")[0]
        id = flag.split("_")[1]
        xyz = load_xyz(f"data/pcn/partial/{category}/{id}.pcd")
        GT_xyz = load_xyz(f"data/pcn/complete/{category}/{id}.pcd")
        extr_matrix = cam.extrinsics.view_matrix().cpu().numpy()
        extr_matrix_inv = cam.extrinsics.inv_view_matrix().cpu().numpy()
        rotation_matrix = extr_matrix_inv[:3, :3]
        # 只应用旋转，不使用平移
        xyz = np.dot(xyz, rotation_matrix.T)[:,:3]  # 进行旋转变换
        GT_xyz = np.dot(GT_xyz, rotation_matrix.T)[:,:3]  # 进行旋转变换
    # 如果point_uv是numpy则转换为torch
    if isinstance(point_uv, np.ndarray):
        point_uv = torch.tensor(point_uv).to('cuda')
    if dataset in ['pcn','kitti']:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    else:
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    image = transforms.ToTensor()(img).to("cuda")
    image_np = image.detach().cpu().numpy()

    point_pixel = point_uv * res  # [piont_num,2]
    point_pixel = point_pixel.long()
    point_pixel = torch.cat((point_pixel[:, 1].unsqueeze(-1), point_pixel[:, 0].unsqueeze(-1)), dim=-1)
    point_pixel = point_pixel.clip(0, res - 1)

    colors = np.zeros_like(xyz)
    for i, (x, y) in enumerate(point_pixel.detach().cpu().numpy()):
        colors[i] = image_np[:, x, y]
    angle = np.pi  # 180度
    up_down_flip = np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
    if dataset == "redwood":
        xyz = np.dot(xyz, up_down_flip.T)
        GT_xyz = np.dot(GT_xyz, up_down_flip.T)
    if dataset in ['redwood','pcn']:
        o3d.io.write_point_cloud(f"workspace/{flag}/GT.ply", numpy2o3d(GT_xyz))
    save_ply_xyzrgb(xyz, colors, f"workspace/{flag}/color_point.ply")




def get_rotate_matrix(axis,angle):
    pi = np.pi
    angle = angle * pi / 180
    if axis == "x":
        return np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    elif axis == "y":
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == "z":
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    else:
        raise ValueError("axis should be x,y,z")

def obj2point(mesh,point_num=10000):
    # 提取坐标
    coordinates = np.asarray(mesh.vertices)
    # 检查 mesh 是否包含颜色信息
    if hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0:
        # 提取颜色
        colors = np.asarray(mesh.vertex_colors)
    else:
        # 如果 mesh 不包含颜色信息，创建一个全零数组作为颜色
        print("No color information found in the mesh.")
        colors = np.zeros_like(coordinates)
    return coordinates, colors

def get_arrow(start, end):
    direction = end - start
    length = np.linalg.norm(direction)
    direction_normalized = direction / length
    # 创建箭头
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.05,
        cylinder_height=0.8 * length,  # 调整箭头的大小与方向长度一致
        cone_height=0.2 * length
    )
    # 计算旋转矩阵，使箭头指向目标点
    z_axis = np.array([0, 0, 1])  # 原始箭头方向是沿着Z轴
    axis = np.cross(z_axis, direction_normalized)  # 旋转轴
    angle = np.arccos(np.dot(z_axis, direction_normalized))  # 旋转角度
    # 处理可能的数值问题
    if np.linalg.norm(axis) < 1e-6:  # 当起点和终点几乎在同一条线上时
        rotation_matrix = np.eye(3)  # 使用单位矩阵，无需旋转
    else:
        axis = axis / np.linalg.norm(axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    # 旋转箭头
    arrow.rotate(rotation_matrix)
    # 将箭头平移到起点
    arrow.translate(start)
    return arrow


def normalize_torch(xyz: torch.Tensor, range: float = 1.0) -> torch.Tensor:
    '''
    xyz: torch.tensor
    range: float - 归一化的目标范围，默认为1.0（即[-1, 1]）
    Returns: torch.tensor - 归一化后的点云数据
    '''
    # 计算最小值和最大值
    vertices_min = xyz.min(0)[0]
    vertices_max = xyz.max(0)[0]
    # 归一化到中心对称
    xyz_centered = xyz - (vertices_max + vertices_min) / 2.0
    xyz_normalized = xyz_centered / (vertices_max - vertices_min).max()
    # 根据range调整范围
    scale = range / 0.5
    xyz_normalized *= scale

    return xyz_normalized

def normalize_open3d(pcd: o3d.geometry.PointCloud, range: float = 1.0) -> o3d.geometry.PointCloud:
    """
    对 open3d 点云进行归一化处理，使其中心对称，并缩放到指定范围。
    保留颜色信息。

    Args:
        pcd (o3d.geometry.PointCloud): 输入点云（可能含颜色）。
        range (float)

    Returns:
        o3d.geometry.PointCloud: 归一化后的点云。
    """
    # 复制以防修改原始点云
    pcd_norm = o3d.geometry.PointCloud(pcd)
    # 处理坐标
    points = np.asarray(pcd_norm.points, dtype=np.float32)
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    centered = points - (min_vals + max_vals) / 2.0
    scale = (max_vals - min_vals).max()
    normalized = centered / (scale + 1e-6)
    normalized *= (range / 0.5)
    pcd_norm.points = o3d.utility.Vector3dVector(normalized)
    # 如果有颜色，则保持颜色不变
    if pcd.has_colors():
        pcd_norm.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors, dtype=np.float32))
    return pcd_norm


def normalize_numpy(xyz, range=1.0):
    '''
    xyz: np.ndarray - 输入的点云数据（N x 3）
    range: float - 归一化的目标范围，默认为1.0（即[-1, 1]）
    Returns:
    - np.ndarray - 归一化后的点云数据
    - np.ndarray - 归一化之前的中心点
    - float - 归一化的比例尺度
    '''
    # 计算最小值和最大值
    vertices_min = xyz.min(axis=0)
    vertices_max = xyz.max(axis=0)
    # 归一化到中心对称
    center = (vertices_max + vertices_min) / 2.0
    xyz_centered = xyz - center
    scale_factor = (vertices_max - vertices_min).max()
    xyz_normalized = xyz_centered / scale_factor
    # 根据range调整范围
    scale = range / 0.5
    xyz_normalized *= scale
    return xyz_normalized, center, scale_factor

def getID(flag:str):
    kv={
        "car":"car",
        "Wheelie Bin":"01184",
        "chair":"05117",
        "armchair":"05452",
        "Plant vases":"06127",
        "table_base":"06145",
        "vespa":"06188",
        "Kid tricycle":"06830",
        "sofa":"07136",
        "trash can":"07306",
        "swivel chair": "09639",
        "airplane":"airplane",
        "Square table_base":"Square table_base",
    }
    return kv[flag]

def getCategory(flag:str):
    kv = {
        "01184": "Wheelie Bin",
        "05117": "chair",
        "05452": "armchair",
        "06127": "Plant vases",
        "06145": "table",
        "06188": "vespa",
        "06830": "Kid tricycle",
        "07136": "sofa",
        "07306": "trash can",
        "09639": "swivel chair",
    }
    return kv[flag]

def getPrompt(flag:str):
    kv={
        "car":"car",
        "Wheelie Bin":"a green Wheelie Bin",
        "chair":"chair",
        "armchair":"armchair",
        "Plant vases":"plant in a large vase",
        "table_base":"one leg square table_base",
        "vespa":"vespa",
        "Kid tricycle":"Children's tricycle with handle",
        "sofa":"sofa",
        "trash can":"a office trash can ",
        "swivel chair": "swivel chair with brown legs",
        "airplane":"airplane",
        "Square table_base":"Square table_base",
        "02691156":"airplane",
        "02933112":"cabinet",
        "02958343":"car",
        "03001627":"chair",
        "03636649":"lamp",
        "04256520":"sofa",
        "04379243":"table_base",
        "04530566":"vessel",
        "0kitti":"car",
        'scanchair':"chair",
        "scantable":"table_base",
        "scansofa":"sofa",
        "scancar":"car",
        "scanlamp":"lamp",

    }
    return kv[flag]




def remove_noise_from_point_cloud(pcd, nb_neighbors=20, std_ratio=1.5):
    """
    使用统计滤波方法去除点云中的噪声。
    参数:
    - pcd: open3d.geometry.PointCloud 点云对象
    - nb_neighbors: int, 考虑的邻居点数（默认20）
    - std_ratio: float, 标准差比例，用于确定一个点是否为噪声（默认2.0）
    返回:
    - denoised_pcd: open3d.geometry.PointCloud 去噪后的点云对象
    """
    # 统计滤波，移除噪声点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # 提取去噪后的点云
    denoised_pcd = pcd.select_by_index(ind)
    return denoised_pcd


# 给点云添加噪声
