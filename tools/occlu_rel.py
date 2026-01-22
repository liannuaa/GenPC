import os, sys
import cv2
import torch
import numpy as np
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import json
import utils3d


# 设置导入路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MOGE_PATH = os.path.join(PROJECT_ROOT, 'models', 'MoGe')
if MOGE_PATH not in sys.path:
    sys.path.append(MOGE_PATH)

from moge.model.v2 import MoGeModel
from moge.utils.geometry_numpy import depth_occlusion_edge_numpy


class OcclusionDetector:
    """遮挡关系检测器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load_moge_model(self):
        """加载MoGe深度估计模型"""
        if self.model is None:
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal", 
                        cache_dir="/home/lian/Desktop/genpc_base/models/MoGe/weights").to(self.device)
            self.model.eval()
        return self.model
    
    def estimate_depth(self, image_path, masks=None):
        """估计深度图"""
        model = self.load_moge_model()
        
        # 加载图片
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        
        # 推理
        with torch.inference_mode():
            output = model.infer(input_tensor)
        points, depth, moge_pred_mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        normal = output['normal'].cpu().numpy() if 'normal' in output else None
        edge_mask = depth_occlusion_edge_numpy(depth, moge_pred_mask, thickness=1, tol=0.001)
        return depth, moge_pred_mask, edge_mask, points, normal, intrinsics

    def load_object_masks(self, masks_dir):
        """加载物体掩码"""
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        masks = {}
        
        for i, mask_file in enumerate(mask_files):
            mask_name = f"object_{i+1}"
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            masks[mask_name] = (mask > 128).astype(np.uint8)  # 二值化
            
        return masks
    
    def find_object_boundaries(self, masks, dilate_size=3):
        """找到物体边界并膨胀"""
        boundaries = {}
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        
        for name, mask in masks.items():
            # 边界检测
            boundary = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
            # 膨胀边界
            boundary_dilated = cv2.dilate(boundary, kernel, iterations=1)
            boundaries[name] = boundary_dilated > 0
            
        return boundaries
    
    def detect_occlusion_pairs(self, masks, boundaries, depth):
        """检测物体间的遮挡关系"""
        occlusion_graph = defaultdict(list)
        object_names = list(masks.keys())
        
        print("检测遮挡关系...")
        
        for i, obj1 in enumerate(object_names):
            for j, obj2 in enumerate(object_names):
                if i >= j:
                    continue
                
                # 检查边界接触
                boundary_contact = np.sum(boundaries[obj1] & boundaries[obj2])
                
                print(f"  检查 {obj1} vs {obj2}: 边界接触={boundary_contact}")
                
                if boundary_contact < 2:
                    print(f"    跳过: 接触不足")
                    continue
                
                # 获取深度值 - 只从非重叠区域取样避免深度混淆
                overlap_region = masks[obj1] & masks[obj2]
                
                # 从每个物体的独占区域取深度值
                exclusive_mask1 = masks[obj1] & (~overlap_region)
                exclusive_mask2 = masks[obj2] & (~overlap_region)
                
                # 如果独占区域太小，使用整个物体区域
                if np.sum(exclusive_mask1) < 10:
                    exclusive_mask1 = masks[obj1]
                if np.sum(exclusive_mask2) < 10:
                    exclusive_mask2 = masks[obj2]
                    
                depth1_values = depth[exclusive_mask1 > 0]
                depth2_values = depth[exclusive_mask2 > 0]
                
                # 过滤无效深度值
                depth1_values = depth1_values[np.isfinite(depth1_values) & (depth1_values > 0)]
                depth2_values = depth2_values[np.isfinite(depth2_values) & (depth2_values > 0)]
                
                if len(depth1_values) == 0 or len(depth2_values) == 0:
                    print(f"    跳过: 无有效深度值")
                    continue
                
                median_depth1 = np.median(depth1_values)
                median_depth2 = np.median(depth2_values)
                
                print(f"    深度: {obj1}={median_depth1:.1f}, {obj2}={median_depth2:.1f}")
                print(f"    深度值范围: {obj1}=[{depth1_values.min():.1f}-{depth1_values.max():.1f}], {obj2}=[{depth2_values.min():.1f}-{depth2_values.max():.1f}]")
                
                # 检查深度差异 - 降低阈值以检测更细微的深度变化
                if abs(median_depth1 - median_depth2) >= 0.01:  # 从0.5降低到0.3
                    # 计算遮挡分数（基于边界接触）
                    contact_score = min(boundary_contact / min(np.sum(masks[obj1]), np.sum(masks[obj2])), 1.0)
                    
                    # 设置最小遮挡分数阈值，避免检测到微小的接触
                    min_occlusion_score = 0.0001  # 最小遮挡分数阈值
                    
                    if contact_score >= min_occlusion_score:
                        if median_depth1 < median_depth2:
                            # obj1 遮挡 obj2
                            occlusion_graph[obj1].append((obj2, contact_score))
                            print(f"    ✓ {obj1} 遮挡 {obj2}, 分数: {contact_score:.3f}")
                        else:
                            # obj2 遮挡 obj1
                            occlusion_graph[obj2].append((obj1, contact_score))
                            print(f"    ✓ {obj2} 遮挡 {obj1}, 分数: {contact_score:.3f}")
                    else:
                        print(f"    跳过: 遮挡分数过低 ({contact_score:.3f} < {min_occlusion_score})")
                else:
                    print(f"    跳过: 深度差异不足 ({abs(median_depth1 - median_depth2):.1f})")
        
        return dict(occlusion_graph)

    def process_image(self, image_path, masks_dir, output_dir='.', dilate_size=10):
        """处理单张图像，检测遮挡关系"""
        print(f"处理图像: {image_path}")
        
        # 1. 深度估计
        print("正在估计深度...")
        depth, moge_pred_mask, edge_mask, points, normal, intrinsics = self.estimate_depth(image_path)

        # 2. 加载物体掩码
        print("加载物体掩码...")
        object_masks = self.load_object_masks(masks_dir)
        print(f"找到 {len(object_masks)} 个物体")
        
        # 3. 边界膨胀
        boundaries = self.find_object_boundaries(object_masks, dilate_size=dilate_size)

        # 4. 检测遮挡关系
        print("检测遮挡关系...")
        occlusion_graph = self.detect_occlusion_pairs(object_masks, boundaries, depth)
        
        # 5. 保存结果
        self.save_occl_results(image_path, depth, edge_mask, object_masks, 
                         occlusion_graph, output_dir)
        
        return occlusion_graph, depth, object_masks
    
    def save_occl_results(self, image_path, depth, edge_mask, masks, occlusion_graph, output_dir):
        """保存深度图和边缘掩码"""
        # 保存深度图
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)
        else:
            depth_norm = np.zeros_like(depth)
        
        depth_u16 = (depth_norm * 65535).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(f'{output_dir}/depth.png', depth_u16)
        
        # 保存边缘掩码
        cv2.imwrite(f'{output_dir}/edge_mask.png', edge_mask.astype(np.uint8) * 255)
        
        print(f"✓ 深度图保存: {output_dir}/depth.png")
        print(f"✓ 边缘掩码保存: {output_dir}/edge_mask.png")

    def save_moge_results(self, depth, moge_pred_mask, edge_mask, points, normal, intrinsics):
        pass

def visualize_boundaries_debug(masks, boundaries, output_path):
    """可视化物体边界用于调试"""
    fig, axes = plt.subplots(2, len(masks), figsize=(15, 8))
    if len(masks) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (name, mask) in enumerate(masks.items()):
        # 原始mask
        axes[0, i].imshow(mask, cmap='gray')
        axes[0, i].set_title(f'{name} - Original Mask')
        axes[0, i].axis('off')
        
        # 膨胀后的边界
        boundary = boundaries[name]
        axes[1, i].imshow(boundary, cmap='hot')
        axes[1, i].set_title(f'{name} - Dilated Boundary')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 边界调试图保存: {output_path}")
    plt.close()


def visualize_intersections_debug(masks, boundaries, output_path):
    """可视化边界交集用于调试"""
    object_names = list(masks.keys())
    n_pairs = len(object_names) * (len(object_names) - 1) // 2
    
    if n_pairs == 0:
        return
    
    fig, axes = plt.subplots(2, n_pairs, figsize=(n_pairs * 4, 8))
    if n_pairs == 1:
        axes = axes.reshape(2, 1)
    
    pair_idx = 0
    for i in range(len(object_names)):
        for j in range(i+1, len(object_names)):
            name1, name2 = object_names[i], object_names[j]
            
            # 显示两个边界
            combined_boundaries = boundaries[name1].astype(float) + boundaries[name2].astype(float) * 0.5
            axes[0, pair_idx].imshow(combined_boundaries, cmap='viridis')
            axes[0, pair_idx].set_title(f'{name1} & {name2} Boundaries')
            axes[0, pair_idx].axis('off')
            
            # 显示交集
            intersection = boundaries[name1] & boundaries[name2]
            axes[1, pair_idx].imshow(intersection, cmap='Reds')
            intersection_count = np.sum(intersection)
            axes[1, pair_idx].set_title(f'Intersection: {intersection_count} pixels')
            axes[1, pair_idx].axis('off')
            
            pair_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 交集调试图保存: {output_path}")
    plt.close()


def create_hierarchical_layout(G, occlusion_graph):
    """创建优美的布局，根据遮挡关系确定节点位置"""
    import math
    
    pos = {}
    nodes = list(G.nodes())
    
    # 计算每个节点的层级（入度）
    node_levels = {}
    for node in nodes:
        node_levels[node] = 0
    
    # 根据遮挡关系计算层级
    for occluder, relations in occlusion_graph.items():
        for occluded, _ in relations:
            node_levels[occluded] = max(node_levels[occluded], node_levels[occluder] + 1)
    
    # 按层级分组节点
    levels = {}
    for node, level in node_levels.items():
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
    
    # 选择布局类型
    layout_type = "spiral"  # 可选：arc, spiral, diamond
    
    if layout_type == "arc":
        # 弧形布局
        max_level = max(levels.keys()) if levels else 0
        radius = 3.0
        total_angle = math.pi * 0.8
        
        for level, level_nodes in levels.items():
            level_nodes.sort()
            num_nodes_in_level = len(level_nodes)
            current_radius = radius + level * 1.5
            
            if num_nodes_in_level == 1:
                angle = total_angle / 2
                x = current_radius * math.cos(angle - total_angle/2)
                y = current_radius * math.sin(angle - total_angle/2)
                pos[level_nodes[0]] = (x, y)
            else:
                angle_step = total_angle / (num_nodes_in_level - 1) if num_nodes_in_level > 1 else 0
                for i, node in enumerate(level_nodes):
                    angle = i * angle_step
                    x = current_radius * math.cos(angle - total_angle/2)
                    y = current_radius * math.sin(angle - total_angle/2)
                    pos[node] = (x, y)
    
    elif layout_type == "spiral":
        # 螺旋布局：从中心向外螺旋展开
        sorted_nodes = []
        for level in sorted(levels.keys()):
            sorted_nodes.extend(sorted(levels[level]))
        
        for i, node in enumerate(sorted_nodes):
            angle = i * (2 * math.pi / len(sorted_nodes)) + i * 0.5  # 螺旋角度
            radius = 1.5 + i * 0.8  # 递增半径
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            pos[node] = (x, y)
    
    elif layout_type == "diamond":
        # 菱形布局：创建钻石形状
        if len(nodes) == 4:
            # 特殊处理4个节点的菱形
            sorted_nodes = []
            for level in sorted(levels.keys()):
                sorted_nodes.extend(sorted(levels[level]))
            
            positions = [
                (0, 2),      # 上
                (-1.5, 0),   # 左
                (1.5, 0),    # 右  
                (0, -2)      # 下
            ]
            for i, node in enumerate(sorted_nodes):
                pos[node] = positions[i % len(positions)]
        else:
            # 回退到弧形布局
            return create_hierarchical_layout(G, occlusion_graph)
    
    return pos


def visualize_occlusion_graph(image_path, masks, occlusion_graph, output_path=None):
    """可视化遮挡关系图"""
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左侧：在原图上标注物体和遮挡关系
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    ax1.imshow(img)
    ax1.set_title("Objects and Occlusion Relations")
    ax1.axis('off')
    
    # 为每个物体添加标签和边界框
    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
    object_centers = {}
    
    for i, (name, mask) in enumerate(masks.items()):
        # 找到物体中心
        y_coords, x_coords = np.where(mask)
        if len(x_coords) > 0:
            center_x, center_y = np.mean(x_coords), np.mean(y_coords)
            object_centers[name] = (center_x, center_y)
            
            # 绘制物体轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.reshape(-1, 2)
                ax1.plot(contour[:, 0], contour[:, 1], color=colors[i], linewidth=2)
            
            # 添加标签
            ax1.text(center_x, center_y, name.replace('object_', 'O'), 
                    fontsize=12, color='white', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8))
    
    # 绘制遮挡关系箭头
    for occluder, relations in occlusion_graph.items():
        if occluder in object_centers:
            for occluded, score in relations:
                if occluded in object_centers:
                    x1, y1 = object_centers[occluder]
                    x2, y2 = object_centers[occluded]
                    
                    # 绘制箭头
                    ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.7))
                    
                    # 添加遮挡程度标签
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax1.text(mid_x, mid_y, f'{score:.2f}', 
                           fontsize=8, color='red', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 右侧：遮挡关系图
    G = nx.DiGraph()
    
    # 添加节点
    for name in masks.keys():
        G.add_node(name)
    
    # 添加边
    edge_labels = {}
    for occluder, relations in occlusion_graph.items():
        for occluded, score in relations:
            G.add_edge(occluder, occluded, weight=score)
            edge_labels[(occluder, occluded)] = f'{score:.2f}'
    
    # 绘制网络图 - 使用固定的布局策略
    # 方法1：使用固定seed的spring_layout
    # pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 方法2：根据遮挡关系创建层次化布局（推荐）
    pos = create_hierarchical_layout(G, occlusion_graph)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='lightblue', 
                          node_size=1500, alpha=0.9)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='red', 
                          arrows=True, arrowsize=20, arrowstyle='->', 
                          width=2, alpha=0.7)
    
    # 绘制标签
    labels = {name: name.replace('object_', 'O') for name in masks.keys()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax2, font_size=10)
    
    ax2.set_title("Occlusion Relationship Graph")
    ax2.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化结果保存: {output_path}")
    
    plt.show()


def print_occlusion_results(occlusion_graph):
    """打印遮挡关系结果"""
    print("\n=== 遮挡关系检测结果 ===")
    
    if not occlusion_graph:
        print("未检测到遮挡关系")
        return
    
    for occluder, relations in occlusion_graph.items():
        print(f"\n{occluder} 遮挡了:")
        for occluded, score in relations:
            print(f"  → {occluded} (遮挡程度: {score:.2f})")
    
    # 统计信息
    total_relations = sum(len(relations) for relations in occlusion_graph.values())
    print(f"\n总共检测到 {total_relations} 个遮挡关系")


def compute_occlusion_layers(occlusion_graph, masks):
    """
    根据遮挡关系图计算物体的层级结构
    
    Args:
        occlusion_graph: {occluder: [(occluded, score), ...]}
        masks: {object_name: mask_array}
    
    Returns:
        layers: [[object_1, object_2, ...], [object_3, ...], ...]
                按照层级从前到后排列，同一层的物体在同一个列表中
                遮挡别人的物体层级更前面
    """
    # 初始化：计算每个物体的最大深度
    all_objects = set(masks.keys())
    
    # 构建反向遮挡图：被谁遮挡
    # reverse_graph[occluded] = [occluder1, occluder2, ...]
    reverse_graph = defaultdict(list)
    out_degree = defaultdict(int)  # 该物体遮挡了多少个
    
    for occluder, relations in occlusion_graph.items():
        out_degree[occluder] = len(relations)
        for occluded, _ in relations:
            reverse_graph[occluded].append(occluder)
    
    # 补全所有没有出现的物体
    for obj in all_objects:
        if obj not in out_degree:
            out_degree[obj] = 0
    
    # 计算每个物体的深度（遮挡链的最长长度）
    # 深度 = 从该物体向前遮挡链的最长距离
    depth_map = {}
    
    def compute_depth(obj, visited=None):
        """递归计算物体的深度"""
        if visited is None:
            visited = set()
        
        if obj in depth_map:
            return depth_map[obj]
        
        if obj in visited:
            # 有环，返回0
            return 0
        
        visited.add(obj)
        
        # 如果这个物体遮挡了其他物体，它的深度 = max(被遮挡物体的深度) + 1
        if obj in occlusion_graph and occlusion_graph[obj]:
            max_occluded_depth = max(
                compute_depth(occluded, visited.copy()) 
                for occluded, _ in occlusion_graph[obj]
            )
            depth_map[obj] = max_occluded_depth + 1
        else:
            # 没有遮挡任何物体，深度为0（最前面）
            depth_map[obj] = 0
        
        return depth_map[obj]
    
    # 计算所有物体的深度
    for obj in all_objects:
        if obj not in depth_map:
            compute_depth(obj)
    
    # 第二步：按深度分组物体
    depth_groups = defaultdict(list)
    for obj, depth in depth_map.items():
        depth_groups[depth].append(obj)
    
    # 按深度排序（从大到小，因为深度大的在前面）
    layers = []
    for d in sorted(depth_groups.keys(), reverse=True):
        layers.append(depth_groups[d])
    
    return layers


def print_occlusion_layers(layers, masks_dir):
    """
    打印遮挡关系层级
    
    Args:
        layers: [[object_1, ...], [object_2, ...], ...]
        masks_dir: 掩码目录
    """
    print("\n=== 物体层级结构 ===")
    print("（同一层的物体可以同时移除，保持最稳定的移除顺序）")
    print("（按照从前到后的顺序，遮挡别人的物体在前面）\n")
    
    # 构建object_name到mask文件的映射
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    object_to_file = {}
    for i, mask_file in enumerate(mask_files):
        object_name = f"object_{i+1}"
        mask_filename = os.path.basename(mask_file)
        object_to_file[object_name] = mask_filename
    
    # 构建最终结果
    result_layers = []
    
    for layer_idx, layer_objects in enumerate(layers):
        layer_files = []
        for obj_name in sorted(layer_objects):
            if obj_name in object_to_file:
                layer_files.append(object_to_file[obj_name])
        
        if layer_files:
            result_layers.append(layer_files)
            print(f"第 {layer_idx + 1} 层 (从前往后): {layer_files}")
    
    print(f"\n结果列表: {result_layers}")
    
    return result_layers


def run_occlusion_detection(image_path, 
                           masks_dir,
                           dilate_size=20,
                           debug=False):
    """
    运行遮挡关系检测
    
    Args:
        image_path: 输入图像路径
        masks_dir: 掩码目录路径
        dilate_size: 边界膨胀大小，默认20
        debug: 是否开启调试模式（生成边界调试图）
    
    Returns:
        dict: {
            'occlusion_graph': 遮挡关系图,
            'layers': 层级结构,
            'depth': 深度图,
            'masks': 物体掩码
        } 或 None（失败时）
    """
    # 检查文件
    if not os.path.exists(image_path):
        print(f"❌ 未找到输入图片: {image_path}")
        return None
    
    if not os.path.exists(masks_dir):
        print(f"❌ 未找到掩码目录: {masks_dir}")
        return None
    
    print(f"输入图像: {image_path}")
    print(f"掩码目录: {masks_dir}")
    print(f"边界膨胀: {dilate_size}")
    print(f"调试模式: {'开启' if debug else '关闭'}\n")
    
    # 创建检测器
    detector = OcclusionDetector()
    
    try:
        # 处理图像
        occlusion_graph, depth, masks = detector.process_image(
            image_path, masks_dir, PROJECT_ROOT, dilate_size=dilate_size)
        
        boundaries = detector.find_object_boundaries(masks, dilate_size=dilate_size)
        
        # 调试模式：可视化边界
        if debug:
            print("\n正在生成边界调试图...")
            visualize_boundaries_debug(masks, boundaries, 
                                     os.path.join(PROJECT_ROOT, 'boundaries_debug.png'))
            visualize_intersections_debug(masks, boundaries, 
                                        os.path.join(PROJECT_ROOT, 'intersections_debug.png'))
            # 打印结果
            print_occlusion_results(occlusion_graph)
            # 可视化
            print("\n正在生成可视化...")
            visualize_occlusion_graph(
                image_path, masks, occlusion_graph, 
                os.path.join(PROJECT_ROOT, 'occlusion_visualization.png'))
        # 计算并打印层级结构
        layers = compute_occlusion_layers(occlusion_graph, masks)
        result_layers = print_occlusion_layers(layers, masks_dir)
        # 返回结果
        return {
            'occlusion_graph': occlusion_graph,
            'layers': result_layers,
            'depth': depth,
            'masks': masks
        }
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # 示例用法
    flag = 'table1'
    image_path = os.path.join(PROJECT_ROOT, 'data', 'scene', 'imgs', f'{flag}_0.png')
    masks_dir = os.path.join(PROJECT_ROOT, 'data', 'scene', 'masks', flag)
    
    # 基本调用
    run_occlusion_detection(image_path, masks_dir)
    
    # 开启调试模式
    # run_occlusion_detection(image_path, masks_dir, debug=True)
    
    # 自定义边界膨胀
    # run_occlusion_detection(image_path, masks_dir, dilate_size=15, debug=True)



