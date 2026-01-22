import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image

# 添加sam3模块路径
sam3_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sam3')
if sam3_path not in sys.path:
    sys.path.insert(0, sam3_path)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def load_sam3(checkpoint_path=None):
    if checkpoint_path is None:
        # 默认路径：相对于tools目录的sam3模型文件
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sam3', 'sam3_ckpt', 'sam3.pt')
    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    processor = Sam3Processor(model)
    return processor

def sam3_infer_prompt(processor, image_path, prompt=""):
    # Load an image
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    return masks, boxes, scores


def extract_category_from_filename(filename):
    """从文件名提取类别名，如 'mug_0.png' -> 'mug'"""
    return Path(filename).stem.rsplit('_', 1)[0]


def load_old_mask(mask_path):
    """加载旧的mask文件"""
    if os.path.exists(mask_path):
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return None


def compute_mask_iou(mask1, mask2):
    """计算两个mask的IoU"""
    if mask1 is None or mask2 is None:
        return 0.0
    
    # 确保尺寸一致
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    
    intersection = np.logical_and(mask1 > 127, mask2 > 127).sum()
    union = np.logical_or(mask1 > 127, mask2 > 127).sum()
    
    return intersection / union if union > 0 else 0.0


def match_new_mask_to_old(new_mask, old_masks_dict, category):
    """
    将新mask与旧mask匹配，找到最佳对应关系
    
    Args:
        new_mask: 新分割的mask
        old_masks_dict: {filename: mask_array} 旧mask字典
        category: 当前类别
    
    Returns:
        best_filename: 匹配的旧文件名，如果没有匹配则返回None
    """
    best_iou = 0.0
    best_filename = None
    
    for filename, old_mask in old_masks_dict.items():
        if extract_category_from_filename(filename) == category:
            iou = compute_mask_iou(new_mask, old_mask)
            if iou > best_iou:
                best_iou = iou
                best_filename = filename
    
    return best_filename if best_iou > 0.3 else None  # IoU阈值0.3


def crop_and_resize_by_mask(image, mask, target_size=512, object_ratio=0.85):
    """
    根据mask crop出对应区域，然后resize到目标大小，物体占据指定比例，物体居中
    使用仿射变换确保物体精确居中
    
    Args:
        image: 原始RGB图像
        mask: 二值化的mask (H, W)
        target_size: 目标输出大小（512）
        object_ratio: 物体应占据的比例（0.85）
    
    Returns:
        cropped_resized: 处理后的图像 (target_size, target_size, 3)
    """
    # 找到mask的边界
    y_coords, x_coords = np.where(mask > 127)
    
    if len(x_coords) == 0:
        # 如果mask为空，返回黑色图像
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 获取mask的边界框大小
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    
    # 物体的中心点（在原图坐标系中）
    object_center_x = (x_min + x_max) / 2.0
    object_center_y = (y_min + y_max) / 2.0
    
    # 计算缩放比例，使物体占据目标图像的 object_ratio
    # 物体的最大边 * scale = target_size * object_ratio
    scale = (target_size * object_ratio) / max(bbox_width, bbox_height)
    
    # 构建仿射变换矩阵：先缩放，再平移使物体中心对齐目标图像中心
    # 目标图像中心
    target_center = target_size / 2.0
    
    # 仿射变换矩阵：缩放 + 平移
    # 先将物体中心移到原点，然后缩放，最后移到目标中心
    M = np.array([
        [scale, 0, target_center - scale * object_center_x],
        [0, scale, target_center - scale * object_center_y]
    ], dtype=np.float32)
    
    # 应用仿射变换
    warped_image = cv2.warpAffine(
        image, M, (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    warped_mask = cv2.warpAffine(
        mask, M, (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # 只保留mask区域的RGB，其他区域为黑色
    warped_image[warped_mask <= 127] = 0
    
    return warped_image


def remove_duplicate_masks(mask_indices, mask_list, score_list, iou_threshold=0.8):
    """
    通过mask像素位置去除重复检测，保留score最高的
    
    Args:
        mask_indices: 排序后的mask索引列表
        mask_list: 对应的mask numpy数组列表
        score_list: 对应的score列表
        iou_threshold: IoU阈值，超过此值认为是重复的
    
    Returns:
        keep_indices: 保留的mask索引列表
    """
    if len(mask_indices) == 0:
        return []
    
    keep_indices = [mask_indices[0]]  # 总是保留score最高的
    
    for i in range(1, len(mask_indices)):
        current_mask = mask_list[i]
        is_duplicate = False
        
        # 与已保留的mask比较
        for kept_idx in keep_indices:
            kept_mask = mask_list[kept_idx]
            
            # 计算IoU (Intersection over Union)
            intersection = np.logical_and(current_mask > 127, kept_mask > 127).sum()
            union = np.logical_or(current_mask > 127, kept_mask > 127).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            keep_indices.append(mask_indices[i])
    
    return keep_indices


def save_masks_by_category(masks, scores, categories, mask_save_dir, rgb_save_dir=None, seg_rgb_save_dir=None, img_path=None, flag="", erosion=0, iou_threshold=0.8, update_mode=False, old_mask_files=None):
    """
    按类别保存所有masks，按score从高到低排序，去除重复检测
    
    Args:
        masks: 所有检测到的masks
        scores: 对应的scores，形状应为 (num_masks, num_categories)
        categories: 类别列表，如 ["category1", "category2", ...]
        mask_save_dir: mask保存目录
        rgb_save_dir: RGB保存目录（裁剪居中后的，如果为None，则不保存）
        seg_rgb_save_dir: 分割物体原始RGB保存目录（保留原始大小，如果为None，则不保存）
        img_path: 原始图像路径，用于提取RGB
        flag: 文件名前缀标志
        erosion: 腐蚀操作的kernel大小
        iou_threshold: IoU阈值，超过此值认为是重复的（默认0.8）
        update_mode: 是否为更新模式（匹配旧mask）
        old_mask_files: 旧mask文件名列表（用于更新模式）
    
    Returns:
        mask_dict: {category: [mask_np_list]}, 按score从高到低排序
    """
    os.makedirs(mask_save_dir, exist_ok=True)
    if rgb_save_dir:
        os.makedirs(rgb_save_dir, exist_ok=True)
    if seg_rgb_save_dir:
        os.makedirs(seg_rgb_save_dir, exist_ok=True)
    
    mask_dict = {cat: [] for cat in categories}
    
    if masks is None or scores is None:
        print("未检测到任何mask")
        return mask_dict
    
    # 加载原始图像（用于提取RGB）
    original_image = None
    if (rgb_save_dir or seg_rgb_save_dir) and img_path and os.path.exists(img_path):
        original_image = cv2.imread(img_path)
    
    # 更新模式：加载所有旧masks
    old_masks_dict = {}
    if update_mode and old_mask_files:
        for filename in old_mask_files:
            old_mask_path = os.path.join(mask_save_dir, filename)
            old_mask = load_old_mask(old_mask_path)
            if old_mask is not None:
                old_masks_dict[filename] = old_mask
        print(f"更新模式：加载了 {len(old_masks_dict)} 个旧mask用于匹配")
    
    # 处理scores维度
    if len(scores.shape) == 1:
        # 单个类别情况，扩展为 (num_masks, 1)
        scores = scores.unsqueeze(1)
    
    num_masks = masks.shape[0]
    num_categories = len(categories)
    
    # 对每个类别处理
    for cat_idx, category in enumerate(categories):
        if cat_idx >= scores.shape[1]:
            continue
            
        # 获取该类别的所有scores
        cat_scores = scores[:, cat_idx]
        
        # 按score排序（从高到低）
        sorted_indices = torch.argsort(cat_scores, descending=True)
        
        # 先提取所有masks并转为numpy
        all_masks_np = []
        all_scores = []
        for mask_idx in sorted_indices:
            mask = masks[mask_idx]
            score = cat_scores[mask_idx].item()
            
            # 处理不同的形状
            if len(mask.shape) == 3:  # (1, H, W)
                mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255
            else:  # (H, W)
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            
            all_masks_np.append(mask_np)
            all_scores.append(score)
        
        # 去除重复的masks
        keep_indices = remove_duplicate_masks(
            list(range(len(sorted_indices))), 
            all_masks_np, 
            all_scores,
            iou_threshold=iou_threshold
        )
        
        # 保存去重后的masks和RGB
        mask_list = []
        for rank, local_idx in enumerate(keep_indices):
            mask_np = all_masks_np[local_idx]
            score = all_scores[local_idx]
            
            # 应用腐蚀操作
            if erosion > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))
                mask_np = cv2.erode(mask_np, kernel, iterations=1)
            
            # 更新模式：匹配旧mask文件名
            if update_mode and old_masks_dict:
                matched_filename = match_new_mask_to_old(mask_np, old_masks_dict, category)
                if matched_filename:
                    # 使用匹配到的旧文件名
                    mask_filename = matched_filename
                    rgb_filename = matched_filename
                    seg_rgb_filename = matched_filename
                    print(f"  匹配成功: 新mask -> {matched_filename} (IoU based)")
                else:
                    # 没有匹配，使用新文件名
                    if flag:
                        mask_filename = f"{flag}_{category}_{rank}.png"
                        rgb_filename = f"{flag}_{category}_{rank}.png"
                        seg_rgb_filename = f"{flag}_{category}_{rank}.png"
                    else:
                        mask_filename = f"{category}_{rank}.png"
                        rgb_filename = f"{category}_{rank}.png"
                        seg_rgb_filename = f"{category}_{rank}.png"
                    print(f"  未匹配: 使用新文件名 {mask_filename}")
            else:
                # 普通模式：生成保存路径 {flag}_category_id
                if flag:
                    mask_filename = f"{flag}_{category}_{rank}.png"
                    rgb_filename = f"{flag}_{category}_{rank}.png"
                    seg_rgb_filename = f"{flag}_{category}_{rank}.png"
                else:
                    mask_filename = f"{category}_{rank}.png"
                    rgb_filename = f"{category}_{rank}.png"
                    seg_rgb_filename = f"{category}_{rank}.png"
            
            # 保存mask
            mask_path = os.path.join(mask_save_dir, mask_filename)
            cv2.imwrite(mask_path, mask_np)
            print(f"  已保存mask: {mask_filename} (score: {score:.4f})")
            
            # 保存裁剪居中的RGB图像（如果提供了rgb_save_dir）
            if rgb_save_dir and original_image is not None:
                rgb_path = os.path.join(rgb_save_dir, rgb_filename)
                # 根据mask crop出对应区域，resize到512*512，物体占85%，物体居中
                cropped_rgb = crop_and_resize_by_mask(original_image, mask_np, target_size=512, object_ratio=0.85)
                cv2.imwrite(rgb_path, cropped_rgb)
                print(f"  已保存裁剪RGB: {rgb_filename}")
            
            # 保存分割物体的原始RGB图像（如果提供了seg_rgb_save_dir）
            if seg_rgb_save_dir and original_image is not None:
                seg_rgb_path = os.path.join(seg_rgb_save_dir, seg_rgb_filename)
                # 保留原始大小，只在mask区域保留RGB，其他区域为黑色
                segmented_rgb = original_image.copy()
                segmented_rgb[mask_np <= 127] = 0
                cv2.imwrite(seg_rgb_path, segmented_rgb)
                print(f"  已保存分割RGB: {seg_rgb_filename}")
            
            mask_list.append(mask_np)
        
        mask_dict[category] = mask_list
    
    return mask_dict


def process_single_image(processor, img_path, prompt, mask_save_dir, rgb_save_dir=None, seg_rgb_save_dir=None, categories=None, flag="", debug=False, erosion=5, iou_threshold=0.8, update_mode=False):
    """
    处理单张图像进行SAM3分割，支持多个类别
    
    Args:
        processor: SAM3处理器
        img_path: 输入图像路径
        prompt: 分割提示词（字符串）或 旧mask文件名列表（更新模式）
        mask_save_dir: mask保存目录
        rgb_save_dir: RGB保存目录（裁剪居中后的，可选）
        seg_rgb_save_dir: 分割物体原始RGB保存目录（保留原始大小，可选）
        categories: 类别列表，如 ["ball", "cube"]，如果为None则自动提取
        flag: 文件名前缀标志（可选）
        debug: 是否保存叠加结果
        erosion: 腐蚀操作的kernel大小
        iou_threshold: IoU阈值，超过此值认为是重复的（默认0.8）
        update_mode: 是否为更新模式（从文件名列表提取类别并匹配旧mask）
    
    Returns:
        mask_dict: {category: [mask_np_list]}, 每个类别对应的mask列表（按score从高到低排序）
    """
    try:
        # 更新模式：从文件名列表提取类别和prompt
        old_mask_files = None
        if update_mode:
            if isinstance(prompt, list):
                # prompt是文件名列表
                old_mask_files = prompt
                categories = list(set(extract_category_from_filename(f) for f in prompt))
                prompt_str = " ".join(categories)
                print(f"[更新模式] 从文件名提取类别: {categories}")
                print(f"[更新模式] 旧mask文件: {old_mask_files}")
            else:
                raise ValueError("更新模式下，prompt必须是文件名列表")
        else:
            # 普通模式：从prompt字符串提取类别
            if categories is None:
                categories = [cat.strip() for cat in prompt.replace('，', ',').split(',')]
            prompt_str = prompt
        
        print(f"处理图像: {img_path}")
        print(f"提示词: {prompt_str}")
        print(f"类别: {categories}\n")
        
        masks, boxes, scores = sam3_infer_prompt(processor, img_path, prompt_str)
        mask_dict = save_masks_by_category(
            masks, scores, categories, mask_save_dir, 
            rgb_save_dir=rgb_save_dir, 
            seg_rgb_save_dir=seg_rgb_save_dir, 
            img_path=img_path, 
            flag=flag, 
            erosion=erosion, 
            iou_threshold=iou_threshold,
            update_mode=update_mode,
            old_mask_files=old_mask_files
        )
        
        # 加载原始图像
        original_image = cv2.imread(img_path)
        
        if debug and original_image is not None and mask_save_dir:
            # 为每个类别生成叠加结果
            colors = [
                [0, 0, 255],      # 红色
                [0, 255, 0],      # 绿色
                [255, 0, 0],      # 蓝色
                [0, 255, 255],    # 黄色
                [255, 0, 255],    # 洋红色
                [255, 255, 0],    # 青色
            ]
            
            for cat_idx, (category, mask_list) in enumerate(mask_dict.items()):
                if not mask_list:
                    continue
                
                # 使用该类别对应的颜色
                color = colors[cat_idx % len(colors)]
                
                # 将所有该类别的masks合并
                combined_mask = np.zeros_like(mask_list[0])
                for mask_np in mask_list:
                    combined_mask = np.maximum(combined_mask, mask_np)
                
                # 确保mask和原图大小一致
                if original_image.shape[:2] != combined_mask.shape[:2]:
                    combined_mask = cv2.resize(combined_mask, (original_image.shape[1], original_image.shape[0]))
                
                # 创建叠加图像
                mask_indices = combined_mask > 127
                overlay_rgb = original_image.copy()
                overlay_rgb[mask_indices] = color
                result_mask_rgb = cv2.addWeighted(original_image, 0.7, overlay_rgb, 0.3, 0)
                
                # 保存叠加结果
                if flag:
                    overlay_filename = f"{flag}_{category}_overlay.png"
                else:
                    overlay_filename = f"{category}_overlay.png"
                overlay_path = os.path.join(mask_save_dir, overlay_filename)
                cv2.imwrite(overlay_path, result_mask_rgb)
                print(f"已保存叠加结果: {overlay_filename}")
        
        print(f"✓ 处理完成！\n")
        return mask_dict
        
    except Exception as e:
        print(f"✗ 处理失败: {e}\n")
        return {}


if __name__ == "__main__":
    # 加载SAM3模型
    
    flag = 'table2'
    img_path = f'/home/lian/Desktop/genpc_base/data/scene/imgs/{flag}.png'
    mask_save_dir = f'/home/lian/Desktop/genpc_base/data/scene/masks/{flag}/'
    rgb_save_dir = f'/home/lian/Desktop/genpc_base/data/scene/rgb/{flag}/'
    seg_rgb_save_dir = f'/home/lian/Desktop/genpc_base/data/scene/seg_rgb/{flag}/'
    qwen3vl_result = 'chair table'
    processor = load_sam3()
    for prompt in qwen3vl_result.split():
        process_single_image(
            processor=processor,
            img_path=img_path,
            prompt=prompt,
            mask_save_dir=mask_save_dir,
            rgb_save_dir=rgb_save_dir,
            seg_rgb_save_dir=seg_rgb_save_dir,
            debug=False,
        )