# 使用 RMBG-2.0 模型移除背景
from transformers import AutoModelForImageSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import numpy as np

# 加载模型
model = AutoModelForImageSegmentation.from_pretrained(
    "/root/shared-nvme/RMBG-2.0", 
    trust_remote_code=True
)
model.eval()

# 如果有GPU，使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def RMBG_pred(input_path, output_path=None):
    """
    使用 RMBG-2.0 模型移除图片背景
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径（可选）
    
    Returns:
        output_path: 保存的输出文件路径
    """
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        import os
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_no_bg.png"
    
    # 读取图片
    image = Image.open(input_path).convert('RGB')
    original_size = image.size
    
    # 预处理：调整大小到模型输入尺寸
    input_size = (1024, 1024)
    image_resized = image.resize(input_size, Image.BILINEAR)
    
    # 转换为tensor并归一化
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    image_tensor = normalize(image_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        preds = model(image_tensor)[-1].sigmoid()
    
    # 后处理：调整回原始大小
    pred = preds[0].squeeze()
    pred_pil = Image.fromarray((pred.cpu().numpy() * 255).astype(np.uint8))
    pred_pil = pred_pil.resize(original_size, Image.BILINEAR)
    
    # 创建RGBA图像
    image_rgba = image.convert('RGBA')
    mask = pred_pil.convert('L')
    image_rgba.putalpha(mask)
    
    # 保存结果
    image_rgba.save(output_path, 'PNG')
    print(f"✓ 背景已移除，保存到: {output_path}")
    
    return output_path

if __name__ == "__main__":
    result_path = RMBG_pred("/home/engineai/code/genpc_open/workspace/01184/img.png")