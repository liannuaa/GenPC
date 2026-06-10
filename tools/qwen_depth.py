import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from PIL import Image, ImageFilter, ImageOps
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision
import logging
import math

logger = logging.getLogger(__name__)


class Qwen_depth:
    """
    Qwen Image Edit 深度图生成类
    用于将深度图转换为真实感图像
    """

    def __init__(
        self,
        device,
        rank=128,
        step=8,
        transformer_path="models/nunchaku-qwen-image-edit-2509/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors",
        pipeline_path="models/Qwen-Image-Edit-2509",
    ):
        """
        初始化 Qwen Image Edit 模型

        Args:
            device: 计算设备 (cuda 或 cpu)
            rank: 量化等级 (默认128，可选64/128)
            step: 推理步数 (默认8，可选4/8/16)
            transformer_path: transformer 模型路径 (默认根据 rank 和 step 自动生成)
            pipeline_path: Qwen Image Edit pipeline 模型路径 (默认 "models/Qwen-Image-Edit")
        """
        self.device = device
        self.rank = rank
        self.step = step

        logger.info(f"Loading Qwen Image Edit (rank={rank}, step={step})...")
        logger.info(f"  Transformer: {transformer_path}")
        logger.info(f"  Pipeline: {pipeline_path}")

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # 加载 transformer 模型
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            transformer_path
        )

        # 加载 pipeline
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            pipeline_path, transformer=self.transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
        )

        # 启用 CPU offload 以节省显存
        self.pipeline.enable_model_cpu_offload()

        logger.info("✓ Qwen Image Edit 模型加载完成")

    def generate(self, depth_image, flag, size=1280):
        """
        从深度图生成真实感图像

        Args:
            depth_image: PIL Image 或图像路径
            flag: 物体标签/描述 (例如 'rubbish bin')
            size: 生成图像尺寸 (默认1024)

        Returns:
            PIL Image: 生成的图像
        """
        # 处理输入
        if isinstance(depth_image, str):
            depth_image = Image.open(depth_image)

        # 调整尺寸
        if depth_image.size != (size, size):
            depth_image = depth_image.resize((size, size), Image.LANCZOS)

        # 构建专业级 prompt
        prompt = self._build_prompt(flag)
        negative_prompt = "blurry, low resolution, out of focus, soft details, fuzzy edges, noisy, distorted, hazy, unclear, cropped, partial object, incomplete object, crop, occlusion"


        logger.info(f"Generating image from depth map (flag={flag})...")

        # 推理
        inputs = {
            "image": depth_image,
            "prompt": prompt,
            "true_cfg_scale": 4.0,
            "height": size,
            "width": size,
            "negative_prompt":  negative_prompt,
            "num_inference_steps": self.step
        }

        output = self.pipeline(**inputs)
        output_image = output.images[0]

        logger.info("✓ 图像生成完成")
        return output_image

    def _build_prompt(self, flag):
        """
        构建专业级 prompt

        Args:
            flag: 物体标签

        Returns:
            str: 完整的 prompt
        """

        return  f"Generate a clear, high-quality side-view image of a {flag} on a pure white background. Use the provided depth map only as a loose layout and pose reference, not an exact shape or silhouette constraint. Complete any missing parts naturally. The {flag} should be fully visible, centered in the image, with realistic geometry, consistent material, accurate surface details, and a realistic style."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 示例使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    qwen_depth = Qwen_depth(
        device=device
    )

    # 生成图像
    depth_path = "workspace/06127/depth.png"
    output_image = qwen_depth.generate(depth_path, flag="a vase with green leaves", size=1280)

    # 保存结果
    output_image.save("qwen_output.png")
    logger.info("✓ 结果已保存到 qwen_output.png")
