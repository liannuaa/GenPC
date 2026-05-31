import torch
from diffusers import QwenImageEditPipeline
from PIL import Image
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision
import logging

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
        transformer_path="models/nunchaku-qwen-image-edit/svdq-int4_r128-qwen-image-edit-lightningv1.0-8steps.safetensors",
        pipeline_path="models/Qwen-Image-Edit",
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

        # 如果未指定 transformer 路径，则根据 rank 和 step 自动生成
        if transformer_path is None:
            transformer_path = NunchakuQwenImageTransformer2DModel.from_pretrained(
                f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
            )

        logger.info(f"Loading Qwen Image Edit (rank={rank}, step={step})...")
        logger.info(f"  Transformer: {transformer_path}")
        logger.info(f"  Pipeline: {pipeline_path}")

        # 加载 transformer 模型
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            transformer_path
        )

        # 加载 pipeline
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            pipeline_path, transformer=self.transformer, torch_dtype=torch.bfloat16
        )

        # 启用 CPU offload 以节省显存
        self.transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=50)
        self.pipeline._exclude_from_cpu_offload.append("transformer")
        self.pipeline.enable_sequential_cpu_offload()

        logger.info("✓ Qwen Image Edit 模型加载完成")

    def generate(self, depth_image, flag, size=1024, cfg_scale=4.0):
        """
        从深度图生成真实感图像

        Args:
            depth_image: PIL Image 或图像路径
            flag: 物体标签/描述 (例如 'rubbish bin')
            size: 生成图像尺寸 (默认1024)
            cfg_scale: 引导尺度 (默认4.0，范围1-10)

        Returns:
            PIL Image: 生成的图像
        """
        # 处理输入
        if isinstance(depth_image, str):
            depth_image = Image.open(depth_image)

        # 调整尺寸
        if depth_image.size[0] != size:
            depth_image = depth_image.resize((size, size), Image.LANCZOS)

        # 构建专业级 prompt
        prompt = self._build_prompt(flag)
        negative_prompt = (
            "dirty, stains, rust, broken, damaged, torn, crumpled, wrinkled, melted, deformed, "
            "jagged edges, fragmented outline, paper, cardboard, clay, sculpture, toy-like, "
            "extra parts, missing parts, multiple objects, text, logo, watermark, unrealistic texture, "
            "overexposed, washed out, low contrast, transparent, ghostly, pure white object"
        )

        logger.info(f"Generating image from depth map (flag={flag})...")
        # print(f"  Prompt: {prompt}")

        # 推理
        with torch.no_grad():
            inputs = {
                "image": depth_image,
                "prompt": prompt,
                "true_cfg_scale": cfg_scale,
                "negative_prompt": negative_prompt,
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

        return (
            f"A realistic everyday {flag}, matching the object category and overall shape indicated by the input depth map. "
            f"Preserve the same silhouette, proportions, and main structural parts from the depth image, "
            f"but present the object in a straight-on front view at eye level, with minimal top-down perspective. "
            f"Render it as a clean, ordinary real-world object with plausible materials, natural colors, and simple practical design. "
            f"Use smooth, coherent surfaces with clear edges and stable geometry, avoiding melted, crumpled, torn, or broken shapes. "
            f"Use natural object colors, visible material texture, and moderate contrast; "
            f"avoid a washed-out or pure white object unless the category is normally white. "
            f"Professional product photography on a clean white background, soft natural studio lighting, sharp focus, "
            f"realistic scale and physically plausible details."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 示例使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    qwen_depth = Qwen_depth(
        device,
        step=8,
        transformer_path="models/nunchaku-qwen-image-edit/svdq-int4_r128-qwen-image-edit-lightningv1.0-8steps.safetensors",
        pipeline_path="models/Qwen-Image-Edit",
    )

    # 生成图像
    depth_path = "workspace/01184/depth.png"
    output_image = qwen_depth.generate(depth_path, flag="rubbish bin", size=1024)

    # 保存结果
    output_image.save("qwen_output.png")
    logger.info("✓ 结果已保存到 qwen_output.png")
