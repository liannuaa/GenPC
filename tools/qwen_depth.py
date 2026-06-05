import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from PIL import Image
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

        # 如果未指定 transformer 路径，则根据 rank 和 step 自动生成
        if transformer_path is None:
            transformer_path = NunchakuQwenImageTransformer2DModel.from_pretrained(
                f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
            )

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
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            pipeline_path, transformer=self.transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
        )

        # 启用 CPU offload 以节省显存
        self.transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        self.pipeline._exclude_from_cpu_offload.append("transformer")
        self.pipeline.enable_sequential_cpu_offload()

        logger.info("✓ Qwen Image Edit 模型加载完成")

    def generate(self, depth_image, flag, size=1024):
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
        negative_prompt = (
            "dirty, stains, rust, broken, damaged, torn, crumpled, wrinkled, melted, deformed, "
            "jagged edges, fragmented outline, paper, cardboard, clay, sculpture, toy-like, "
            "extra parts, missing parts, multiple objects, text, logo, watermark, unrealistic texture, "
            "overexposed, washed out, low contrast, transparent, ghostly, pure white object"
        )

        logger.info(f"Generating image from depth map (flag={flag})...")

        # 推理
        with torch.no_grad():
            inputs = {
                "image": depth_image,
                "prompt": prompt,
                "height": size,
                "width": size,
                "true_cfg_scale": 1.0,
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

        return  f"Generate an image that conforms to the depth map outlined in Figure 1 and follows the description below: a real-world physical {flag}, The entire object should be fully visible in the image, including all major parts from top to bottom, with no cropping, no missing sections, and no close-up view. Photographed in a studio, with sharp details, clear edges, coherent surfaces, realistic material appearance, and a clean white background."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 示例使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    qwen_depth = Qwen_depth(
        device,
        step=8,
        transformer_path="models/nunchaku-qwen-image-edit-2509/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors",
        pipeline_path="models/Qwen-Image-Edit-2509",
    )

    # 生成图像
    depth_path = "workspace/01184/depth.png"
    output_image = qwen_depth.generate(depth_path, flag="rubbish bin", size=768)

    # 保存结果
    output_image.save("qwen_output.png")
    logger.info("✓ 结果已保存到 qwen_output.png")
