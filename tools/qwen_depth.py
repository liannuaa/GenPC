import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageControlNetModel,
    QwenImageControlNetPipeline,
)
from PIL import Image
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
import logging
import math
import resource
import gc
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_limit(value):
    if value == resource.RLIM_INFINITY:
        return "unlimited"
    return f"{value} bytes"


class Qwen_depth:
    """
    Qwen Image ControlNet 深度图生成类
    用于将深度图转换为真实感图像
    """

    def __init__(
        self,
        device,
        rank=128,
        step=4,
        transformer_path="models/nunchaku-qwen-image/svdq-int4_r128-qwen-image-lightningv1.0-4steps.safetensors",
        pipeline_path="models/Qwen-Image",
        controlnet_path="models/Qwen-Image-ControlNet-Union",
        cpu_offload=True,
        cpu_text_encoder=False,
    ):
        """
        初始化 Qwen Image ControlNet 模型

        Args:
            device: 计算设备 (cuda 或 cpu)
            rank: 量化等级 (默认128，可选64/128)
            step: 推理步数 (默认4)
            transformer_path: Nunchaku transformer 模型路径
            pipeline_path: Qwen Image pipeline 模型路径
            controlnet_path: Qwen Image ControlNet Union 模型路径
        """
        self.device = device
        self.rank = rank
        self.step = step
        self.cpu_offload = cpu_offload
        self.cpu_text_encoder = cpu_text_encoder
        if str(self.device).startswith("cuda"):
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            min_memlock = 256 * 1024 * 1024
            if (
                hard_limit != resource.RLIM_INFINITY
                and hard_limit < min_memlock
                and bool(getattr(self, "require_memlock", False))
            ):
                raise RuntimeError(
                    "Qwen CUDA generation cannot run with the current locked-memory "
                    f"limit ({_format_limit(hard_limit)} hard limit). The process "
                    "hangs while CUDA page-locks model weights for CPU-to-GPU "
                    "transfer. Start the container/session with a larger memlock "
                    "limit, for example Docker '--ulimit memlock=-1:-1' or an "
                    "equivalent CAP_IPC_LOCK/memlock setting, then rerun."
                )
            if hard_limit != resource.RLIM_INFINITY and hard_limit < min_memlock:
                logger.warning(
                    "Low locked-memory hard limit detected: %s. CUDA model transfer "
                    "may hang on this host.",
                    _format_limit(hard_limit),
                )

        logger.info(f"Loading Qwen Image ControlNet (rank={rank}, step={step})...")
        logger.info(f"  Transformer: {transformer_path}")
        logger.info(f"  Pipeline: {pipeline_path}")
        logger.info(f"  ControlNet: {controlnet_path}")
        transformer_path = Path(transformer_path).expanduser().resolve()
        pipeline_path = Path(pipeline_path).expanduser().resolve()
        controlnet_path = Path(controlnet_path).expanduser().resolve()
        if not transformer_path.exists():
            raise FileNotFoundError(f"Qwen transformer not found at {transformer_path}")
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Qwen pipeline not found at {pipeline_path}")
        if not controlnet_path.exists():
            raise FileNotFoundError(f"Qwen ControlNet not found at {controlnet_path}")

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
            str(transformer_path)
        )
        self.controlnet = QwenImageControlNetModel.from_pretrained(
            str(controlnet_path), torch_dtype=torch.bfloat16
        )

        # 加载 pipeline
        self.pipeline = QwenImageControlNetPipeline.from_pretrained(
            str(pipeline_path),
            controlnet=self.controlnet,
            transformer=self.transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )

        if self.cpu_text_encoder:
            self.pipeline.vae.to(self.device)
            self.pipeline.transformer.to(self.device)
            self.pipeline.controlnet.to(self.device)
        elif self.cpu_offload:
            # 启用 CPU offload 以节省显存
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

        logger.info("✓ Qwen Image ControlNet 模型加载完成")

    def close(self):
        self.pipeline = None
        self.controlnet = None
        self.transformer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def generate(
        self,
        depth_image,
        flag,
        size=1280,
        input_size=None,
        mode="depth",
        true_cfg_scale=None,
        controlnet_conditioning_scale=0.9,
        seed=None,
    ):
        """
        从深度图生成真实感图像

        Args:
            depth_image: PIL Image 或图像路径
            flag: 物体标签/描述 (例如 'rubbish bin')
            size: 生成图像尺寸 (默认1024)
            input_size: 输入条件图尺寸；None 表示使用生成图像尺寸

        Returns:
            PIL Image: 生成的图像
        """
        # 处理输入
        if isinstance(depth_image, str):
            depth_image = Image.open(depth_image)

        # 调整尺寸
        condition_size = size if input_size is None else input_size
        if depth_image.size != (condition_size, condition_size):
            depth_image = depth_image.resize(
                (condition_size, condition_size), Image.LANCZOS
            )

        # 构建专业级 prompt
        prompt = self._build_prompt(flag, mode=mode)
        negative_prompt = " "


        logger.info(f"Generating image with Qwen (flag={flag}, mode={mode})...")

        inputs = {
            "control_image": depth_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "height": size,
            "width": size,
            "num_inference_steps": self.step,
            "true_cfg_scale": 1.0 if true_cfg_scale is None else true_cfg_scale,
        }
        if self.cpu_text_encoder:
            prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
                prompt=prompt,
                device=torch.device("cpu"),
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            negative_prompt_embeds, negative_prompt_embeds_mask = self.pipeline.encode_prompt(
                prompt=negative_prompt,
                device=torch.device("cpu"),
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            inputs.update(
                {
                    "prompt": None,
                    "negative_prompt": None,
                    "prompt_embeds": prompt_embeds.to(self.device),
                    "prompt_embeds_mask": prompt_embeds_mask.to(self.device),
                    "negative_prompt_embeds": negative_prompt_embeds.to(self.device),
                    "negative_prompt_embeds_mask": negative_prompt_embeds_mask.to(self.device),
                }
            )
        else:
            inputs.update(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            )
        if seed is not None and str(self.device).startswith("cuda"):
            inputs["generator"] = torch.Generator(device="cuda").manual_seed(seed)

        output = self.pipeline(**inputs)
        output_image = output.images[0]

        logger.info("✓ 图像生成完成")
        return output_image

    def _build_prompt(self, flag, mode="depth"):
        """
        构建专业级 prompt

        Args:
            flag: 物体标签

        Returns:
            str: 完整的 prompt
        """

        return f"a {flag} on a pure white background"
