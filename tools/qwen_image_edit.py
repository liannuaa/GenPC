import gc
import logging
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
from PIL import Image


logger = logging.getLogger(__name__)


def build_completion_prompt(flag):
    return (
        f"Complete the {flag} 's missing silhouette, outer contour, and visible "
        f"structure while preserving the original position and pose. Strictly keep the "
        f"{flag}'s side-view viewpoint, scale, orientation, and location unchanged. "
        f"Generate a realistic RGB {flag} image. The completed {flag} "
        f"outline must stay aligned with the input silhouette. Make the background realistic "
    )


def build_completion_negative_prompt(flag):
    return (
        "depth map, grayscale depth rendering, changed viewpoint, shifted object, "
        f"extra objects, changed scale, "
        "changed center location, distorted structure, broken silhouette, text, watermark, "
        "people, clutter, occlusion"
    )


class QwenImageEdit:
    def __init__(
        self,
        device,
        transformer_path,
        pipeline_path,
        step=40,
        true_cfg_scale=4.0,
        generation_size=1024,
        cpu_offload=True,
    ):
        self.device = torch.device(device)
        self.step = int(step)
        self.true_cfg_scale = float(true_cfg_scale)
        self.generation_size = int(generation_size)
        transformer_path = Path(transformer_path).expanduser().resolve()
        pipeline_path = Path(pipeline_path).expanduser().resolve()
        if not transformer_path.exists():
            raise FileNotFoundError(f"Qwen-Image-Edit transformer not found: {transformer_path}")
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Qwen-Image-Edit pipeline not found: {pipeline_path}")

        logger.info("Loading Qwen-Image-Edit (steps=%d)...", self.step)
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            str(transformer_path)
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            {
                "base_image_seq_len": 256,
                "base_shift": 3.0,
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": 3.0,
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        )
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            str(pipeline_path),
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )
        if cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

    def generate(
        self,
        image,
        flag,
        size=512,
        input_size=None,
        mode="depth",
        controlnet_conditioning_scale=None,
        seed=None,
        **kwargs,
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        generation_size = int(kwargs.get("generation_size", self.generation_size))
        image = image.resize((generation_size, generation_size), Image.Resampling.LANCZOS)
        prompt = build_completion_prompt(flag)
        generator = None
        if seed is not None and str(self.device).startswith("cuda"):
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        output = self.pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=build_completion_negative_prompt(flag),
            true_cfg_scale=self.true_cfg_scale,
            height=generation_size,
            width=generation_size,
            num_inference_steps=self.step,
            generator=generator,
        )
        result = output.images[0]
        if result.size != (size, size):
            result = result.resize((size, size), Image.Resampling.LANCZOS)
        return result

    def close(self):
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
