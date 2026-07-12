import gc
import logging
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
from PIL import Image


logger = logging.getLogger(__name__)


def _zh_object_labels(flag):
    normalized = str(flag).strip().lower()
    if normalized in {"car", "cars", "vehicle"}:
        return "车", "汽车"
    return str(flag), str(flag)


def build_completion_prompt(flag):
    _, photo_label = _zh_object_labels(flag)
    return (
        f"根据这张不完整的{photo_label}深度图生成完整的真实{photo_label}照片，"
        "保持已有部分的轮廓、姿态、朝向和相机视角不变，合理补全缺失部分。"
    )


def build_refinement_prompt(flag):
    _, photo_label = _zh_object_labels(flag)
    return (
        f"根据这张{photo_label}图生成更贴近真实{photo_label}的照片。"
        "只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，"
        "不需要保留原图的颜色、材质、光照和背景细节；"
        "让物体结构、材质和外观更真实自然，背景为真实场景。"
    )


class QwenImageEdit:
    def __init__(
        self,
        device,
        transformer_path,
        pipeline_path,
        step,
        generation_size,
        true_cfg_scale,
        negative_prompt,
        refine_stage=False,
        refine_step=None,
        cpu_offload=True,
    ):
        self.device = torch.device(device)
        self.step = int(step)
        self.refine_stage = bool(refine_stage)
        self.refine_step = int(refine_step) if refine_step is not None else int(step)
        self.generation_size = int(generation_size)
        self.true_cfg_scale = float(true_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.last_prompt = None
        self.last_stage1_prompt = None
        self.last_refinement_prompt = None
        self.last_stage1_image = None
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
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(pipeline_path),
            transformer=transformer,
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
        seed=None,
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        generator = None
        if seed is not None and str(self.device).startswith("cuda"):
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        prompt = build_completion_prompt(flag)
        output = self.pipeline(
            image=[image.convert("RGB")],
            prompt=prompt,
            true_cfg_scale=self.true_cfg_scale,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.step,
            generator=generator,
        )
        self.last_stage1_prompt = prompt
        self.last_stage1_image = output.images[0].convert("RGB")
        result = self.last_stage1_image
        if self.refine_stage:
            refinement_prompt = build_refinement_prompt(flag)
            output = self.pipeline(
                image=[result.convert("RGB")],
                prompt=refinement_prompt,
                true_cfg_scale=self.true_cfg_scale,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.refine_step,
                generator=generator,
            )
            self.last_refinement_prompt = refinement_prompt
            result = output.images[0]
            self.last_prompt = refinement_prompt
        else:
            self.last_refinement_prompt = None
            self.last_prompt = prompt
        if result.size != (size, size):
            result = result.resize((size, size), Image.Resampling.LANCZOS)
        return result

    def close(self):
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
