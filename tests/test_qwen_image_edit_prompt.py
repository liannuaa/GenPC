import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tools.qwen_image_edit import (
    QwenImageEdit,
    build_completion_prompt,
    resize_stage1_image_for_output,
)


class QwenImageEditPromptTest(unittest.TestCase):
    def test_completion_prompt_requests_completed_depth_from_incomplete_depth(self):
        prompt = build_completion_prompt("car")

        self.assertEqual(prompt, "生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的汽车，纯白背景")
        self.assertNotIn("真实汽车照片", prompt)

    def test_completion_prompt_uses_object_parameter(self):
        prompt = build_completion_prompt("chair")

        self.assertEqual(prompt, "生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的chair，纯白背景")
        self.assertNotIn("汽车", prompt)

    def test_qwen_edit_uses_explicit_generation_settings(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch("tools.qwen_image_edit.NunchakuQwenImageTransformer2DModel"),
            patch("tools.qwen_image_edit.QwenImageEditPlusPipeline"),
        ):
            editor = QwenImageEdit(
                device="cuda",
                transformer_path="/tmp/transformer.safetensors",
                pipeline_path="/tmp/pipeline",
                step=16,
                generation_size=1024,
                true_cfg_scale=4.0,
                negative_prompt=" ",
            )

        self.assertEqual(editor.step, 16)
        self.assertEqual(editor.generation_size, 1024)
        self.assertEqual(editor.true_cfg_scale, 4.0)
        self.assertEqual(editor.negative_prompt, " ")

    def test_stage1_output_is_resized_for_saved_main_flow_artifact(self):
        image = Image.new("RGB", (1024, 1024), (120, 120, 120))

        resized = resize_stage1_image_for_output(image, 512)

        self.assertEqual(resized.size, (512, 512))
        self.assertEqual(resized.mode, "RGB")

    def test_generate_runs_single_stage_then_downscales_to_requested_size(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch("tools.qwen_image_edit.NunchakuQwenImageTransformer2DModel"),
            patch("tools.qwen_image_edit.QwenImageEditPlusPipeline"),
        ):
            editor = QwenImageEdit(
                device="cuda",
                transformer_path="/tmp/transformer.safetensors",
                pipeline_path="/tmp/pipeline",
                step=16,
                generation_size=1024,
                true_cfg_scale=4.0,
                negative_prompt=" ",
            )

        calls = []

        def fake_pipeline(**kwargs):
            calls.append(kwargs)
            shade = 120 + len(calls) * 20
            return SimpleNamespace(images=[Image.new("RGB", (1024, 1024), (shade, shade, shade))])

        editor.pipeline = fake_pipeline
        result = editor.generate(Image.new("RGB", (512, 512)), "car", size=512)

        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0]["image"], list)
        self.assertEqual(calls[0]["image"][0].size, (512, 512))
        self.assertEqual(calls[0]["image"][0].mode, "RGB")
        self.assertNotIn("height", calls[0])
        self.assertNotIn("width", calls[0])
        self.assertEqual(calls[0]["prompt"], "生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的汽车，纯白背景")
        self.assertEqual(calls[0]["negative_prompt"], " ")
        self.assertEqual(calls[0]["true_cfg_scale"], 4.0)
        self.assertEqual(calls[0]["num_inference_steps"], 16)
        self.assertEqual(editor.last_stage1_image.size, (1024, 1024))
        self.assertEqual(editor.last_stage1_prompt, "生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的汽车，纯白背景")
        self.assertEqual(result.size, (512, 512))

    def test_generate_with_prompt_consumes_bounded_agent_instruction(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch("tools.qwen_image_edit.NunchakuQwenImageTransformer2DModel"),
            patch("tools.qwen_image_edit.QwenImageEditPlusPipeline"),
        ):
            editor = QwenImageEdit(
                device="cuda", transformer_path="/tmp/transformer.safetensors",
                pipeline_path="/tmp/pipeline", step=16, generation_size=512,
                true_cfg_scale=4.0, negative_prompt=" ",
            )
        calls = []
        editor.pipeline = lambda **kwargs: (calls.append(kwargs) or SimpleNamespace(
            images=[Image.new("RGB", (512, 512), "white")]))
        prompt = "Preserve pose. Correct only the observed surface locally."
        result = editor.generate_with_prompt(Image.new("RGB", (512, 512)), prompt, seed=7)
        self.assertEqual(result.size, (512, 512))
        self.assertEqual(calls[0]["prompt"], prompt)
        self.assertEqual(editor.last_stage1_prompt, prompt)

    def test_generate_with_prompt_rejects_empty_instruction(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch("tools.qwen_image_edit.NunchakuQwenImageTransformer2DModel"),
            patch("tools.qwen_image_edit.QwenImageEditPlusPipeline"),
        ):
            editor = QwenImageEdit(
                device="cuda", transformer_path="/tmp/transformer.safetensors",
                pipeline_path="/tmp/pipeline", step=16, generation_size=512,
                true_cfg_scale=4.0, negative_prompt=" ",
            )
        with self.assertRaises(ValueError):
            editor.generate_with_prompt(Image.new("RGB", (512, 512)), "  ")


if __name__ == "__main__":
    unittest.main()
