import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tools.qwen_image_edit import (
    QwenImageEdit,
    build_completion_negative_prompt,
    build_completion_prompt,
)


class QwenImageEditPromptTest(unittest.TestCase):
    def test_completion_prompt_requests_rgb_object_not_depth_map(self):
        prompt = build_completion_prompt("car")

        self.assertIn("Complete the car 's missing silhouette", prompt)
        self.assertIn("preserving the original position and pose", prompt)
        self.assertIn("side-view viewpoint, scale, orientation, and location", prompt)
        self.assertIn("Generate a realistic RGB car image", prompt)
        self.assertIn("Make the background realistic", prompt)
        self.assertNotIn("remain a clean depth-map style image", prompt)

    def test_negative_prompt_rejects_depth_style_and_pose_changes(self):
        negative_prompt = build_completion_negative_prompt("car")

        self.assertIn("depth map", negative_prompt)
        self.assertIn("grayscale depth rendering", negative_prompt)
        self.assertIn("changed center location", negative_prompt)

    def test_qwen_edit_defaults_match_manual_2511_experiment_settings(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch("tools.qwen_image_edit.NunchakuQwenImageTransformer2DModel"),
            patch("tools.qwen_image_edit.FlowMatchEulerDiscreteScheduler"),
            patch("tools.qwen_image_edit.QwenImageEditPipeline"),
        ):
            editor = QwenImageEdit(
                device="cuda",
                transformer_path="/tmp/transformer.safetensors",
                pipeline_path="/tmp/pipeline",
            )

        self.assertEqual(editor.step, 40)
        self.assertEqual(editor.true_cfg_scale, 4.0)
        self.assertEqual(editor.generation_size, 1024)

    def test_generate_uses_1024_pipeline_size_then_downscales_to_requested_size(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch("tools.qwen_image_edit.NunchakuQwenImageTransformer2DModel"),
            patch("tools.qwen_image_edit.FlowMatchEulerDiscreteScheduler"),
            patch("tools.qwen_image_edit.QwenImageEditPipeline"),
        ):
            editor = QwenImageEdit(
                device="cuda",
                transformer_path="/tmp/transformer.safetensors",
                pipeline_path="/tmp/pipeline",
                generation_size=1024,
            )

        calls = {}

        def fake_pipeline(**kwargs):
            calls.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (1024, 1024))])

        editor.pipeline = fake_pipeline
        result = editor.generate(Image.new("RGB", (512, 512)), "car", size=512)

        self.assertEqual(calls["height"], 1024)
        self.assertEqual(calls["width"], 1024)
        self.assertEqual(calls["image"].size, (1024, 1024))
        self.assertEqual(result.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
