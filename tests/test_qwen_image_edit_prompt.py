import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tools.qwen_image_edit import (
    QwenImageEdit,
    build_completion_prompt,
    build_refinement_prompt,
)


class QwenImageEditPromptTest(unittest.TestCase):
    def test_completion_prompt_requests_complete_photo_from_incomplete_depth(self):
        prompt = build_completion_prompt("car")

        self.assertIn("根据这张不完整的汽车深度图生成完整的真实汽车照片", prompt)
        self.assertIn("保持已有部分的轮廓、姿态、朝向和相机视角不变", prompt)
        self.assertIn("合理补全缺失部分", prompt)

    def test_completion_prompt_uses_object_parameter(self):
        prompt = build_completion_prompt("chair")

        self.assertIn("不完整的chair深度图", prompt)
        self.assertIn("真实chair照片", prompt)
        self.assertNotIn("汽车", prompt)

    def test_refinement_prompt_keeps_only_shape_pose_category_and_camera(self):
        prompt = build_refinement_prompt("red chair")

        self.assertIn("更贴近真实red chair", prompt)
        self.assertIn("只保留物体的轮廓、大小、种类、朝向、姿态和相机视角", prompt)
        self.assertIn("不需要保留原图的颜色、材质、光照和背景细节", prompt)
        self.assertIn("背景为真实场景", prompt)

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
                refine_stage=True,
                refine_step=16,
            )

        self.assertEqual(editor.step, 16)
        self.assertTrue(editor.refine_stage)
        self.assertEqual(editor.refine_step, 16)
        self.assertEqual(editor.generation_size, 1024)
        self.assertEqual(editor.true_cfg_scale, 4.0)
        self.assertEqual(editor.negative_prompt, " ")

    def test_generate_runs_two_plus_stages_then_downscales_to_requested_size(self):
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
                refine_stage=True,
                refine_step=16,
            )

        calls = []

        def fake_pipeline(**kwargs):
            calls.append(kwargs)
            shade = 120 + len(calls) * 20
            return SimpleNamespace(images=[Image.new("RGB", (1024, 1024), (shade, shade, shade))])

        editor.pipeline = fake_pipeline
        result = editor.generate(Image.new("RGB", (512, 512)), "car", size=512)

        self.assertEqual(len(calls), 2)
        self.assertIsInstance(calls[0]["image"], list)
        self.assertEqual(calls[0]["image"][0].size, (512, 512))
        self.assertEqual(calls[0]["image"][0].mode, "RGB")
        self.assertNotIn("height", calls[0])
        self.assertNotIn("width", calls[0])
        self.assertIn("不完整的汽车深度图", calls[0]["prompt"])
        self.assertEqual(calls[0]["negative_prompt"], " ")
        self.assertEqual(calls[0]["true_cfg_scale"], 4.0)
        self.assertEqual(calls[0]["num_inference_steps"], 16)
        self.assertIsInstance(calls[1]["image"], list)
        self.assertEqual(calls[1]["image"][0].size, (1024, 1024))
        self.assertEqual(calls[1]["image"][0].mode, "RGB")
        self.assertIn("更贴近真实汽车", calls[1]["prompt"])
        self.assertIn("只保留物体的轮廓、大小、种类、朝向、姿态和相机视角", calls[1]["prompt"])
        self.assertIn("背景为真实场景", calls[1]["prompt"])
        self.assertEqual(calls[1]["negative_prompt"], " ")
        self.assertEqual(calls[1]["true_cfg_scale"], 4.0)
        self.assertEqual(calls[1]["num_inference_steps"], 16)
        self.assertEqual(editor.last_stage1_image.size, (1024, 1024))
        self.assertIn("不完整的汽车深度图", editor.last_stage1_prompt)
        self.assertIn("更贴近真实汽车", editor.last_refinement_prompt)
        self.assertEqual(result.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
