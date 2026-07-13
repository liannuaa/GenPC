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
    def test_completion_prompt_requests_completed_depth_from_incomplete_depth(self):
        prompt = build_completion_prompt("car")

        self.assertEqual(prompt, "这是一个汽车的深度图，补全它")
        self.assertNotIn("真实汽车照片", prompt)

    def test_completion_prompt_uses_object_parameter(self):
        prompt = build_completion_prompt("chair")

        self.assertEqual(prompt, "这是一个chair的深度图，补全它")
        self.assertNotIn("汽车", prompt)

    def test_refinement_prompt_keeps_only_shape_pose_category_and_camera(self):
        prompt = build_refinement_prompt("red chair")

        self.assertIn("根据这张完整的red chair深度图生成真实red chair照片", prompt)
        self.assertIn("只保留物体的轮廓、大小、种类、朝向、姿态和相机视角", prompt)
        self.assertIn("严格保持输入图中的2D投影轮廓、物体位置和大小", prompt)
        self.assertIn("不要旋转、平移、缩放、换视角或重新构图", prompt)
        self.assertIn("不需要保留原图的颜色、材质、光照和背景细节", prompt)
        self.assertIn("普通摄影棚背景", prompt)

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
        self.assertEqual(calls[0]["prompt"], "这是一个汽车的深度图，补全它")
        self.assertEqual(calls[0]["negative_prompt"], " ")
        self.assertEqual(calls[0]["true_cfg_scale"], 4.0)
        self.assertEqual(calls[0]["num_inference_steps"], 16)
        self.assertIsInstance(calls[1]["image"], list)
        self.assertEqual(len(calls[1]["image"]), 1)
        self.assertEqual(calls[1]["image"][0].size, (1024, 1024))
        self.assertEqual(calls[1]["image"][0].mode, "RGB")
        self.assertIn("根据这张完整的汽车深度图生成真实汽车照片", calls[1]["prompt"])
        self.assertIn("只保留物体的轮廓、大小、种类、朝向、姿态和相机视角", calls[1]["prompt"])
        self.assertIn("不要旋转、平移、缩放、换视角或重新构图", calls[1]["prompt"])
        self.assertIn("普通摄影棚背景", calls[1]["prompt"])
        self.assertEqual(calls[1]["negative_prompt"], " ")
        self.assertEqual(calls[1]["true_cfg_scale"], 4.0)
        self.assertEqual(calls[1]["num_inference_steps"], 16)
        self.assertEqual(editor.last_stage1_image.size, (1024, 1024))
        self.assertEqual(editor.last_stage1_prompt, "这是一个汽车的深度图，补全它")
        self.assertIn("完整的汽车深度图生成真实汽车照片", editor.last_refinement_prompt)
        self.assertEqual(result.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
