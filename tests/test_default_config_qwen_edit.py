import unittest
from pathlib import Path

from munch import Munch
import yaml

from utils.dataUtils import resolve_prompt_label
from tools.qwen_image_edit import build_refinement_prompt


class DefaultConfigQwenEditTest(unittest.TestCase):
    def test_default_qwen_edit_uses_manual_2511_experiment_assets(self):
        config = yaml.safe_load(Path("configs/config.yaml").read_text())

        self.assertEqual(
            config["models"]["qwen_edit_pipeline_path"],
            "Qwen-Image-Edit-2511",
        )
        self.assertEqual(
            config["models"]["qwen_edit_transformer_path"],
            "nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors",
        )
        self.assertNotIn("qwen_transformer_path", config["models"])
        self.assertNotIn("qwen_pipeline_path", config["models"])
        self.assertNotIn("qwen_controlnet_path", config["models"])
        self.assertEqual(config["qwen_edit_steps"], 16)
        self.assertEqual(config["qwen_edit_generate_res"], 1024)
        self.assertEqual(config["qwen_edit_true_cfg_scale"], 4.0)
        self.assertEqual(config["qwen_edit_negative_prompt"], " ")
        self.assertTrue(config["qwen_edit_refine_stage"])
        self.assertEqual(config["qwen_edit_refine_steps"], 16)
        self.assertNotIn("qwen_controlnet_conditioning_scale", config)
        self.assertEqual(config["depth_projection"], "view_select")
        self.assertFalse(config["save_depth_view_point_cloud"])
        self.assertEqual(config["reg_backend"], "render_to_moge_sim3")
        self.assertNotIn("semantic_view_selector", config)
        for key in config:
            self.assertFalse(key.startswith("semantic_view_"), key)

    def test_redwood_05117_prompt_label_is_red_chair(self):
        config = Munch.fromDict(yaml.safe_load(Path("configs/config.yaml").read_text()))

        self.assertEqual(resolve_prompt_label("05117", config), "red chair")

    def test_redwood_06127_prompt_label_uses_dataset_category(self):
        config = Munch.fromDict(yaml.safe_load(Path("configs/config.yaml").read_text()))

        self.assertEqual(
            resolve_prompt_label("06127", config),
            "a vase with leafy plant",
        )

    def test_qwen_refinement_prompt_discourages_environment_foreground(self):
        prompt = build_refinement_prompt("a vase with leafy plant")

        self.assertIn("干净", prompt)
        self.assertIn("普通摄影棚背景", prompt)
        self.assertIn("不要生成桌面", prompt)
        self.assertNotIn("背景为真实场景", prompt)


if __name__ == "__main__":
    unittest.main()
