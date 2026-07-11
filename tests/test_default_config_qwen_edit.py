import unittest
from pathlib import Path

import yaml


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
        self.assertEqual(config["qwen_edit_steps"], 40)
        self.assertEqual(config["qwen_edit_true_cfg_scale"], 4.0)
        self.assertEqual(config["qwen_edit_generate_res"], 1024)


if __name__ == "__main__":
    unittest.main()
