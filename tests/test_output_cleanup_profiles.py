import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.runtime import cleanup_intermediates, cleanup_stage1_intermediates


def make_cfg(root, keep_profile="lean", save_intermediates=False):
    return SimpleNamespace(
        paths=SimpleNamespace(output_dir=str(root)),
        outputs=SimpleNamespace(
            keep_profile=keep_profile,
            save_intermediates=save_intermediates,
        ),
    )


class OutputCleanupProfileTests(unittest.TestCase):
    def test_lean_cleanup_keeps_useful_pipeline_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "01184"
            sample.mkdir()
            files = [
                "depth.png",
                "img.png",
                "camera.pth",
                "point_uv.npy",
                "qwen_edit_prompt.txt",
                "img_sam.png",
                "01184_hunyuan2.1.ply",
                "01184_moge_to_raw_partial_moge_object_only.ply",
                "01184_moge_to_raw_partial_partial_to_moge_index.npy",
                "01184_moge_to_raw_partial_moge_to_raw_partial_transform.npy",
                "01184_moge_to_raw_partial_info.json",
                "01184_moge_to_raw_partial_object_mask.png",
                "01184_moge_to_raw_partial_raw_partial_gray_moge_red_aligned.ply",
                "01184_freereg_original_depthpro_objectmask_complete_registered_to_object_depthpro.ply",
                "01184_freereg_original_depthpro_objectmask_gray_object_depthpro_blue_complete_fused.ply",
                "01184_freereg_original_depthpro_objectmask_info.json",
                "01184_complete_registered_to_moge.ply",
                "01184_moge_gray_complete_blue_fused.ply",
                "01184_complete_aligned_to_raw_partial.ply",
                "01184_raw_partial_gray_complete_blue_aligned.ply",
                "01184_complete_to_moge_transform.npy",
                "01184_complete_to_partial_transform.npy",
                "01184_render_to_moge_overlay.png",
                "01184_render_to_moge_sim3_info.json",
                "01184_fused.ply",
                "raw_depth.png",
                "mask.png",
                "viewpoint.npy",
                "qwen_edit_stage1.png",
                "qwen_edit_stage1_prompt.txt",
                "color_point.ply",
                "01184_freereg_original_depthpro_gray_image_blue_complete_fused.ply",
                "01184_moge_to_raw_partial_moge_object_partial_hits_red.ply",
                "01184_moge_to_raw_partial_raw_partial_camera_points.npy",
                "01184_moge_to_raw_partial_rmbg.png",
            ]
            for name in files:
                (sample / name).write_text("x")

            cleanup_intermediates(make_cfg(root), "01184")

            kept = {path.name for path in sample.iterdir()}
            self.assertIn("depth.png", kept)
            self.assertIn("img.png", kept)
            self.assertIn("camera.pth", kept)
            self.assertIn("01184_hunyuan2.1.ply", kept)
            self.assertIn("01184_moge_to_raw_partial_moge_object_only.ply", kept)
            self.assertIn("01184_moge_to_raw_partial_partial_to_moge_index.npy", kept)
            self.assertIn("01184_moge_to_raw_partial_moge_to_raw_partial_transform.npy", kept)
            self.assertIn("01184_freereg_original_depthpro_objectmask_gray_object_depthpro_blue_complete_fused.ply", kept)
            self.assertIn("01184_complete_registered_to_moge.ply", kept)
            self.assertIn("01184_complete_aligned_to_raw_partial.ply", kept)
            self.assertIn("01184_complete_to_partial_transform.npy", kept)
            self.assertIn("01184_render_to_moge_sim3_info.json", kept)
            self.assertIn("01184_fused.ply", kept)
            self.assertNotIn("qwen_edit_stage1.png", kept)
            self.assertNotIn("raw_depth.png", kept)
            self.assertNotIn("01184_freereg_original_depthpro_gray_image_blue_complete_fused.ply", kept)
            self.assertNotIn("01184_moge_to_raw_partial_moge_object_partial_hits_red.ply", kept)

    def test_stage1_lean_cleanup_keeps_camera_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "abc"
            sample.mkdir()
            for name in [
                "depth.png",
                "img.png",
                "camera.pth",
                "point_uv.npy",
                "qwen_edit_prompt.txt",
                "qwen_edit_stage1.png",
                "raw_depth.png",
            ]:
                (sample / name).write_text("x")

            cleanup_stage1_intermediates(make_cfg(root), "abc")

            kept = {path.name for path in sample.iterdir()}
            self.assertEqual(
                kept,
                {
                    "depth.png",
                    "img.png",
                    "camera.pth",
                    "point_uv.npy",
                    "qwen_edit_prompt.txt",
                },
            )

    def test_save_intermediates_keeps_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "abc"
            sample.mkdir()
            (sample / "debug.tmp").write_text("x")

            cleanup_intermediates(make_cfg(root, save_intermediates=True), "abc")

            self.assertTrue((sample / "debug.tmp").exists())


if __name__ == "__main__":
    unittest.main()
