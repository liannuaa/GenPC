import unittest

import numpy as np

from scripts.run_render_to_moge_sim3 import (
    apply_sim3,
    compose_complete_to_partial,
    infer_paths,
    make_sim3,
    score_depth_render,
    zbuffer_depth,
)


class RenderToMogeSim3Test(unittest.TestCase):
    def test_make_sim3_applies_scale_rotation_then_translation(self):
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        transform = make_sim3(scale=2.0, rotation=rotation, translation=[1.0, 2.0, 3.0])
        moved = apply_sim3(np.array([[1.0, 0.0, 0.5]], dtype=np.float64), transform)

        np.testing.assert_allclose(moved, [[1.0, 4.0, 4.0]])

    def test_zbuffer_depth_keeps_nearest_positive_depth_per_pixel(self):
        uv = np.array(
            [
                [1.2, 1.1],
                [1.4, 1.2],
                [2.0, 0.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        )
        depth = np.array([3.0, 2.0, -1.0, 4.0], dtype=np.float64)

        rendered, mask = zbuffer_depth(uv, depth, image_shape=(4, 4), splat_radius=0)

        self.assertTrue(mask[1, 1])
        self.assertEqual(rendered[1, 1], 2.0)
        self.assertFalse(mask[0, 2])
        self.assertEqual(rendered[3, 3], 4.0)

    def test_score_depth_render_rewards_overlap_and_depth_agreement(self):
        target_mask = np.array([[1, 1], [0, 0]], dtype=bool)
        target_depth = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
        good_depth = np.array([[1.1, 1.9], [0.0, 0.0]], dtype=np.float64)
        bad_depth = np.array([[4.0, 5.0], [3.0, 3.0]], dtype=np.float64)
        good_mask = np.array([[1, 1], [0, 0]], dtype=bool)
        bad_mask = np.array([[1, 0], [1, 1]], dtype=bool)

        good = score_depth_render(good_depth, good_mask, target_depth, target_mask)
        bad = score_depth_render(bad_depth, bad_mask, target_depth, target_mask)

        self.assertGreater(good["score"], bad["score"])
        self.assertLess(good["depth_mae"], bad["depth_mae"])
        self.assertGreater(good["iou"], bad["iou"])

    def test_compose_complete_to_partial_left_multiplies_moge_to_partial(self):
        complete_to_moge = np.eye(4, dtype=np.float64)
        complete_to_moge[:3, 3] = [1.0, 0.0, 0.0]
        moge_to_partial = np.eye(4, dtype=np.float64)
        moge_to_partial[:3, 3] = [0.0, 2.0, 0.0]

        composed = compose_complete_to_partial(moge_to_partial, complete_to_moge)

        np.testing.assert_allclose(composed[:3, 3], [1.0, 2.0, 0.0])

    def test_infer_paths_uses_flag_for_default_sample_files(self):
        paths = infer_paths(
            flag="12345",
            sample_root="/tmp/samples",
            sample_dir=None,
            out_root="/tmp/out",
            out_dir=None,
            image_name=None,
            complete_name=None,
            object_mask_name=None,
            moge_to_partial_name=None,
        )

        self.assertEqual(str(paths.sample_dir), "/tmp/samples/12345")
        self.assertEqual(str(paths.out_dir), "/tmp/out/render_to_moge_sim3_12345")
        self.assertEqual(paths.image_path.name, "img.png")
        self.assertEqual(paths.complete_path.name, "12345_hunyuan2.1.ply")
        self.assertEqual(paths.object_mask_path.name, "12345_moge_to_raw_partial_object_mask.png")
        self.assertEqual(
            paths.moge_to_partial_path.name,
            "12345_moge_to_raw_partial_moge_to_raw_partial_transform.npy",
        )


if __name__ == "__main__":
    unittest.main()
