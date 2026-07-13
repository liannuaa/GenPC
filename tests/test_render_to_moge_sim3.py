import unittest

import numpy as np

from scripts.run_render_to_moge_sim3 import (
    apply_sim3,
    choose_icp_refinement,
    compose_complete_to_partial,
    infer_paths,
    make_sim3,
    optimize_silhouette_delta_sim3,
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

    def test_score_depth_render_penalizes_edge_misalignment(self):
        target_mask = np.zeros((8, 8), dtype=bool)
        target_mask[2:6, 2:6] = True
        target_depth = np.ones((8, 8), dtype=np.float64)
        aligned_mask = target_mask.copy()
        shifted_mask = np.zeros((8, 8), dtype=bool)
        shifted_mask[2:6, 3:7] = True

        aligned = score_depth_render(target_depth, aligned_mask, target_depth, target_mask)
        shifted = score_depth_render(target_depth, shifted_mask, target_depth, target_mask)

        self.assertGreater(aligned["score"], shifted["score"])
        self.assertGreater(aligned["edge_iou"], shifted["edge_iou"])
        self.assertLess(aligned["edge_chamfer_norm"], shifted["edge_chamfer_norm"])

    def test_choose_icp_refinement_rolls_back_when_render_score_drops(self):
        baseline = np.eye(4, dtype=np.float64)
        icp = np.eye(4, dtype=np.float64)
        icp[:3, 3] = [1.0, 0.0, 0.0]

        chosen, final_score, acceptance = choose_icp_refinement(
            baseline,
            {"score": 1.0},
            icp,
            {"score": 0.9},
            rollback_on_score_drop=True,
        )

        np.testing.assert_allclose(chosen, baseline)
        self.assertEqual(final_score["score"], 1.0)
        self.assertFalse(acceptance["accepted"])

    def test_choose_icp_refinement_accepts_when_render_score_improves(self):
        baseline = np.eye(4, dtype=np.float64)
        icp = np.eye(4, dtype=np.float64)
        icp[:3, 3] = [1.0, 0.0, 0.0]

        chosen, final_score, acceptance = choose_icp_refinement(
            baseline,
            {"score": 1.0},
            icp,
            {"score": 1.1},
            rollback_on_score_drop=True,
        )

        np.testing.assert_allclose(chosen, icp)
        self.assertEqual(final_score["score"], 1.1)
        self.assertTrue(acceptance["accepted"])

    def test_silhouette_optimizer_disabled_returns_initial_transform(self):
        transform = np.eye(4, dtype=np.float64)
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, 0.1, 1.0],
            ],
            dtype=np.float64,
        )
        intrinsic = np.eye(3, dtype=np.float64)
        mask = np.ones((8, 8), dtype=bool)

        optimized, info = optimize_silhouette_delta_sim3(
            transform,
            points,
            intrinsic,
            image_shape=(8, 8),
            target_mask=mask,
            render_size=8,
            max_points=4,
            iterations=0,
            lr=0.01,
            splat_radius=1,
            sigma=0.75,
            opacity=0.1,
            leakage_weight=1.0,
            miss_weight=1.0,
            outside_distance_weight=1.0,
            area_weight=0.1,
            center_weight=1.0,
            transform_reg_weight=0.01,
            seed=1,
            device="cpu",
        )

        np.testing.assert_allclose(optimized, transform)
        self.assertFalse(info["enabled"])

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
