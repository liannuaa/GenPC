import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import scripts.run_render_to_moge_sim3 as render_sim3
from scripts.run_render_to_moge_sim3 import (
    anchor_alignment_stats,
    apply_sim3,
    apply_axis_scale_delta,
    choose_bridge_anchor_refinement,
    choose_icp_refinement,
    choose_partial_refinement,
    choose_retry_continuous_variant,
    choose_visible_3d_refinement,
    compose_complete_to_partial,
    evaluate_2d_acceptance,
    infer_paths,
    load_bridge_anchor_points,
    make_sim3,
    optimize_partial_affine_delta,
    optimize_partial_delta_sim3,
    apply_silhouette_refinement,
    optimize_silhouette_delta_sim3,
    optimize_visible_3d_delta_sim3,
    parse_scale_triplets,
    refine_transform_coordinate_search,
    refine_pca_anisotropic_partial,
    score_2d_anchor_objective,
    score_depth_render,
    score_2d_gate_objective,
    visible_distance_stats,
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

    def test_coordinate_search_can_select_2d_gate_objective(self):
        original_evaluate = render_sim3.evaluate_transform

        def fake_evaluate(_points, transform, _intrinsic, _depth, _mask, *, splat_radius):
            tx = float(transform[0, 3])
            if tx > 0.5:
                return {
                    "score": 0.5,
                    "iou": 0.9,
                    "coverage": 0.9,
                    "leakage": 0.01,
                    "edge_chamfer_px": 8.0,
                }
            if tx < -0.5:
                return {
                    "score": 1.0,
                    "iou": 0.5,
                    "coverage": 0.6,
                    "leakage": 0.25,
                    "edge_chamfer_px": 30.0,
                }
            return {
                "score": 0.7,
                "iou": 0.6,
                "coverage": 0.7,
                "leakage": 0.20,
                "edge_chamfer_px": 20.0,
            }

        render_sim3.evaluate_transform = fake_evaluate
        try:
            transform, score, _history = refine_transform_coordinate_search(
                np.eye(4, dtype=np.float64),
                np.zeros((1, 3), dtype=np.float64),
                np.eye(3, dtype=np.float64),
                np.ones((4, 4), dtype=np.float64),
                np.ones((4, 4), dtype=bool),
                splat_radius=0,
                translation_steps=[1.0],
                rotation_steps_deg=[],
                scale_steps=[],
                rounds=1,
                selection_objective="2d_gate",
            )
        finally:
            render_sim3.evaluate_transform = original_evaluate

        self.assertGreater(transform[0, 3], 0.5)
        self.assertEqual(score["score"], 0.5)
        self.assertGreater(score_2d_gate_objective(score), 1.0)

    def test_apply_silhouette_refinement_can_run_after_retry_transform(self):
        original_optimize = render_sim3.optimize_silhouette_delta_sim3
        original_evaluate = render_sim3.evaluate_transform
        calls = []

        def fake_optimize(initial_transform, *_args, **_kwargs):
            calls.append(float(initial_transform[0, 3]))
            optimized = np.asarray(initial_transform, dtype=np.float64).copy()
            optimized[0, 3] += 0.25
            return optimized, {"enabled": True}

        def fake_evaluate(_points, transform, _intrinsic, _depth, _mask, *, splat_radius):
            tx = float(transform[0, 3])
            return {
                "score": tx,
                "iou": min(1.0, tx),
                "coverage": min(1.0, tx),
                "leakage": 0.0,
                "edge_iou": 0.1,
                "edge_chamfer_px": 5.0,
            }

        render_sim3.optimize_silhouette_delta_sim3 = fake_optimize
        render_sim3.evaluate_transform = fake_evaluate
        try:
            retry_transform = np.eye(4, dtype=np.float64)
            retry_transform[0, 3] = 0.8
            refined, score, info = apply_silhouette_refinement(
                retry_transform,
                {"score": 0.8},
                np.zeros((1, 3), dtype=np.float64),
                np.zeros((1, 3), dtype=np.float64),
                np.eye(3, dtype=np.float64),
                (4, 4),
                np.ones((4, 4), dtype=bool),
                np.ones((4, 4), dtype=np.float64),
                args=type(
                    "Args",
                    (),
                    {
                        "silhouette_opt_render_size": 4,
                        "silhouette_opt_max_points": 8,
                        "silhouette_opt_iterations": 1,
                        "silhouette_opt_lr": 0.1,
                        "silhouette_opt_splat_radius": 1,
                        "silhouette_opt_sigma": 0.75,
                        "silhouette_opt_opacity": 0.08,
                        "silhouette_opt_leakage_weight": 1.0,
                        "silhouette_opt_miss_weight": 1.0,
                        "silhouette_opt_outside_distance_weight": 1.0,
                        "silhouette_opt_depth_weight": 0.5,
                        "silhouette_opt_boundary_weight": 1.0,
                        "silhouette_opt_area_weight": 0.1,
                        "silhouette_opt_center_weight": 0.1,
                        "silhouette_opt_transform_reg_weight": 0.01,
                        "silhouette_opt_min_score_gain": 0.0,
                        "splat_radius": 0,
                        "seed": 7,
                        "device": "cpu",
                    },
                )(),
                reason_prefix="post_retry",
            )
        finally:
            render_sim3.optimize_silhouette_delta_sim3 = original_optimize
            render_sim3.evaluate_transform = original_evaluate

        self.assertEqual(calls, [0.8])
        self.assertTrue(info["accepted"])
        self.assertEqual(info["reason"], "post_retry_render_score_improved")
        self.assertAlmostEqual(refined[0, 3], 1.05)
        self.assertAlmostEqual(score["score"], 1.05)

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

    def test_evaluate_2d_acceptance_rejects_low_iou_high_leakage(self):
        accepted = evaluate_2d_acceptance(
            {
                "iou": 0.6867,
                "coverage": 0.8162,
                "leakage": 0.1877,
                "edge_iou": 0.0208,
                "edge_chamfer_px": 16.59,
            },
            enabled=True,
            min_iou=0.78,
            min_coverage=0.80,
            max_leakage=0.12,
            min_edge_iou=0.02,
            max_edge_chamfer_px=18.0,
        )

        self.assertFalse(accepted["accepted"])
        self.assertEqual(accepted["reason"], "2d_threshold_not_met")
        self.assertIn("iou", accepted["failed"])
        self.assertIn("leakage", accepted["failed"])

    def test_evaluate_2d_acceptance_rejects_edge_threshold_failures(self):
        accepted = evaluate_2d_acceptance(
            {
                "iou": 0.862,
                "coverage": 0.868,
                "leakage": 0.007,
                "edge_iou": 0.0112,
                "edge_chamfer_px": 19.94,
            },
            enabled=True,
            min_iou=0.78,
            min_coverage=0.80,
            max_leakage=0.12,
            min_edge_iou=0.02,
            max_edge_chamfer_px=18.0,
        )

        self.assertFalse(accepted["accepted"])
        self.assertEqual(accepted["reason"], "2d_threshold_not_met")
        self.assertIn("edge_iou", accepted["failed"])
        self.assertIn("edge_chamfer_px", accepted["failed"])
        self.assertEqual(accepted["warnings"], [])

    def test_evaluate_2d_acceptance_rejects_07136_bad_boundary(self):
        accepted = evaluate_2d_acceptance(
            {
                "iou": 0.799960815047022,
                "coverage": 0.8279809748543617,
                "leakage": 0.04058701642706086,
                "edge_iou": 0.020970457676412063,
                "edge_chamfer_px": 24.442657947540283,
            },
            enabled=True,
            min_iou=0.82,
            min_coverage=0.84,
            max_leakage=0.10,
            min_edge_iou=0.025,
            max_edge_chamfer_px=18.0,
        )

        self.assertFalse(accepted["accepted"])
        self.assertIn("iou", accepted["failed"])
        self.assertIn("coverage", accepted["failed"])
        self.assertIn("edge_iou", accepted["failed"])
        self.assertIn("edge_chamfer_px", accepted["failed"])

    def test_score_2d_gate_objective_prefers_lower_leakage_over_composite_score(self):
        leaky = {
            "score": 1.2,
            "iou": 0.70,
            "coverage": 0.84,
            "leakage": 0.20,
            "edge_chamfer_px": 15.0,
        }
        cleaner = {
            "score": 1.1,
            "iou": 0.72,
            "coverage": 0.82,
            "leakage": 0.10,
            "edge_chamfer_px": 15.0,
        }

        self.assertGreater(score_2d_gate_objective(cleaner), score_2d_gate_objective(leaky))

    def test_score_2d_gate_objective_prefers_better_boundary_when_overlap_is_similar(self):
        weak_boundary = {
            "iou": 0.86,
            "coverage": 0.88,
            "leakage": 0.05,
            "edge_iou": 0.010,
            "edge_chamfer_px": 17.0,
        }
        better_boundary = {
            "iou": 0.85,
            "coverage": 0.87,
            "leakage": 0.05,
            "edge_iou": 0.035,
            "edge_chamfer_px": 17.5,
        }

        self.assertGreater(score_2d_gate_objective(better_boundary), score_2d_gate_objective(weak_boundary))

    def test_score_2d_anchor_objective_penalizes_worse_anchor_distance(self):
        same_score = {
            "iou": 0.72,
            "coverage": 0.84,
            "leakage": 0.15,
            "edge_chamfer_px": 14.0,
        }
        close_anchor = {"anchor_to_complete": {"mean": 0.02}}
        far_anchor = {"anchor_to_complete": {"mean": 0.08}}

        self.assertGreater(
            score_2d_anchor_objective(same_score, close_anchor, anchor_scale=0.5, anchor_weight=0.35),
            score_2d_anchor_objective(same_score, far_anchor, anchor_scale=0.5, anchor_weight=0.35),
        )

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

    def test_silhouette_optimizer_depth_loss_moves_points_toward_target_depth(self):
        transform = np.eye(4, dtype=np.float64)
        points = np.array(
            [
                [2.0, 2.0, 2.0],
                [2.2, 2.0, 2.0],
                [2.0, 2.2, 2.0],
                [2.2, 2.2, 2.0],
            ],
            dtype=np.float64,
        )
        target_depth = np.ones((8, 8), dtype=np.float64)

        optimized, info = optimize_silhouette_delta_sim3(
            transform,
            points,
            np.eye(3, dtype=np.float64),
            image_shape=(8, 8),
            target_mask=np.ones((8, 8), dtype=bool),
            target_depth=target_depth,
            render_size=8,
            max_points=4,
            iterations=50,
            lr=0.03,
            splat_radius=1,
            sigma=0.75,
            opacity=0.4,
            leakage_weight=0.0,
            miss_weight=0.0,
            outside_distance_weight=0.0,
            depth_weight=10.0,
            boundary_weight=0.0,
            area_weight=0.0,
            center_weight=0.0,
            transform_reg_weight=0.001,
            seed=1,
            device="cpu",
        )

        initial_mean_z = apply_sim3(points, transform)[:, 2].mean()
        optimized_mean_z = apply_sim3(points, optimized)[:, 2].mean()

        self.assertTrue(info["enabled"])
        self.assertLess(optimized_mean_z, initial_mean_z - 0.2)
        self.assertIn("depth_loss", info["best"])

    def test_visible_distance_stats_reports_basic_distance_summary(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float64)
        targets = np.zeros((2, 3), dtype=np.float64)

        stats = visible_distance_stats(points, targets)

        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["max"], 5.0)

    def test_load_bridge_anchor_points_filters_unmatched_partial_points_in_moge_frame(self):
        with TemporaryDirectory() as tmp:
            sample_dir = Path(tmp)
            np.save(sample_dir / "00001_moge_to_raw_partial_partial_to_moge_index.npy", np.array([2, -1, 5]))
            partial_points = np.array(
                [
                    [1.0, 2.0, 3.0],
                    [9.0, 9.0, 9.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=np.float64,
            )
            moge_to_partial = np.eye(4, dtype=np.float64)
            moge_to_partial[:3, 3] = [10.0, 0.0, 0.0]

            anchors, info = load_bridge_anchor_points(
                partial_points,
                sample_dir=sample_dir,
                flag="00001",
                moge_to_partial=moge_to_partial,
                max_points=10,
                seed=1,
            )

        np.testing.assert_allclose(anchors, [[-9.0, 2.0, 3.0], [-6.0, 5.0, 6.0]])
        self.assertTrue(info["enabled"])
        self.assertEqual(info["valid_matches"], 2)
        self.assertEqual(info["match_ratio"], 2 / 3)

    def test_anchor_alignment_stats_prefers_better_complete_to_moge_transform(self):
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        anchors = source + np.array([2.0, 0.0, 0.0], dtype=np.float64)
        bad = np.eye(4, dtype=np.float64)
        good = np.eye(4, dtype=np.float64)
        good[:3, 3] = [2.0, 0.0, 0.0]

        bad_stats = anchor_alignment_stats(source, anchors, bad, max_points=10, seed=1)
        good_stats = anchor_alignment_stats(source, anchors, good, max_points=10, seed=1)

        self.assertLess(good_stats["anchor_to_complete"]["mean"], bad_stats["anchor_to_complete"]["mean"])
        self.assertLess(good_stats["complete_to_anchor_trim70"]["mean"], bad_stats["complete_to_anchor_trim70"]["mean"])

    def test_choose_bridge_anchor_refinement_accepts_anchor_gain_with_2d_guard(self):
        baseline = np.eye(4, dtype=np.float64)
        candidate = np.eye(4, dtype=np.float64)
        candidate[:3, 3] = [0.01, 0.0, 0.0]
        optimization = {
            "initial_anchor": {"anchor_to_complete": {"mean": 1.0}},
            "candidate_anchor": {"anchor_to_complete": {"mean": 0.8}},
        }

        chosen, score, info = choose_bridge_anchor_refinement(
            baseline,
            {"iou": 0.72, "coverage": 0.84, "leakage": 0.15, "edge_chamfer_px": 14.0, "score": 0.6},
            candidate,
            {"iou": 0.73, "coverage": 0.84, "leakage": 0.15, "edge_chamfer_px": 14.0, "score": 0.61},
            optimization,
            min_2d_objective_gain=0.0,
            max_2d_objective_drop=0.02,
            min_anchor_improvement=0.05,
        )

        np.testing.assert_allclose(chosen, candidate)
        self.assertEqual(score["score"], 0.61)
        self.assertTrue(info["accepted"])

    def test_choose_bridge_anchor_refinement_rejects_2d_drop_even_if_anchor_improves(self):
        baseline = np.eye(4, dtype=np.float64)
        candidate = np.eye(4, dtype=np.float64)
        optimization = {
            "initial_anchor": {"anchor_to_complete": {"mean": 1.0}},
            "candidate_anchor": {"anchor_to_complete": {"mean": 0.7}},
        }

        chosen, score, info = choose_bridge_anchor_refinement(
            baseline,
            {"iou": 0.75, "coverage": 0.84, "leakage": 0.12, "edge_chamfer_px": 12.0, "score": 0.8},
            candidate,
            {"iou": 0.65, "coverage": 0.80, "leakage": 0.22, "edge_chamfer_px": 18.0, "score": 0.5},
            optimization,
            min_2d_objective_gain=0.0,
            max_2d_objective_drop=0.02,
            min_anchor_improvement=0.05,
        )

        np.testing.assert_allclose(chosen, baseline)
        self.assertEqual(score["score"], 0.8)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "2d_objective_drop")

    def test_choose_retry_continuous_variant_replaces_coordinate_result_when_joint_objective_improves(self):
        coordinate = np.eye(4, dtype=np.float64)
        continuous = np.eye(4, dtype=np.float64)
        continuous[:3, 3] = [0.01, 0.0, 0.0]
        coordinate_score = {
            "iou": 0.70,
            "coverage": 0.82,
            "leakage": 0.18,
            "edge_chamfer_px": 15.0,
            "score": 1.0,
        }
        continuous_score = {
            "iou": 0.71,
            "coverage": 0.82,
            "leakage": 0.17,
            "edge_chamfer_px": 15.0,
            "score": 1.1,
        }
        coordinate_anchor = {"anchor_to_complete": {"mean": 0.05}}
        continuous_info = {
            "enabled": True,
            "initial_anchor": {"anchor_to_complete": {"mean": 0.05}},
            "candidate_anchor": {"anchor_to_complete": {"mean": 0.04}},
        }

        chosen, score, anchor, info = choose_retry_continuous_variant(
            coordinate,
            coordinate_score,
            coordinate_anchor,
            continuous,
            continuous_score,
            continuous_info,
            anchor_scale=1.0,
            anchor_weight=0.35,
            max_2d_objective_drop=0.02,
            min_anchor_improvement=0.05,
        )

        np.testing.assert_allclose(chosen, continuous)
        self.assertEqual(score["score"], 1.1)
        self.assertEqual(anchor["anchor_to_complete"]["mean"], 0.04)
        self.assertTrue(info["accepted"])

    def test_choose_retry_continuous_variant_keeps_coordinate_result_on_2d_drop(self):
        coordinate = np.eye(4, dtype=np.float64)
        continuous = np.eye(4, dtype=np.float64)
        coordinate_score = {
            "iou": 0.72,
            "coverage": 0.84,
            "leakage": 0.15,
            "edge_chamfer_px": 14.0,
            "score": 1.0,
        }
        continuous_score = {
            "iou": 0.62,
            "coverage": 0.72,
            "leakage": 0.10,
            "edge_chamfer_px": 18.0,
            "score": 0.8,
        }
        coordinate_anchor = {"anchor_to_complete": {"mean": 0.05}}
        continuous_info = {
            "enabled": True,
            "initial_anchor": {"anchor_to_complete": {"mean": 0.05}},
            "candidate_anchor": {"anchor_to_complete": {"mean": 0.03}},
        }

        chosen, score, anchor, info = choose_retry_continuous_variant(
            coordinate,
            coordinate_score,
            coordinate_anchor,
            continuous,
            continuous_score,
            continuous_info,
            anchor_scale=1.0,
            anchor_weight=0.35,
            max_2d_objective_drop=0.02,
            min_anchor_improvement=0.05,
        )

        np.testing.assert_allclose(chosen, coordinate)
        self.assertEqual(score["score"], 1.0)
        self.assertEqual(anchor["anchor_to_complete"]["mean"], 0.05)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "2d_objective_drop")

    def test_parse_scale_triplets_reads_semicolon_separated_triplets(self):
        self.assertEqual(
            parse_scale_triplets("1,0.85,1;1.08,0.95,1.08"),
            [(1.0, 0.85, 1.0), (1.08, 0.95, 1.08)],
        )

    def test_parse_args_accepts_continuous_affine_partial_refinement(self):
        original_argv = sys.argv
        sys.argv = [
            "run_render_to_moge_sim3.py",
            "--partial_refine_mode",
            "continuous_affine",
        ]
        try:
            args = render_sim3.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.partial_refine_mode, "continuous_affine")

    def test_apply_axis_scale_delta_scales_about_center(self):
        transform = np.eye(4, dtype=np.float64)
        axes = np.eye(3, dtype=np.float64)
        scaled = apply_axis_scale_delta(
            transform,
            center=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            axes=axes,
            factors=(2.0, 1.0, 1.0),
        )

        moved = apply_sim3(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), scaled)

        np.testing.assert_allclose(moved, [[3.0, 0.0, 0.0]])

    def test_pca_anisotropic_partial_refinement_disabled_without_triplets(self):
        transform = np.eye(4, dtype=np.float64)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        )

        optimized, info = refine_pca_anisotropic_partial(
            transform,
            points,
            points,
            scale_triplets=[],
            pre_icp_iterations=0,
            icp_iterations=1,
            max_pairs=4,
            complete_trim_quantile=0.5,
            partial_trim_quantile=1.0,
            partial_weight=1.0,
            max_step_translation=0.1,
            min_step_scale=0.8,
            max_step_scale=1.2,
            objective_pc_p95_weight=0.3,
            objective_cp70_weight=0.2,
            objective_scale_reg_weight=0.004,
            seed=1,
        )

        np.testing.assert_allclose(optimized, transform)
        self.assertFalse(info["enabled"])

    def test_continuous_affine_partial_refinement_reduces_symmetric_distance(self):
        coords = np.linspace(-0.15, 0.15, 5)
        source = np.array([[x, y, z] for x in coords for y in coords for z in coords], dtype=np.float64)
        target = source * np.array([1.08, 0.92, 1.04], dtype=np.float64)
        target += np.array([0.018, -0.012, 0.01], dtype=np.float64)

        optimized, info = optimize_partial_affine_delta(
            np.eye(4, dtype=np.float64),
            source,
            target,
            pre_icp_iterations=0,
            max_pairs=200,
            complete_trim_quantile=1.0,
            partial_trim_quantile=1.0,
            partial_weight=1.0,
            max_step_translation=0.08,
            min_step_scale=0.85,
            max_step_scale=1.15,
            iterations=80,
            lr=0.02,
            distance_weight=1.0,
            transform_reg_weight=0.01,
            axis_scale_reg_weight=0.001,
            seed=7,
            device="cpu",
        )

        self.assertTrue(info["enabled"])
        self.assertLess(
            info["candidate_alignment"]["partial_to_complete"]["mean"],
            info["initial_alignment"]["partial_to_complete"]["mean"] * 0.8,
        )
        self.assertLess(
            info["candidate_alignment"]["complete_to_partial_trim70"]["mean"],
            info["initial_alignment"]["complete_to_partial_trim70"]["mean"] * 0.8,
        )
        self.assertGreater(np.linalg.det(optimized[:3, :3]), 0.0)

    def test_visible_3d_optimizer_disabled_returns_initial_transform(self):
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

        optimized, info = optimize_visible_3d_delta_sim3(
            transform,
            points,
            points,
            np.eye(3, dtype=np.float64),
            image_shape=(8, 8),
            target_mask=np.ones((8, 8), dtype=bool),
            max_depth_delta=0.015,
            max_pairs=4,
            trim_quantile=0.7,
            iterations=0,
            lr=0.01,
            distance_loss="smooth_l1",
            distance_weight=1.0,
            silhouette_weight=0.35,
            silhouette_render_size=8,
            silhouette_points=4,
            silhouette_splat_radius=1,
            silhouette_sigma=0.75,
            silhouette_opacity=0.1,
            leakage_weight=0.5,
            miss_weight=0.25,
            transform_reg_weight=0.01,
            seed=1,
            device="cpu",
        )

        np.testing.assert_allclose(optimized, transform)
        self.assertFalse(info["enabled"])

    def test_choose_visible_3d_refinement_accepts_distance_gain_with_score_guard(self):
        baseline = np.eye(4, dtype=np.float64)
        candidate = np.eye(4, dtype=np.float64)
        candidate[:3, 3] = [0.01, 0.0, 0.0]
        optimization = {
            "initial_distance": {"mean": 1.0},
            "candidate_distance": {"mean": 0.95},
        }

        chosen, score, info = choose_visible_3d_refinement(
            baseline,
            {"score": 1.0},
            candidate,
            {"score": 0.99},
            optimization,
            max_score_drop=0.02,
            min_distance_improvement=0.02,
        )

        np.testing.assert_allclose(chosen, candidate)
        self.assertEqual(score["score"], 0.99)
        self.assertTrue(info["accepted"])

    def test_choose_visible_3d_refinement_rejects_render_score_drop(self):
        baseline = np.eye(4, dtype=np.float64)
        candidate = np.eye(4, dtype=np.float64)
        candidate[:3, 3] = [0.01, 0.0, 0.0]
        optimization = {
            "initial_distance": {"mean": 1.0},
            "candidate_distance": {"mean": 0.9},
        }

        chosen, score, info = choose_visible_3d_refinement(
            baseline,
            {"score": 1.0},
            candidate,
            {"score": 0.9},
            optimization,
            max_score_drop=0.02,
            min_distance_improvement=0.02,
        )

        np.testing.assert_allclose(chosen, baseline)
        self.assertEqual(score["score"], 1.0)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "render_score_drop")

    def test_partial_optimizer_disabled_returns_initial_transform(self):
        transform = np.eye(4, dtype=np.float64)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        )

        optimized, info = optimize_partial_delta_sim3(
            transform,
            points,
            points,
            max_pairs=4,
            trim_quantile=0.5,
            iterations=0,
            lr=0.01,
            distance_loss="smooth_l1",
            distance_weight=1.0,
            transform_reg_weight=0.01,
            seed=1,
            device="cpu",
        )

        np.testing.assert_allclose(optimized, transform)
        self.assertFalse(info["enabled"])

    def test_choose_partial_refinement_rejects_large_delta(self):
        baseline = np.eye(4, dtype=np.float64)
        candidate = np.eye(4, dtype=np.float64)
        delta = np.eye(4, dtype=np.float64)
        delta[:3, 3] = [1.0, 0.0, 0.0]
        optimization = {
            "initial_distance": {"mean": 1.0},
            "candidate_distance": {"mean": 0.8},
            "delta": delta.tolist(),
        }

        chosen, info = choose_partial_refinement(
            baseline,
            candidate,
            optimization,
            min_distance_improvement=0.02,
            max_delta_rotation_deg=12.0,
            max_delta_translation=0.12,
            min_delta_scale=0.9,
            max_delta_scale=1.1,
        )

        np.testing.assert_allclose(chosen, baseline)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "delta_translation_out_of_bounds")

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
