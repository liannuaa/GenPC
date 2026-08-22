import math
import unittest

import numpy as np
import torch

from scripts.run_registration_deformation_fusion_ablation import (
    build_deformation_graph,
    clamp_deformation_parameters,
    evaluate_raw_gate_from_projection,
    graph_points_numpy,
    partial_priority_fusion,
)
from scripts.run_render_to_moge_sim3 import (
    project_points,
    rank_refined_sim3_candidates,
    zbuffer_depth,
    zbuffer_depth_with_indices,
)


class FakeProjector:
    image_shape = (16, 16)

    def project(self, points):
        points = np.asarray(points, dtype=np.float64)
        uv = points[:, :2] * 2.0 + np.array([4.0, 4.0])
        return uv, points[:, 2]


class RegistrationDeformationFusionAblationTest(unittest.TestCase):
    def test_zbuffer_depth_with_indices_keeps_nearest_source(self):
        uv = np.array([[1.0, 1.0], [1.2, 1.1], [3.0, 3.0]])
        depth = np.array([2.0, 1.0, 4.0])
        rendered, mask, indices = zbuffer_depth_with_indices(
            uv, depth, (5, 5), splat_radius=0
        )

        self.assertTrue(mask[1, 1])
        self.assertEqual(indices[1, 1], 1)
        self.assertEqual(rendered[1, 1], 1.0)
        self.assertEqual(indices[3, 3], 2)

    def test_raw_gate_accepts_full_match_and_rejects_local_only_match(self):
        pixel = []
        points = []
        for gy in range(8):
            for gx in range(8):
                x = gx * 2 + 1
                y = gy * 2 + 1
                pixel.append([x, y])
                points.append([gx / 7.0, gy / 7.0, 1.0])
        pixel = np.asarray(pixel, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        depth = np.ones(len(points), dtype=np.float64)
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        diagonal = float(np.linalg.norm(points.max(0) - points.min(0)))

        good, partial_ids, complete_ids = evaluate_raw_gate_from_projection(
            points,
            points.copy(),
            pixel,
            depth,
            pixel.copy(),
            depth.copy(),
            image_shape=(16, 16),
            bbox_diagonal=diagonal,
            partial_normals=normals,
            complete_normals=normals,
        )
        self.assertTrue(good["accepted"])
        self.assertEqual(len(partial_ids), 64)
        self.assertEqual(len(complete_ids), 64)
        self.assertAlmostEqual(good["grid_coverage"], 1.0)

        local = np.arange(16)
        bad, _, _ = evaluate_raw_gate_from_projection(
            points,
            points[local],
            pixel,
            depth,
            pixel[local],
            depth[local],
            image_shape=(16, 16),
            bbox_diagonal=diagonal,
            partial_normals=normals,
            complete_normals=normals[local],
        )
        self.assertFalse(bad["accepted"])
        self.assertIn("coverage", bad["failed"])
        self.assertIn("grid_coverage", bad["failed"])

    def test_zero_deformation_graph_is_identity(self):
        rng = np.random.default_rng(6145)
        points = rng.normal(size=(160, 3))
        graph = build_deformation_graph(
            points,
            node_spacing_ratio=0.1,
            min_nodes=16,
            max_nodes=32,
            graph_knn=4,
            influence_k=4,
        )
        state = {
            "rotvec": np.zeros((len(graph["nodes"]), 3), dtype=np.float32),
            "translations": np.zeros((len(graph["nodes"]), 3), dtype=np.float32),
            "log_scales": np.zeros(len(graph["nodes"]), dtype=np.float32),
        }
        moved = graph_points_numpy(graph, state)
        np.testing.assert_allclose(moved, points, atol=2e-6)

    def test_deformation_parameter_clamps_are_enforced(self):
        rotvec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        translations = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
        log_scales = torch.tensor([math.log(2.0)], dtype=torch.float32)
        clamp_deformation_parameters(
            rotvec,
            translations,
            log_scales,
            allow_scale=True,
        )
        self.assertLessEqual(float(torch.linalg.norm(rotvec[0])), 0.350001)
        self.assertLessEqual(float(torch.linalg.norm(translations[0])), 0.080001)
        self.assertLessEqual(float(torch.exp(log_scales[0])), 1.150001)

        clamp_deformation_parameters(
            rotvec,
            translations,
            log_scales,
            allow_scale=False,
        )
        self.assertEqual(float(log_scales[0]), 0.0)

    def test_partial_priority_fusion_preserves_partial_and_removes_duplicates(self):
        partial = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        generated = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [3.0, 3.0, 1.0]]
        )
        fused, info = partial_priority_fusion(
            generated,
            partial,
            FakeProjector(),
            bbox_diagonal=5.0,
            remove_ratio=0.015,
            voxel_ratio=0.003,
        )

        np.testing.assert_allclose(fused[: len(partial)], partial)
        self.assertEqual(info["generated_removed_points"], 2)
        self.assertTrue(np.any(np.linalg.norm(fused - [3.0, 3.0, 1.0], axis=1) < 1e-8))

    def test_top_k_candidate_ranking_is_deterministic(self):
        source = np.array(
            [
                [-0.5, -0.5, 1.0],
                [0.5, -0.5, 1.0],
                [-0.5, 0.5, 1.0],
                [0.5, 0.5, 1.0],
            ]
        )
        intrinsic = np.array(
            [[2.0, 0.0, 4.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]]
        )
        uv, valid = project_points(source, intrinsic, (8, 8))
        target_depth, target_mask = zbuffer_depth(
            uv[valid], source[valid, 2], (8, 8), splat_radius=1
        )
        candidates = [np.eye(4), np.eye(4)]
        first = rank_refined_sim3_candidates(
            candidates,
            source,
            intrinsic,
            target_depth,
            target_mask,
            top_k=2,
            splat_radius=1,
            translation_steps=(),
            rotation_steps_deg=(),
            scale_steps=(),
            rounds=0,
            require_2d_acceptance=False,
        )
        second = rank_refined_sim3_candidates(
            candidates,
            source,
            intrinsic,
            target_depth,
            target_mask,
            top_k=2,
            splat_radius=1,
            translation_steps=(),
            rotation_steps_deg=(),
            scale_steps=(),
            rounds=0,
            require_2d_acceptance=False,
        )
        self.assertEqual([item["index"] for item in first], [0, 1])
        self.assertEqual(
            [item["index"] for item in first],
            [item["index"] for item in second],
        )


if __name__ == "__main__":
    unittest.main()
