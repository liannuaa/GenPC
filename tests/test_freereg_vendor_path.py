import sys
import unittest
from pathlib import Path

import numpy as np

from scripts.run_freereg_original_depthpro import (
    DEFAULT_FREEREG_ROOT,
    add_freereg_to_path,
    build_ir_3d_candidates,
    parse_float_list,
    projected_silhouette,
    rank_matches_by_semantic_features,
    score_projected_silhouette,
)


class FreeRegVendorPathTest(unittest.TestCase):
    def test_default_freereg_root_is_vendored_source(self):
        root = add_freereg_to_path(DEFAULT_FREEREG_ROOT)

        self.assertEqual(root, Path("third_party/FreeReg").resolve())
        self.assertTrue((root / "demo.py").exists())
        self.assertIn(str(root), sys.path)
        self.assertIn(str(root / "tools" / "DepthPro" / "src"), sys.path)

    def test_ir_3d_candidates_keep_explicit_value_single_candidate(self):
        candidates = build_ir_3d_candidates(
            auto_ir_3d=0.05,
            explicit_ir_3d=0.2,
            fallback_ir_3d=[0.2, 0.3],
        )

        self.assertEqual(candidates, [{"label": "explicit", "ir_3d": 0.2}])

    def test_ir_3d_candidates_try_auto_then_unique_fallbacks(self):
        candidates = build_ir_3d_candidates(
            auto_ir_3d=0.05,
            explicit_ir_3d=None,
            fallback_ir_3d=parse_float_list("0.05,0.2"),
        )

        self.assertEqual(
            candidates,
            [
                {"label": "auto", "ir_3d": 0.05},
                {"label": "fallback_0.2", "ir_3d": 0.2},
            ],
        )

    def test_rank_matches_by_semantic_features_filters_and_sorts_by_similarity(self):
        features = np.asarray(
            [
                [[[1.0, 0.0], [0.0, 1.0]]],
            ],
            dtype=np.float32,
        )[0]
        image_uv = np.asarray([[1.0, 1.0], [1.0, 1.0], [3.0, 1.0]], dtype=np.float64)
        complete_uv = np.asarray([[3.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

        result = rank_matches_by_semantic_features(
            feature_grid=features,
            image_uv=image_uv,
            complete_uv=complete_uv,
            image_size=(4, 2),
            min_similarity=0.5,
            max_matches=2,
        )

        np.testing.assert_array_equal(result["indices"], np.asarray([1]))
        np.testing.assert_allclose(result["scores"], np.asarray([1.0], dtype=np.float32))

    def test_score_projected_silhouette_prefers_overlap(self):
        intrinsic = np.eye(3, dtype=np.float64)
        image_size = (5, 5)
        target_mask = np.zeros((5, 5), dtype=bool)
        target_mask[2, 2] = True

        overlapping_points = np.asarray([[2.0, 2.0, 1.0]], dtype=np.float64)
        shifted_points = np.asarray([[4.0, 4.0, 1.0]], dtype=np.float64)

        overlap = score_projected_silhouette(
            projected_silhouette(overlapping_points, intrinsic, image_size, dilate_pixels=0),
            target_mask,
        )
        shifted = score_projected_silhouette(
            projected_silhouette(shifted_points, intrinsic, image_size, dilate_pixels=0),
            target_mask,
        )

        self.assertEqual(overlap["iou"], 1.0)
        self.assertEqual(shifted["iou"], 0.0)
        self.assertGreater(overlap["score"], shifted["score"])


if __name__ == "__main__":
    unittest.main()
