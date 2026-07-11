import unittest

import numpy as np

from scripts.run_moge_pixel_index_bridge import (
    build_partial_to_moge_index,
    colors_for_moge_hits,
    filter_moge_points_by_object_mask,
)


class MogePixelIndexBridgeTest(unittest.TestCase):
    def test_build_partial_to_moge_index_matches_nearest_pixels_with_radius(self):
        point_uv = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )
        moge_pixels = np.array(
            [
                [0, 0],
                [4, 4],
                [8, 8],
            ],
            dtype=np.float64,
        )

        result = build_partial_to_moge_index(
            point_uv=point_uv,
            image_size=9,
            moge_pixel_xy=moge_pixels,
            max_pixel_distance=1.5,
        )

        np.testing.assert_array_equal(result.partial_to_moge, np.array([0, 1, 2, -1]))
        np.testing.assert_array_equal(result.matched_partial_indices, np.array([0, 1, 2]))
        np.testing.assert_array_equal(result.matched_moge_indices, np.array([0, 1, 2]))
        self.assertEqual(result.pixel_distances[3], np.inf)

    def test_colors_for_moge_hits_paints_only_indexed_moge_points_red(self):
        colors = colors_for_moge_hits(num_points=4, hit_indices=np.array([1, 3]))

        np.testing.assert_allclose(colors[0], [0.55, 0.55, 0.55])
        np.testing.assert_allclose(colors[1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(colors[2], [0.55, 0.55, 0.55])
        np.testing.assert_allclose(colors[3], [1.0, 0.0, 0.0])

    def test_filter_moge_points_by_object_mask_keeps_only_masked_pixels(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        colors = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
            ],
            dtype=np.float64,
        )
        pixel_xy = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        object_mask = np.array([[0, 255, 0]], dtype=np.uint8)

        filtered = filter_moge_points_by_object_mask(
            points=points,
            colors=colors,
            pixel_xy=pixel_xy,
            object_mask=object_mask,
            alpha_threshold=128,
        )

        np.testing.assert_allclose(filtered.points, points[[1]])
        np.testing.assert_allclose(filtered.colors, colors[[1]])
        np.testing.assert_allclose(filtered.pixel_xy, pixel_xy[[1]])
        np.testing.assert_array_equal(filtered.original_indices, np.array([1]))


if __name__ == "__main__":
    unittest.main()
