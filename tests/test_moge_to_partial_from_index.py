import unittest

import numpy as np

from scripts.run_moge_to_partial_from_index import (
    apply_transform,
    estimate_similarity_transform,
)
from scripts.run_moge_to_raw_partial_from_camera import camera_xy_to_uv
from scripts.run_moge_to_raw_partial_from_camera import depth_image_uv_from_projection_uv


class MogeToPartialFromIndexTest(unittest.TestCase):
    def test_estimate_similarity_transform_recovers_scaled_translation(self):
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        target = source * 2.5 + np.array([1.0, -2.0, 0.5], dtype=np.float64)

        transform = estimate_similarity_transform(source, target)
        aligned = apply_transform(source, transform)

        np.testing.assert_allclose(aligned, target, atol=1e-8)
        np.testing.assert_allclose(transform[:3, :3], np.eye(3) * 2.5, atol=1e-8)
        np.testing.assert_allclose(transform[:3, 3], [1.0, -2.0, 0.5], atol=1e-8)

    def test_camera_xy_to_uv_rescales_with_padding(self):
        camera_xy = np.array(
            [
                [-2.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float64,
        )

        uv = camera_xy_to_uv(camera_xy, padding=0.1)

        np.testing.assert_allclose(uv[:, 0], [0.1, 0.5, 0.9])
        np.testing.assert_allclose(uv[:, 1], [0.5, 0.5, 0.5])

    def test_depth_image_uv_from_projection_uv_compensates_paintpixels_vertical_flip(self):
        projection_uv = np.array(
            [
                [0.25, 0.1],
                [0.75, 0.9],
            ],
            dtype=np.float64,
        )

        image_uv = depth_image_uv_from_projection_uv(projection_uv)

        np.testing.assert_allclose(image_uv, [[0.25, 0.9], [0.75, 0.1]])


if __name__ == "__main__":
    unittest.main()
