import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_mogev2_registration_probe as probe


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def test_foreground_mask_removes_invalid_and_white_pixels():
    points = np.ones((2, 2, 3), dtype=np.float32)
    mask = np.array([[True, True], [False, True]])
    image = np.array(
        [
            [[255, 255, 255], [240, 100, 100]],
            [[10, 10, 10], [249, 249, 249]],
        ],
        dtype=np.uint8,
    )

    valid = probe.build_foreground_mask(points, mask, image, white_threshold=248)

    assert valid.tolist() == [[False, True], [False, False]]


def test_image_foreground_mask_uses_alpha_before_white_threshold():
    image = np.array(
        [
            [[10, 10, 10, 0], [20, 20, 20, 255]],
            [[255, 255, 255, 255], [30, 40, 50, 4]],
        ],
        dtype=np.uint8,
    )

    foreground = probe.image_foreground_mask(
        image, white_threshold=248, alpha_threshold=8
    )

    assert foreground.tolist() == [[False, True], [False, False]]


def test_crop_to_foreground_bbox_applies_padding_and_returns_mask_crop():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    foreground = np.zeros((6, 8), dtype=bool)
    foreground[2:4, 3:5] = True

    cropped_image, cropped_mask, box = probe.crop_to_foreground(
        image, foreground, padding=1
    )

    assert box == (2, 1, 6, 5)
    assert cropped_image.shape == (4, 4, 3)
    assert cropped_mask.shape == (4, 4)
    assert int(cropped_mask.sum()) == 4


def test_initial_scale_transform_matches_bbox_diagonal_ratio():
    source = make_pcd([[0, 0, 0], [1, 0, 0]])
    target = make_pcd([[2, 0, 0], [4, 0, 0]])

    transform, scale = probe.initial_similarity_transform(source, target)

    assert np.isclose(scale, 2.0)
    np.testing.assert_allclose(transform[:3, :3], np.eye(3) * 2.0)
    np.testing.assert_allclose(transform[:3, 3], np.array([2.0, 0.0, 0.0]))


def test_make_visualization_cloud_uses_expected_color_counts():
    partial = make_pcd([[0, 0, 0], [1, 0, 0]])
    moge = make_pcd([[0, 1, 0]])
    omni = make_pcd([[0, 0, 1], [1, 1, 1], [2, 2, 2]])

    vis = probe.make_visualization_cloud(partial, moge, omni)
    colors = np.asarray(vis.colors)

    red = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
    blue = (colors[:, 2] > 0.9) & (colors[:, 0] < 0.1) & (colors[:, 1] < 0.3)
    gray = np.isclose(colors[:, 0], 0.68) & np.isclose(colors[:, 1], 0.68)

    assert len(vis.points) == 6
    assert int(red.sum()) == 2
    assert int(blue.sum()) == 1
    assert int(gray.sum()) == 3


if __name__ == "__main__":
    test_foreground_mask_removes_invalid_and_white_pixels()
    test_image_foreground_mask_uses_alpha_before_white_threshold()
    test_crop_to_foreground_bbox_applies_padding_and_returns_mask_crop()
    test_initial_scale_transform_matches_bbox_diagonal_ratio()
    test_make_visualization_cloud_uses_expected_color_counts()
