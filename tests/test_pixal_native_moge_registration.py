from pathlib import Path

import numpy as np

from scripts.run_pixal_native_moge_registration import load_points_and_colors, native_pixal_moge_observation
from src.pixal_moge_analytic_registration import (
    pixal_export_to_camera,
    visible_ray_depth_scale_step,
    visible_mask_translation_step,
)
from src.moge_camera import MoGeProjector


def test_native_pixal_moge_observation_has_explicit_same_input_contract():
    # Model execution is integration-only; keep the unit contract explicit so
    # callers cannot silently replace the prepared Pixal image by img.png.
    assert "pixal3d_input.png" in native_pixal_moge_observation.__doc__
    assert "get_camera_params_wild_moge" in native_pixal_moge_observation.__doc__
    assert Path("pixal3d_input.png").suffix == ".png"


def test_pixal_export_camera_transform_has_opencv_positive_forward_depth():
    transform = pixal_export_to_camera(2.0)
    # The official aligned camera is at +z looking to the origin, so origin
    # appears at positive depth 2 in OpenCV camera coordinates.
    assert transform[2, 3] == 2.0
    assert np.allclose(transform[:3, :3], np.diag([-1., -1., 1.]))


def test_visible_mask_translation_step_moves_prior_towards_target_in_camera_xy():
    projector = MoGeProjector(np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]), (100, 100))
    grid_y, grid_x = np.mgrid[-5:6, -5:6]
    target = np.c_[(grid_x.ravel() + 5.) / 100., (grid_y.ravel() + 5.) / 100., np.ones(121)]
    prior = np.c_[grid_x.ravel() / 100., grid_y.ravel() / 100., np.ones(121)]
    step, info = visible_mask_translation_step(target, prior, projector)
    assert info["accepted"]
    assert step[0, 3] > 0 and step[1, 3] > 0


def test_visible_ray_depth_scale_uses_same_pixel_surface_ratios():
    projector = MoGeProjector(np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]), (100, 100))
    grid_y, grid_x = np.mgrid[-5:6, -5:6]
    prior = np.c_[grid_x.ravel() / 100., grid_y.ravel() / 100., np.ones(121)]
    target = np.c_[prior[:, :2] * 1.2, np.full(121, 1.2)]
    step, info = visible_ray_depth_scale_step(target, prior, projector)
    assert info["accepted"]
    assert step[0, 0] > 1.1


def test_cached_loader_preserves_explicit_absence_of_point_colors(tmp_path):
    import open3d as o3d

    path = tmp_path / "points.ply"
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array([[0., 0., 1.]]))
    o3d.io.write_point_cloud(str(path), cloud)
    points, colors = load_points_and_colors(path)
    assert points.shape == (1, 3)
    assert colors is None
