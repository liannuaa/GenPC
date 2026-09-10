import numpy as np

from src.multiview_proxy_deformation import (
    OrbitProjector,
    _interpolate_field,
    _pixel_distance_schedule,
    _visible_pairs,
)


def test_orbit_projector_centres_optical_axis():
    pose = np.eye(4)
    projector = OrbitProjector(pose, 60.0, (101, 101))
    uv, depth = projector.project(np.array([[0.0, 0.0, -2.0]]))
    assert np.allclose(uv[0], [50.0, 50.0])
    assert np.allclose(depth, [2.0])


def test_interpolation_preserves_constant_shared_field():
    anchors = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    field = np.tile(np.array([[.2, -.1, .3]]), (len(anchors), 1))
    query = np.array([[.25, .25, 0.], [.7, .6, 0.]])
    assert np.allclose(_interpolate_field(query, anchors, field), field[:2])


def test_visible_pairs_returns_identity_surface_residual():
    projector = OrbitProjector(np.eye(4), 60.0, (101, 101))
    points = np.array([[0.0, 0.0, -2.0]])
    pairs = _visible_pairs(points, points, projector, max_pixel_distance=2.0)
    assert len(pairs["prior_ids"]) >= 1
    assert np.allclose(pairs["residual"], 0.0)


def test_broad_to_narrow_schedule_is_monotonic_and_ends_before_fine_stage():
    schedule = _pixel_distance_schedule(5.0, 64.0, 3)
    assert len(schedule) == 3
    assert schedule[0] == 64.0
    assert schedule[0] > schedule[1] > schedule[2] > 5.0


def test_broad_to_narrow_schedule_can_be_disabled_without_changing_baseline():
    assert _pixel_distance_schedule(5.0, 64.0, 0) == ()
