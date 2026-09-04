import numpy as np

from src.ray_consistent_registration import apply_transform
from src.bidirectional_cycle_registration import visible_score
from src.visible_pixel_sim3_refinement import (
    local_camera1_visible_refine,
    pixel_pair_residual_candidates,
    visible_pixel_pairs,
)


class _OrthographicProjector:
    image_shape = (96, 96)

    def project(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.c_[points[:, 0] * 24. + 48., points[:, 1] * 24. + 48.], points[:, 2]


def test_visible_pixel_pairs_recover_same_saved_view_surface_ids():
    y, x = np.mgrid[-3:4, -3:4]
    partial = np.c_[x.ravel() / 10., y.ravel() / 10., 1. + .01 * x.ravel() * y.ravel()]
    prior = partial.copy()
    pairs, info = visible_pixel_pairs(partial, prior, _OrthographicProjector(), max_pixel_distance=0.)
    assert info["mutual_pixel_pairs"] == len(partial)
    assert np.array_equal(pairs[:, 0].astype(int), pairs[:, 1].astype(int))


def test_pixel_pair_sim3_candidates_reduce_a_pure_visible_depth_offset():
    y, x = np.mgrid[-4:5, -4:5]
    partial = np.c_[x.ravel() / 12., y.ravel() / 12., 1. + .01 * x.ravel() * y.ravel()]
    prior = partial.copy()
    prior[:, 2] -= .06
    diagonal = float(np.linalg.norm(np.ptp(partial, axis=0)))
    candidates, info = pixel_pair_residual_candidates(
        partial, prior, _OrthographicProjector(), diagonal=diagonal,
        max_pixel_distance=0., trials=32, fractions=(1.0,),
        max_rotation_deg=5., scale_bounds=(.90, 1.10), max_translation_ratio=.20,
    )
    transform = dict(candidates)["pixel_pair_fraction_1.000"]
    before = np.linalg.norm(prior - partial, axis=1).mean()
    after = np.linalg.norm(apply_transform(prior, transform) - partial, axis=1).mean()
    assert info["robust_fit"]["inlier_ratio"] > .95
    assert after < before * .05


def test_local_camera1_refine_recovers_a_small_visible_translation():
    y, x = np.mgrid[-5:6, -5:6]
    partial = np.c_[x.ravel() / 12., y.ravel() / 12., 1. + .01 * x.ravel() * y.ravel()]
    prior = partial.copy()
    prior[:, 0] += .025
    diagonal = float(np.linalg.norm(np.ptp(partial, axis=0)))
    projector = _OrthographicProjector()
    before = visible_score(partial, prior, projector, diagonal, pixel_radius=5.)
    step, info = local_camera1_visible_refine(
        partial, prior, projector, diagonal=diagonal, search_points=1000,
        levels=((.001, .05, .03),),
    )
    after = visible_score(partial, apply_transform(prior, step), projector, diagonal, pixel_radius=5.)
    assert info["trace"][0]["action"] == "trans_x_-1"
    assert after["objective"] < before["objective"]


def test_local_camera1_parallel_candidate_scores_match_serial_selection():
    y, x = np.mgrid[-5:6, -5:6]
    partial = np.c_[x.ravel() / 12., y.ravel() / 12., 1. + .01 * x.ravel() * y.ravel()]
    prior = partial.copy()
    prior[:, 0] += .025
    diagonal = float(np.linalg.norm(np.ptp(partial, axis=0)))
    kwargs = {
        "diagonal": diagonal, "search_points": 1000,
        "levels": ((.001, .05, .03), (.0005, .025, .01)),
    }
    serial_step, serial = local_camera1_visible_refine(
        partial, prior, _OrthographicProjector(), candidate_workers=1, **kwargs,
    )
    parallel_step, parallel = local_camera1_visible_refine(
        partial, prior, _OrthographicProjector(), candidate_workers=4, **kwargs,
    )
    assert np.array_equal(serial_step, parallel_step)
    assert serial["trace"] == parallel["trace"]
