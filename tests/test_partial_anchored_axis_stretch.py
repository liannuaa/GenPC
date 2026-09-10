"""Unit tests for partial-supported anisotropic Gaussian shape drives."""

from __future__ import annotations

import numpy as np

from src.partial_anchored_gaussian_edit import (
    visible_axis_stretch, visible_depth_extent_calibration, visible_principal_extent_calibration,
)


def _pairs(count: int) -> np.ndarray:
    ids = np.arange(count, dtype=np.float64)
    return np.c_[ids, ids, ids, np.zeros(count, dtype=np.float64)]


def test_visible_axis_stretch_recovers_relative_observed_scale():
    rng = np.random.default_rng(4)
    prior = rng.normal(size=(2048, 3)) * np.array([1.2, .7, .4])
    partial = prior * np.array([1.18, .92, .92])

    drive, info = visible_axis_stretch(
        prior, partial, _pairs(len(prior)), max_anchor_residual=1., minimum_pairs=128,
    )

    assert info["active"]
    assert float(info["anisotropy_log"]) > .05
    assert float(info["residual_reduction"]) > .10
    assert float(np.linalg.norm(drive, axis=1).mean()) > .01


def test_visible_axis_stretch_rejects_isotropic_scale_as_sim3_gauge():
    rng = np.random.default_rng(7)
    prior = rng.normal(size=(1024, 3))
    partial = prior * 1.15

    drive, info = visible_axis_stretch(
        prior, partial, _pairs(len(prior)), max_anchor_residual=1., minimum_pairs=128,
    )

    assert not info["active"]
    assert info["reason"] == "no_supported_anisotropic_residual"
    assert np.allclose(drive, 0.)


def test_visible_axis_stretch_can_retain_supported_uniform_component():
    rng = np.random.default_rng(11)
    prior = rng.normal(size=(2048, 3)) * np.array([1.1, .8, .5])
    partial = prior * np.array([1.02, 1.02, 1.10])

    drive, info = visible_axis_stretch(
        prior, partial, _pairs(len(prior)), max_anchor_residual=1., minimum_pairs=128,
        minimum_anisotropy=.01, minimum_residual_reduction=.001,
        retain_isotropic_component=True,
    )

    assert info["active"]
    assert info["retain_isotropic_component"]
    assert np.allclose(info["applied_scale"], info["visible_scale"])
    assert float(np.linalg.norm(drive, axis=1).mean()) > .01


def test_camera_axis_stretch_keeps_unobserved_depth_axis_fixed():
    rng = np.random.default_rng(12)
    prior = rng.normal(size=(2048, 3)) * np.array([1.2, .8, .015])
    # A near-planar Camera-1 observation constrains image horizontal/vertical
    # extent, but its tiny depth spread must not trigger a depth-scale edit.
    partial = prior * np.array([1.13, .91, 1.8])
    drive, info = visible_axis_stretch(
        prior, partial, _pairs(len(prior)), max_anchor_residual=1., minimum_pairs=128,
        axis_basis=np.eye(3), minimum_axis_spread_ratio=.08,
        minimum_anisotropy=.025, minimum_residual_reduction=.03,
    )
    assert info["active"]
    assert info["axis_frame"] == "saved_camera"
    assert info["axis_supported"] == [True, True, False]
    assert np.isclose(info["relative_scale"][2], 1.)


class _OrthographicProjector:
    image_shape = (128, 128)

    def project(self, points):
        points = np.asarray(points, dtype=np.float64)
        uv = np.c_[12. + 80. * points[:, 0], 12. + 80. * points[:, 1]]
        return uv, points[:, 2]


def test_visible_depth_extent_calibration_expands_entire_carrier():
    axis = np.linspace(0., 1., 64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    prior = np.c_[x.reshape(-1), y.reshape(-1), .1 + .5 * x.reshape(-1)]
    partial = prior.copy()
    partial[:, 2] = 1.2 * prior[:, 2] - .05

    drive, info = visible_depth_extent_calibration(
        prior, partial, _OrthographicProjector(), axis_basis=np.eye(3),
        quantile=.01, minimum_visible_points=512,
        minimum_log_expansion=.01, max_log_expansion=.25,
    )

    assert info["active"]
    assert np.isclose(info["applied_depth_scale"], 1.2, atol=1e-4)
    calibrated = prior + drive
    assert np.allclose(calibrated[:, :2], prior[:, :2])
    assert np.allclose(calibrated[:, 2], partial[:, 2], atol=1e-4)


def test_visible_principal_extent_calibration_expands_about_aligned_endpoint():
    x = np.linspace(0., 1., 64)
    y = np.linspace(.3, .7, 64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    prior = np.c_[xx.reshape(-1), yy.reshape(-1), .2 * yy.reshape(-1)]
    partial = prior.copy()
    partial[:, 0] = 1.2 * prior[:, 0]

    drive, info = visible_principal_extent_calibration(
        prior, partial, _OrthographicProjector(), quantile=.01,
        minimum_visible_points=512, minimum_axis_anisotropy=1.15,
        anchor_tolerance=.03, minimum_log_expansion=.01, max_log_expansion=.25,
    )

    assert info["active"]
    assert info["anchored_endpoint"] == "low"
    assert np.isclose(info["applied_axis_scale"], 1.2, atol=1e-4)
    assert np.allclose(prior + drive, partial, atol=1e-4)
