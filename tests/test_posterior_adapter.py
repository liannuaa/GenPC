"""Generic invariants for complete-prior posterior adaptation."""

from __future__ import annotations

import numpy as np

from src.posterior_adapter import (
    PosteriorAdapter,
    PosteriorAdapterConfig,
    _embedded_arap_posterior,
    _project_small_topology_breaks,
    partial_optimal_transport,
)


class OrthographicProjector:
    image_shape = (192, 192)

    def __init__(self, points: np.ndarray):
        low, high = points[:, :2].min(0), points[:, :2].max(0)
        self.low = low
        self.extent = np.maximum(high - low, 1e-6)

    def project(self, points: np.ndarray):
        points = np.asarray(points)
        uv = (points[:, :2] - self.low) / self.extent * 160. + 16.
        return uv, points[:, 2] - points[:, 2].min() + 1.


def _connected_shape() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.linspace(0., 1., 40), np.linspace(-.22, .22, 18), indexing="ij")
    z = .04 * np.sin(np.pi * x) * np.cos(np.pi * y / .44)
    body = np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)]
    t = np.linspace(0., 1., 12)
    appendage = np.c_[.80 + .08 * t, -.18 * np.ones_like(t), -.12 - .45 * t]
    appendage = np.repeat(appendage, 6, axis=0)
    jitter = np.tile(np.c_[np.zeros(6), np.linspace(-.025, .025, 6), np.zeros(6)], (12, 1))
    prior = np.concatenate((body, appendage + jitter), axis=0)
    partial = prior.copy()
    moved = np.arange(len(body), len(prior))
    ramp = np.linspace(.08, .25, len(moved))
    partial[moved, 0] += ramp
    partial[moved, 1] -= .12
    partial[moved, 2] -= .08
    return prior, partial, moved


def _test_config() -> PosteriorAdapterConfig:
    return PosteriorAdapterConfig(
        partial_samples=1_024, prior_samples=1_536, normal_neighbours=12,
        transport_iterations=25, minimum_transport_pairs=24,
        maximum_3d_ratio=.40, maximum_screen_ratio=.40, maximum_depth_ratio=.40,
        stable_spacing_multiplier=2., coarse_neighbours=12, coarse_edge_ratio=2.8,
        coarse_iterations=3, fine_neighbours=8, fine_edge_ratio=2.2, fine_iterations=3,
        attachment_edge_ratio=4., minimum_edge_compression=.25,
        maximum_edge_stretch=4., minimum_hidden_coverage=.80, coverage_resolution=64,
    )


def test_partial_transport_can_leave_an_unrelated_observation_unmatched():
    prior, partial, _ = _connected_shape()
    outliers = np.array([[4., 4., 4.], [4.1, 4., 4.]], dtype=np.float64)
    observed = np.concatenate((partial, outliers), axis=0)
    projector = OrthographicProjector(observed)
    result = partial_optimal_transport(prior, observed, projector, config=_test_config(), device="cpu")
    assert not np.isin(np.arange(len(partial), len(observed)), result.partial_ids).any()
    assert result.diagnostics["unmatched_partial_fraction"] > 0.


def test_embedded_posterior_preserves_slots_and_continuously_moves_supported_structure():
    prior, partial, moved_ids = _connected_shape()
    stable = np.zeros(len(prior), dtype=bool)
    stable[np.arange(len(prior) - len(moved_ids))[prior[:-len(moved_ids), 0] < .65]] = True
    posterior, info, moved = _embedded_arap_posterior(
        prior, moved_ids, partial[moved_ids], np.ones(len(moved_ids)), stable,
        node_fraction=.12, minimum_nodes=96, maximum_nodes=512, skinning_neighbours=4,
        neighbours=10, edge_ratio=3.0, attachment_edge_ratio=4.0,
        data_weight=4.0, screening=.001, iterations=4,
        maximum_displacement=.5, minimum_improvement=.01,
        minimum_edge_compression=.2, maximum_edge_stretch=4.0,
        minimum_hidden_coverage=.8, coverage_resolution=64,
    )
    assert info["active"]
    assert len(posterior) == len(prior)
    assert moved.any()
    assert np.allclose(posterior[stable], prior[stable])
    before = np.median(np.linalg.norm(prior[moved_ids] - partial[moved_ids], axis=1))
    after = np.median(np.linalg.norm(posterior[moved_ids] - partial[moved_ids], axis=1))
    assert after < before
    neighbour_step = np.linalg.norm(np.diff(posterior[moved_ids], axis=0), axis=1)
    assert np.quantile(neighbour_step, .99) < .12


def test_topology_projection_reconnects_a_new_small_fragment_without_moving_stable_points():
    x = np.linspace(0., 1., 180)
    prior = np.c_[x, .01 * np.sin(8. * x), np.zeros_like(x)]
    candidate = prior.copy()
    candidate[-3:] += np.array([0., .8, 0.])
    stable = np.zeros(len(prior), dtype=bool)
    stable[:20] = True
    repaired, info = _project_small_topology_breaks(
        prior, candidate, stable, neighbours=8, edge_ratio=3.0,
        component_threshold=16, maximum_passes=3,
    )
    assert info["active"]
    assert np.allclose(repaired[stable], prior[stable])
    assert np.linalg.norm(repaired[-1] - repaired[-4]) < np.linalg.norm(candidate[-1] - candidate[-4])
