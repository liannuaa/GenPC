from __future__ import annotations

import numpy as np

from src.residual_component_registration import coherent_residual_component_translation


class _OrthographicProjector:
    image_shape = (160, 160)

    def project(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.c_[10. + 120. * points[:, 0], 10. + 120. * points[:, 1]], points[:, 2]


def _two_parts():
    values = np.linspace(0., 1., 24)
    x, y = np.meshgrid(values, values, indexing="ij")
    body = np.c_[.10 + .20 * x.reshape(-1), .15 + .65 * y.reshape(-1), np.ones(x.size)]
    rear = np.c_[.68 + .22 * x.reshape(-1), .15 + .65 * y.reshape(-1), np.ones(x.size)]
    prior = np.r_[body, rear]
    partial = prior.copy()
    partial[len(body):, 2] += .20
    return prior, partial, len(body)


def test_component_registration_moves_only_coherent_residual_surface():
    prior, partial, split = _two_parts()
    displacement, info = coherent_residual_component_translation(
        prior, partial, _OrthographicProjector(), camera_axes=np.eye(3),
        minimum_residual_ratio=.10, minimum_component_pairs=128,
        screen_component_radius=4., stable_residual_ratio=.04,
        maximum_component_fraction=.60, maximum_influence_fraction=.60,
    )
    assert info["active"]
    candidate = prior + displacement
    assert np.median(candidate[split:, 2]) > .19
    assert np.allclose(candidate[:split], prior[:split])
    assert info["influence_prior_points"] <= len(prior)
    assert info["outer_fixed_prior_points"] > 0
    assert info["carrier_slots_preserved"]


def test_component_registration_rejects_incoherent_residuals():
    prior, partial, _ = _two_parts()
    partial[:, 2] = np.sin(np.arange(len(partial))) * .20
    displacement, info = coherent_residual_component_translation(
        prior, partial, _OrthographicProjector(), camera_axes=np.eye(3),
        minimum_residual_ratio=.10, minimum_component_pairs=64,
    )
    assert not info["active"]
    assert np.allclose(displacement, 0.)
