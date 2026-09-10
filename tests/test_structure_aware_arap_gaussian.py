import numpy as np

from src.structure_aware_arap_gaussian import (
    moge_unsupported_partial_mask,
    structure_aware_arap_gaussian_adaptation,
)


def test_structure_aware_arap_moves_a_coherent_residual_component_and_keeps_stable_support():
    x, y, z = np.meshgrid(np.arange(18), np.arange(12), np.arange(3), indexing="xy")
    prior = np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)] / 18.
    partial = prior.copy()
    appendage = prior[:, 0] > .72
    partial[appendage, 0] += .06
    edited, info, masks = structure_aware_arap_gaussian_adaptation(
        prior, partial, normal_neighbours=12, normal_agreement=.35,
        maximum_correspondence_ratio=.30, residual_ratio=.035, stable_ratio=.012,
        component_radius_ratio=.08, minimum_component_points=18,
        minimum_directional_coherence=.60, influence_radius_ratios=(.18, .28, .38),
        maximum_influence_fraction=.85, maximum_edge_stretch=4., minimum_edge_compression=.30,
    )
    assert info["active"]
    assert info["anchor_residual_after_median"] < 1e-10
    assert masks["anchors"].sum() >= 18
    assert masks["stable"].any()
    assert np.allclose(edited[masks["stable"]], prior[masks["stable"]])


def test_structure_aware_arap_inherits_previous_locks_without_reediting_them():
    x, y, z = np.meshgrid(np.arange(18), np.arange(12), np.arange(3), indexing="xy")
    prior = np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)] / 18.
    partial = prior.copy()
    appendage = prior[:, 0] > .72
    partial[appendage, 0] += .06
    inherited = appendage & ((np.arange(len(prior)) % 5) == 0)
    edited, info, masks = structure_aware_arap_gaussian_adaptation(
        prior, partial, locked_prior_mask=inherited,
        normal_neighbours=12, normal_agreement=.35,
        maximum_correspondence_ratio=.30, residual_ratio=.035, stable_ratio=.012,
        component_radius_ratio=.08, minimum_component_points=18,
        minimum_directional_coherence=.60, influence_radius_ratios=(.18, .28, .38),
        maximum_influence_fraction=.85, maximum_edge_stretch=4., minimum_edge_compression=.30,
    )
    assert info["active"]
    assert info["inherited_locked_gaussians"] == int(inherited.sum())
    assert not masks["anchors"][inherited].any()
    assert np.all(masks["stable"][inherited])
    assert np.allclose(edited[inherited], prior[inherited])


def test_moge_unsupported_support_allows_only_a_relaxed_residual_route():
    x, y, z = np.meshgrid(np.arange(18), np.arange(12), np.arange(3), indexing="xy")
    prior = np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)] / 18.
    partial = prior.copy()
    appendage = prior[:, 0] > .72
    partial[appendage, 0] += .06
    moge = prior.copy()
    moge[appendage, 0] -= .20
    unsupported, support = moge_unsupported_partial_mask(
        partial, moge, normal_neighbours=12, maximum_distance_ratio=.04,
        minimum_normal_agreement=.30,
    )
    assert support["unsupported_partial"] > 0
    assert unsupported[appendage].mean() > .70
    edited, info, masks = structure_aware_arap_gaussian_adaptation(
        prior, partial, bridge_unknown_partial_mask=unsupported,
        normal_neighbours=12, normal_agreement=.35, bridge_unknown_normal_agreement=.25,
        maximum_correspondence_ratio=.30, residual_ratio=.035, stable_ratio=.012,
        component_radius_ratio=.08, minimum_component_points=18,
        minimum_directional_coherence=.60, influence_radius_ratios=(.18, .28, .38),
        maximum_influence_fraction=.85, maximum_edge_stretch=4., minimum_edge_compression=.30,
    )
    assert info["active"]
    assert info["bridge_unknown_partial"] == int(unsupported.sum())
    assert info["bridge_unknown_residual_partial"] > 0
    assert masks["anchors"].any()
    assert np.linalg.norm(edited - prior, axis=1).max() > 0.
