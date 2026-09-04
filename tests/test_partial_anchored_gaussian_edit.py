import numpy as np

from src.partial_anchored_gaussian_edit import (
    boundary_conditioned_graph_displacement,
    compact_anchor_displacement,
)


def test_compact_gaussian_edit_preserves_remote_complete_prior_support():
    prior = np.array(tuple((.1 * index, 0., 0.) for index in range(6)) + ((2., 0., 0.),))
    partial = prior[:6] + np.array((0., .2, 0.))
    pairs = np.array(tuple((index, index, index, 0.) for index in range(6)))
    edited, anchors, info = compact_anchor_displacement(
        prior, partial, pairs, max_anchor_residual=.3, support_radius=.25,
        max_displacement=.3, neighbours=1,
    )
    assert len(anchors) == 6
    assert edited[1, 1] > .19
    assert np.allclose(edited[6], prior[6])
    assert info["edited_prior_gaussians"] == 6
    assert info["untouched_prior_gaussians"] == 1


def test_boundary_graph_keeps_controls_exact_and_disconnected_body_fixed():
    # Two locally sampled surfaces with a large structural gap.  Only the first
    # one is observed, so the screen must leave the disconnected complete-body
    # component untouched while harmonically moving its observed component.
    observed = np.array(tuple((.1 * index, 0., 0.) for index in range(8)))
    hidden = np.array(tuple((5. + .1 * index, 0., 0.) for index in range(8)))
    prior = np.concatenate((observed, hidden), axis=0)
    partial = observed + np.array((0., .15, 0.))
    pairs = np.array(tuple((index, index, index, 0.) for index in range(8)))
    edited, anchors, info = boundary_conditioned_graph_displacement(
        prior, partial, pairs, max_anchor_residual=.25, max_displacement=.25,
        neighbours=3, edge_ratio=1.8, screening=.01,
    )
    assert len(anchors) == 8
    assert np.allclose(edited[anchors[:, 1]], partial[anchors[:, 0]])
    assert np.allclose(edited[8:], prior[8:])
    assert info["control_components"] == 1
    assert info["prior_protection_gaussians"] >= 0
    assert info["field"] == "boundary_conditioned_surface_graph_with_multiview_prior_protection"
