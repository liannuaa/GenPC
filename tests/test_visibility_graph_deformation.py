import numpy as np

from src.visibility_graph_deformation import solve_visibility_deformation_graph


def test_visibility_graph_warp_moves_matched_support_but_is_bounded():
    x, y = np.meshgrid(np.linspace(-1., 1., 10), np.linspace(-1., 1., 10))
    prior = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    partial = prior + np.array([.05, 0., 0.])
    moved, info = solve_visibility_deformation_graph(
        prior, partial, np.arange(len(prior)), np.arange(len(prior)), np.ones(len(prior)),
        nodes=32, max_displacement_ratio=.1, seed=3)
    assert np.mean(moved[:, 0] - prior[:, 0]) > .02
    assert info["max_node_displacement"] <= info["max_displacement"] + 1e-10
