import numpy as np

from src.observation_anchored_fusion import collision_free_anchor_pairs


def test_collision_free_fusion_replaces_each_carrier_slot_at_most_once():
    carrier = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    partial = np.array(((0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (2.1, 0.0, 0.0)))
    pairs = np.array(((0, 0, 0, 0.0), (1, 0, 0, 0.1), (2, 2, 2, 0.0)))
    selected, info = collision_free_anchor_pairs(
        pairs, partial, carrier, max_residual=0.3,
    )
    assert np.array_equal(selected, np.array(((0, 0), (2, 2))))
    assert info["collision_free_pairs"] == 2


def test_collision_free_fusion_rejects_large_geometric_residuals():
    carrier = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    partial = np.array(((5.0, 0.0, 0.0),))
    pairs = np.array(((0, 0, 0, 0.0),))
    selected, info = collision_free_anchor_pairs(
        pairs, partial, carrier, max_residual=0.3,
    )
    assert selected.shape == (0, 2)
    assert info["within_metric_limit"] == 0
