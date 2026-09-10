import numpy as np

from src.complete_prior_bridge import _proper_signed_permutations, _trimmed_mean


def test_proper_signed_permutations_are_the_24_rotational_axis_maps():
    candidates = _proper_signed_permutations()
    assert len(candidates) == 24
    for matrix in candidates:
        assert np.allclose(matrix.T @ matrix, np.eye(3))
        assert np.linalg.det(matrix) > 0.0


def test_trimmed_mean_ignores_large_outlier_tail():
    values = np.concatenate([np.ones(80), np.full(20, 100.0)])
    assert _trimmed_mean(values, 0.8) == 1.0
