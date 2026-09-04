import numpy as np

from src.ray_conditioned_gaussian_edit import _weighted_median, interpolate_ray_edit


def test_interpolate_ray_edit_is_bounded_and_preserves_unedited_rows():
    prior = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    corrected = prior.copy(); corrected[1, 2] += .4
    half = interpolate_ray_edit(prior, corrected, .5)
    assert np.allclose(half[0], prior[0])
    assert np.allclose(half[2], prior[2])
    assert np.isclose(half[1, 2], 1.2)


def test_weighted_median_keeps_hard_observation_dominant_over_soft_samples():
    assert np.isclose(_weighted_median(np.array([0., 1., 2.]), np.array([1., .1, .1])), 0.)
