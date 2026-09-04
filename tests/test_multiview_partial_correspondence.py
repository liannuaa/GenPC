import numpy as np

from src.multiview_partial_correspondence import (
    concatenate_positive_pairs,
    virtual_view_positive_pairs,
)


def test_virtual_views_provide_positive_overlap_without_empty_pixel_penalty():
    prior = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                      (0., 0., 1.), (1., 1., .2), (.2, 1., 1.)))
    partial = prior[[0, 1, 3, 4]] + np.array((.01, 0., 0.))
    virtual, info = virtual_view_positive_pairs(
        partial, prior, views=6, resolution=64, max_pixel_distance=2.,
    )
    assert len(virtual) > 0
    assert np.all((virtual[:, 0] >= 0) & (virtual[:, 0] < len(partial)))
    assert np.all((virtual[:, 1] >= 0) & (virtual[:, 1] < len(prior)))
    assert info["evidence_policy"] == "positive_overlap_only; missing_partial_pixels_are_unconstrained"
    joined = concatenate_positive_pairs(np.array(((0, 0, 0, 0.),)), virtual)
    assert len(joined) == len(virtual) + 1
