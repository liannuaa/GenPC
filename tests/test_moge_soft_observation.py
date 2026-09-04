import numpy as np

from src.moge_soft_observation import (confidence_gated_moge_extension, moge_visible_soft_completion,
                                       robust_moge_to_partial)
from src.ray_consistent_registration import apply_transform


def test_moge_alignment_and_supported_extension_are_category_free():
    rng = np.random.default_rng(7)
    moge = rng.normal(size=(80, 3))
    transform = np.eye(4); transform[:3, :3] = 1.7 * np.eye(3); transform[:3, 3] = (.2, -.1, .3)
    partial = apply_transform(moge, transform)
    pairs = np.c_[np.arange(80), np.arange(80), np.arange(80), np.zeros(80)]
    estimated, info = robust_moge_to_partial(moge, partial, pairs, diagonal=5., trials=64, seed=3)
    assert info["inlier_ratio"] > .95
    assert np.allclose(apply_transform(moge, estimated), partial, atol=1e-6)
    pixel = np.c_[np.arange(80) % 16 + 100, np.arange(80) // 16 + 120]
    extension, confidence, details = confidence_gated_moge_extension(
        partial, pixel, pairs, np.zeros(80), image_size=512, grid_size=8,
        min_cell_matches=1, residual_ratio=.1, diagonal=5.)
    assert len(extension) == 80
    assert details["supported_cells"] > 0
    assert np.all(confidence > 0.)
    visible, visible_confidence, visible_info = moge_visible_soft_completion(
        partial, pixel, pairs, np.zeros(80), image_size=512, grid_size=8,
        min_cell_matches=1, residual_ratio=.1, diagonal=5.)
    assert len(visible) == len(moge)
    assert visible_info["reliable_points"] == len(moge)
    assert np.all(visible_confidence > 0.)
