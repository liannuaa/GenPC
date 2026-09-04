import numpy as np

from src.indexed_pixel_sim3 import robust_indexed_sim3
from src.ray_consistent_registration import apply_transform


def test_indexed_pixel_sim3_recovers_category_free_similarity():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(80, 3))
    transform = np.eye(4)
    transform[:3, :3] *= 1.7
    transform[:3, 3] = (.2, -.1, .3)
    target = apply_transform(source, transform)
    matches = np.c_[np.arange(80), np.arange(80), np.arange(80), np.zeros(80)]
    estimated, info = robust_indexed_sim3(source, target, matches, diagonal=5., trials=64, seed=3)
    assert info["inlier_ratio"] > .95
    assert np.allclose(apply_transform(source, estimated), target, atol=1e-6)
