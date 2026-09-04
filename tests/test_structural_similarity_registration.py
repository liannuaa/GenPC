import numpy as np
from scipy.spatial.transform import Rotation

from src.ray_consistent_registration import apply_transform
from src.structural_similarity_registration import robust_structural_similarity


def test_robust_structural_similarity_recovers_proper_transform_with_outliers():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(200, 3)); target = source.copy()
    truth = np.eye(4); truth[:3, :3] = 1.04 * Rotation.from_rotvec([.03, -.04, .05]).as_matrix(); truth[:3, 3] = [.02, -.01, .03]
    target = apply_transform(target, truth); target[:20] += rng.normal(scale=.8, size=(20, 3))
    fitted, info = robust_structural_similarity(source, target, np.ones(len(source)), diagonal=5., trials=128)
    error = np.linalg.norm(apply_transform(source[20:], fitted) - target[20:], axis=1).mean()
    assert info["bounded_inliers"] > 100
    assert error < .04
    assert np.linalg.det(fitted[:3, :3]) > 0.
