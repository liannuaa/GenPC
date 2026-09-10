import numpy as np

from src.oracle_similarity import gt_oracle_similarity


def test_oracle_similarity_recovers_synthetic_proper_sim3():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(600, 3)) * np.array([1.0, 0.6, 0.3])
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    target = source @ rotation.T * 1.3 + np.array([0.2, -0.4, 0.7])
    result, info = gt_oracle_similarity(source, target, sample_points=600, iterations=8)
    assert info["ground_truth_used"] is True
    assert np.mean(np.linalg.norm(result - target, axis=1)) < 1e-5
