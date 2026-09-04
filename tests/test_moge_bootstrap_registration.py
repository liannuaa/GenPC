import numpy as np
from scipy.spatial.transform import Rotation

from src.moge_bootstrap_registration import moge_coarse_proper_sim3


def test_moge_bootstrap_recovers_an_asymmetric_similarity_up_to_small_error():
    rng = np.random.default_rng(12)
    prior = rng.normal(size=(240, 3)) * np.array([1.3, .7, .35])
    transform = np.eye(4)
    transform[:3, :3] = 1.4 * Rotation.from_euler("xyz", [18., -11., 23.], degrees=True).as_matrix()
    transform[:3, 3] = (.35, -.20, .15)
    moge = prior @ transform[:3, :3].T + transform[:3, 3]
    moved, _, info = moge_coarse_proper_sim3(prior, moge, np.ones(len(moge)),
                                              prior_points=240, moge_points=240, iterations=5)
    assert info["selected_score"]["target_coverage"] > .95
    assert np.mean(np.linalg.norm(moved - moge, axis=1)) < .06
