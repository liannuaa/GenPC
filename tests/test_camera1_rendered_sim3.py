import numpy as np

from src.camera1_rendered_sim3 import _random_rotations


def test_random_rotation_bank_is_deterministic_and_proper():
    first = _random_rotations(32, 6145)
    second = _random_rotations(32, 6145)
    assert first.shape == (33, 3, 3)
    assert np.array_equal(first, second)
    assert np.allclose(first @ np.swapaxes(first, 1, 2), np.eye(3), atol=1e-8)
    assert np.all(np.linalg.det(first) > 0.0)
