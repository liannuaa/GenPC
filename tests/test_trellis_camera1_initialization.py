import numpy as np

from src.trellis_camera1_initialization import _camera_rotation


class _Camera:
    R = __import__("torch").eye(3)[None]


class _Projector:
    camera = _Camera()


def test_camera_rotation_is_proper():
    rotation = _camera_rotation(_Projector())
    np.testing.assert_allclose(rotation, np.eye(3))
