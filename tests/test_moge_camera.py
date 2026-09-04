import numpy as np

from src.moge_camera import MoGeProjector


def test_moge_projector_uses_native_perspective_intrinsics():
    projector = MoGeProjector(np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]), (100, 200))
    pixel, depth = projector.project(np.array([[0., 0., 2.], [1., 0., 2.]]))
    assert np.allclose(pixel, np.array([[100., 50.], [200., 50.]]))
    assert np.allclose(depth, np.array([2., 2.]))
