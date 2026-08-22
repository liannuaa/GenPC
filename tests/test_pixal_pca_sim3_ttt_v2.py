import numpy as np

from scripts.run_pixal_pca_sim3_ttt_v2 import make_transform, proper_pca_rotations


def test_rotation_pool_has_24_proper_rotations():
    rng = np.random.default_rng(17)
    source = rng.normal(size=(2000, 3)) * np.array([3., 2., 1.])
    target = source @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).T
    rotations = proper_pca_rotations(source, target)
    assert len(rotations) == 24
    assert all(np.linalg.det(x["rotation"]) > .999999 for x in rotations)


def test_transform_is_uniform_sim3():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transform = make_transform(rotation, .73, np.array([.1, -.2, .3]))
    np.testing.assert_allclose(np.linalg.svd(transform[:3, :3], compute_uv=False), .73)
