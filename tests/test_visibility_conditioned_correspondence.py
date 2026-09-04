import numpy as np

from src.visibility_conditioned_correspondence import local_shape_descriptor


def test_local_shape_descriptor_is_translation_invariant_and_has_normals():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(128, 3))
    first, normal = local_shape_descriptor(points, neighbours=12)
    second, _ = local_shape_descriptor(points + np.array([5., -2., .7]), neighbours=12)
    assert first.shape == (128, 4)
    assert normal.shape == (128, 3)
    assert np.allclose(first, second, atol=1e-8)
    assert np.allclose(np.linalg.norm(normal, axis=1), 1., atol=1e-6)
