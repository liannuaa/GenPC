import numpy as np

from src.arbor_agent_constraints import partial_to_arbor_hull


def test_arbor_hull_is_normalized_watertight_components_and_reversible():
    points = np.array([[2., 3., 5.], [6., 5., 7.], [4., 9., 6.]])
    vertices, faces, contract = partial_to_arbor_hull(points, max_points=3)
    assert vertices.shape == (18, 3)
    assert faces.shape == (24, 3)
    assert contract.support_radius > 0
    restored = np.array([[0., 0., 0.]]) @ contract.constraint_to_raw[:3, :3].T + contract.constraint_to_raw[:3, 3]
    assert np.allclose(restored, [[4., 6., 6.]])
