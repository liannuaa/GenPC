import numpy as np

from src.ray_consistent_registration import zbuffer_indices
from src.zbuffer import zbuffer_depth_with_indices


def test_square_splat_keeps_historical_offset_tie_order():
    # Point 0 reaches (0, 0) through the first (-1, -1) splat before point 1
    # reaches it through (0, 0).  Equal depths must preserve that old order.
    uv = np.array([[1.2, 1.2], [0.2, 0.2], [2.2, 1.2]])
    depth = np.array([1.0, 1.0, 0.5])
    rendered, mask, indices = zbuffer_indices(uv, depth, (4, 4), splat_radius=1)
    assert mask[0, 0]
    assert rendered[0, 0] == 1.0
    assert indices[0, 0] == 0
    assert indices[1, 2] == 2


def test_circular_splat_returns_zero_only_for_empty_pixels():
    uv = np.array([[1.2, 1.2], [1.2, 1.2], [9.0, 9.0]])
    depth = np.array([1.0, 0.5, 2.0])
    rendered, mask, indices = zbuffer_depth_with_indices(uv, depth, (4, 4), splat_radius=1)
    assert mask[1, 1]
    assert rendered[1, 1] == 0.5
    assert indices[1, 1] == 1
    assert not mask[3, 3]
    assert rendered[3, 3] == 0.0
    assert indices[3, 3] == -1
