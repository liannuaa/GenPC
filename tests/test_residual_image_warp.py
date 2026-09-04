import numpy as np

from src.residual_image_warp import _select_controls


def test_control_thinning_is_deterministic_and_bounded():
    source = np.column_stack((np.arange(20, dtype=float), np.zeros(20)))
    displacement = np.column_stack((np.ones(20), np.zeros(20)))
    selected_source, selected_displacement = _select_controls(source, displacement, 6)
    assert selected_source.shape == (6, 2)
    assert selected_displacement.shape == (6, 2)
    assert np.all(selected_source[1:, 0] >= selected_source[:-1, 0])
