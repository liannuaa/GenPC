import numpy as np

from src.saved_view_text_feedback import _bounded_image_shift_percent, _region_name


def test_region_name_uses_camera_image_coordinates_not_object_labels():
    assert _region_name(np.array([10., 90.]), (100, 100)) == "lower-left"
    assert _region_name(np.array([88., 12.]), (100, 100)) == "upper-right"
    assert _region_name(np.array([50., 50.]), (100, 100)) == "central"


def test_explicit_prompt_shift_is_residual_derived_and_bounded():
    assert _bounded_image_shift_percent(.001) == 2
    assert _bounded_image_shift_percent(.047) == 5
    assert _bounded_image_shift_percent(.50) == 8
