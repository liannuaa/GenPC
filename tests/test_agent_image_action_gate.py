import numpy as np

from src.agent_image_action_gate import (
    accept_camera_locked_image_action,
    measure_camera_locked_image_action,
)


def _canvas(box):
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = 40
    return image


def test_camera_locked_gate_accepts_same_silhouette():
    evidence = measure_camera_locked_image_action(_canvas((30, 25, 70, 75)), _canvas((30, 25, 70, 75)))
    assert evidence.iou == 1.
    assert accept_camera_locked_image_action(evidence)


def test_camera_locked_gate_rejects_shrunken_target():
    evidence = measure_camera_locked_image_action(_canvas((25, 20, 75, 80)), _canvas((35, 30, 65, 70)))
    assert evidence.bbox_scale_ratio < .9
    assert not accept_camera_locked_image_action(evidence)


def test_camera_locked_gate_rejects_translated_target():
    evidence = measure_camera_locked_image_action(_canvas((25, 20, 75, 80)), _canvas((35, 20, 85, 80)))
    assert evidence.centroid_shift_ratio > .035
    assert not accept_camera_locked_image_action(evidence)
