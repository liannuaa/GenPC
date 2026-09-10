import numpy as np
from PIL import Image
import json

from src.trellis_condition_registration import (
    ConditionCamera,
    _foreground_mask,
    _axis_angle_matrix,
    _make_transform,
    _project_perspective,
    load_condition_contract,
)


def test_perspective_projection_centres_camera_target():
    pose = np.eye(4)
    pose[2, 3] = 2.0
    camera = ConditionCamera("front", pose, np.ones((64, 64), dtype=bool))
    uv, depth = _project_perspective(np.array([[0.0, 0.0, 0.0]]), camera, 40.0)
    np.testing.assert_allclose(uv, [[0.5, 0.5]], atol=1e-8)
    np.testing.assert_allclose(depth, [2.0], atol=1e-8)


def test_projection_applies_only_image_plane_nuisance_shift():
    pose = np.eye(4)
    pose[2, 3] = 2.0
    camera = ConditionCamera(
        "front", pose, np.ones((64, 64), dtype=bool),
        image_scale=1.0, image_offset=(0.2, -0.1),
    )
    uv, _ = _project_perspective(np.array([[0.0, 0.0, 0.0]]), camera, 40.0)
    np.testing.assert_allclose(uv, [[0.6, 0.45]], atol=1e-8)


def test_condition_contract_self_calibrates_recentering_without_absorbing_scale(tmp_path):
    original = np.full((128, 128, 3), 255, dtype=np.uint8)
    original[40:88, 44:76] = 20
    edited = np.full_like(original, 255)
    edited[35:93, 54:102] = 20
    original_path, edited_path = tmp_path / "original.png", tmp_path / "edited.png"
    Image.fromarray(original).save(original_path)
    Image.fromarray(edited).save(edited_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "field_of_view_degrees": 40.0,
        "views": [{
            "name": "front", "image": str(original_path),
            "camera_pose": np.eye(4).tolist(),
        }],
    }))
    cameras, manifest = load_condition_contract(
        manifest_path, {"front": edited_path}, resolution=128,
    )
    assert cameras[0].image_scale == 1.0
    assert np.linalg.norm(cameras[0].image_offset) > 0.05
    assert manifest["condition_image_calibration"]["front"]["observed_axis_scale"][0] > 1.2


def test_transform_maps_source_centre_to_world_centre():
    source_centre = np.array([1.0, 2.0, 3.0])
    world_centre = np.array([-2.0, 0.5, 4.0])
    rotation = __import__("scipy").spatial.transform.Rotation.from_euler("z", 37, degrees=True).as_matrix()
    transform = _make_transform(source_centre, rotation, 1.7, world_centre)
    mapped = source_centre @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(mapped, world_centre, atol=1e-10)


def test_foreground_mask_rejects_white_background(tmp_path):
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    image[36:96, 44:84] = np.array([40, 70, 90], dtype=np.uint8)
    path = tmp_path / "condition.png"
    Image.fromarray(image).save(path)
    mask = _foreground_mask(path, 64)
    assert mask.dtype == bool
    assert 400 < int(mask.sum()) < 1000


def test_axis_angle_matrix_is_proper_and_differentiable():
    import torch

    vector = torch.tensor([0.1, -0.2, 0.05], requires_grad=True)
    matrix = _axis_angle_matrix(vector)
    np.testing.assert_allclose(
        (matrix.T @ matrix).detach().numpy(), np.eye(3), atol=1e-5,
    )
    assert float(torch.det(matrix).detach()) > 0.999
    matrix.sum().backward()
    assert torch.isfinite(vector.grad).all()
