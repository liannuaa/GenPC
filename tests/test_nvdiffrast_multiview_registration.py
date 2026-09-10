import numpy as np
from scipy.spatial.transform import Rotation

from src.nvdiffrast_multiview_registration import (
    compose_pivoted_sim3,
    initialise_from_condition_camera,
    initialise_from_multiview_condition_cameras,
)


def test_compose_pivoted_sim3_keeps_pivot_under_scale_and_rotation():
    base = np.eye(4)
    pivot = np.array([1.0, -2.0, 0.5])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = compose_pivoted_sim3(base, pivot, rotation, 1.4, np.zeros(3))
    mapped = pivot @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(mapped, pivot, atol=1e-10)


def test_compose_pivoted_sim3_adds_world_translation():
    base = np.eye(4)
    pivot = np.array([0.2, 0.3, -0.1])
    shift = np.array([0.4, -0.2, 0.7])
    transform = compose_pivoted_sim3(base, pivot, np.eye(3), 1.0, shift)
    mapped = pivot @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(mapped, pivot + shift, atol=1e-10)


def test_condition_camera_initialization_maps_centre_and_radius(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(__import__("json").dumps({
        "selected": {"camera_pose": np.eye(4).tolist()},
    }))
    target_pose = np.eye(4)
    target_pose[:3, :3] = np.array([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    source = np.array([
        [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    manifest = {
        "centre": [2.0, 3.0, 4.0], "radius": 2.0,
        "views": [{"name": "front", "camera_pose": target_pose.tolist()}],
    }
    transform, _ = initialise_from_condition_camera(source, selection, manifest)
    mapped = source @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(np.median(mapped, axis=0), manifest["centre"], atol=1e-10)
    assert np.isclose(np.quantile(np.linalg.norm(mapped - np.median(mapped, axis=0), axis=1), .995), 2.0)


def test_multiview_initialization_rejects_individually_better_inconsistent_view(tmp_path):
    identity = np.eye(4)
    quarter = np.eye(4)
    quarter[:3, :3] = np.array([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    paths = {}
    for name, target in (("front", identity), ("side", quarter)):
        path = tmp_path / f"{name}.json"
        inconsistent = np.eye(4)
        inconsistent[:3, :3] = Rotation.from_euler("x", 80, degrees=True).as_matrix()
        path.write_text(__import__("json").dumps({
            "top": [
                {"camera_pose": inconsistent.tolist(), "joint_score": 1.0,
                 "yaw_degrees": 0, "pitch_degrees": 0, "roll_degrees": 0},
                {"camera_pose": target.tolist(), "joint_score": 0.9,
                 "yaw_degrees": 1, "pitch_degrees": 0, "roll_degrees": 0},
            ],
        }))
        paths[name] = path
    source = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0, -1.0, 0], [0, 1.0, 0]])
    manifest = {
        "centre": [0, 0, 0], "radius": 1.0,
        "views": [
            {"name": "front", "camera_pose": identity.tolist()},
            {"name": "side", "camera_pose": quarter.tolist()},
        ],
    }
    _, record = initialise_from_multiview_condition_cameras(source, paths, manifest)
    assert record["selected_consistency_degrees"] < 1e-5
    assert record["selected_views"]["front"]["yaw_degrees"] == 1


def test_multiview_initialization_searches_full_camera_bank(tmp_path):
    identity = np.eye(4)
    quarter = np.eye(4)
    quarter[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    wrong = np.eye(4)
    wrong[:3, :3] = Rotation.from_euler("x", 80, degrees=True).as_matrix()
    paths = {}
    for name, target in (("front", identity), ("side", quarter)):
        candidates = [
            {"camera_pose": wrong.tolist(), "joint_score": 1.0 - 0.01 * index,
             "yaw_degrees": index, "pitch_degrees": 0, "roll_degrees": 0}
            for index in range(20)
        ]
        candidates.append({
            "camera_pose": target.tolist(), "joint_score": 0.70,
            "yaw_degrees": 99, "pitch_degrees": 0, "roll_degrees": 0,
        })
        path = tmp_path / f"{name}.json"
        path.write_text(__import__("json").dumps({
            "top": candidates[:12], "candidates": candidates,
        }))
        paths[name] = path
    source = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0, -1.0, 0], [0, 1.0, 0]])
    manifest = {
        "centre": [0, 0, 0], "radius": 1.0,
        "views": [
            {"name": "front", "camera_pose": identity.tolist()},
            {"name": "side", "camera_pose": quarter.tolist()},
        ],
    }
    _, record = initialise_from_multiview_condition_cameras(source, paths, manifest)
    assert record["candidate_count_per_view"] == {"front": 21, "side": 21}
    assert record["selected_consistency_degrees"] < 1e-5
    assert record["selected_views"]["front"]["yaw_degrees"] == 99
