import numpy as np

from src.multiview_diagnostics import (
    apply_transform,
    estimate_ordered_similarity,
    look_at_pose,
    rotate_about_axis,
    select_informative_orbit_yaws,
)
from src.multiview_partial_evidence import (
    VIEW_ORDER, build_prompt, compose_contact_sheet, compose_grid, split_grid,
)


def test_ordered_similarity_recovers_proper_sim3():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(128, 3))
    angle = np.deg2rad(37.0)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    target = source @ (1.7 * rotation).T + np.array([0.2, -0.4, 0.6])
    transform, rmse = estimate_ordered_similarity(source, target)
    assert rmse < 1e-10
    assert np.allclose(apply_transform(source, transform), target, atol=1e-10)


def test_orbit_and_look_at_are_orthonormal():
    front = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    side = rotate_about_axis(front, up, 90.0)
    assert np.allclose(side, np.array([1.0, 0.0, 0.0]), atol=1e-8)
    pose = look_at_pose(side * 3.0, np.zeros(3), up)
    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8)
    assert np.linalg.det(pose[:3, :3]) > 0.0


def test_four_view_grid_roundtrip(tmp_path):
    from PIL import Image

    colours = ("red", "green", "blue", "yellow")
    board = compose_grid(
        [Image.new("RGB", (32, 32), colour) for colour in colours],
        tmp_path / "board.png",
    )
    outputs = split_grid(board, tmp_path / "views")
    assert tuple(outputs) == VIEW_ORDER
    for name, colour in zip(VIEW_ORDER, colours):
        assert Image.open(outputs[name]).getpixel((16, 16)) == Image.new("RGB", (1, 1), colour).getpixel((0, 0))


def test_six_view_contact_sheet_layout(tmp_path):
    from PIL import Image

    board = compose_contact_sheet(
        [Image.new("RGB", (24, 16), (index, 0, 0)) for index in range(6)],
        tmp_path / "six.png",
        columns=3,
    )
    image = Image.open(board)
    assert image.size == (72, 32)
    assert image.getpixel((60, 24)) == (5, 0, 0)


def test_partial_evidence_prompt_treats_missing_pixels_as_unknown():
    prompt = build_prompt("an object")
    assert "FRONT, SIDE, BACK, RIGHT" in prompt
    assert "UNKNOWN" in prompt
    assert "must never cause deletion" in prompt
    assert "keeps the complete RGB prior visible" in prompt
    assert "Red points are current prior surface locations" in prompt


def test_informative_orbit_keeps_camera1_and_separates_views():
    rng = np.random.default_rng(19)
    points = rng.normal(size=(1200, 3)) * np.array([1.0, 0.4, 0.7])
    result = select_informative_orbit_yaws(
        partial=points,
        centre=np.zeros(3),
        front_direction=np.array([0.0, 0.0, 1.0]),
        up_direction=np.array([0.0, 1.0, 0.0]),
        camera_distance=5.0,
        field_of_view_degrees=38.0,
        candidate_yaw_step_degrees=15.0,
        min_yaw_separation_degrees=45.0,
        resolution=96,
    )
    yaws = [item["yaw_degrees"] for item in result["selected"]]
    assert yaws[0] == 0.0
    assert len(yaws) == 4
    for index, yaw in enumerate(yaws):
        for previous in yaws[:index]:
            difference = abs((yaw - previous) % 360.0)
            assert min(difference, 360.0 - difference) >= 45.0
    assert result["ground_truth_used"] is False
    assert result["method"] == "camera1_plus_ranked_visible_partial_support"


def test_informative_orbit_supports_six_separated_views():
    rng = np.random.default_rng(23)
    points = rng.normal(size=(1200, 3)) * np.array([1.0, 0.4, 0.7])
    result = select_informative_orbit_yaws(
        partial=points,
        centre=np.zeros(3),
        front_direction=np.array([0.0, 0.0, 1.0]),
        up_direction=np.array([0.0, 1.0, 0.0]),
        camera_distance=5.0,
        field_of_view_degrees=38.0,
        num_views=6,
        candidate_yaw_step_degrees=15.0,
        min_yaw_separation_degrees=45.0,
        resolution=96,
    )
    yaws = [item["yaw_degrees"] for item in result["selected"]]
    assert yaws[0] == 0.0
    assert len(yaws) == 6
    for index, yaw in enumerate(yaws):
        for previous in yaws[:index]:
            difference = abs((yaw - previous) % 360.0)
            assert min(difference, 360.0 - difference) >= 45.0
