import json
import math
from pathlib import Path

from PIL import Image

from src.pixal_multiview import (
    audit_camera_locked_views,
    VIEWS,
    native_camera_distance_from_render_manifest,
    normalise_condition_to_camera_template,
    camera_distance_for_fov,
    canonical_orbit_frames,
    orbit_frames_from_yaws,
    prepare_pixal_mv_views,
)


def test_official_example_distance_is_reproduced() -> None:
    assert math.isclose(camera_distance_for_fov(20.0), 3.119205, rel_tol=2e-6)


def test_four_view_orbit_has_fixed_semantic_orientation() -> None:
    d = 2.0
    frames = canonical_orbit_frames(d)
    assert tuple(frame["name"] for frame in frames) == VIEWS
    assert frames[0]["transform_matrix"][1][3] == -d
    assert frames[1]["transform_matrix"][0][3] == d
    assert frames[2]["transform_matrix"][1][3] == d
    assert frames[0]["transform_matrix"][0][:3] == [1.0, 0.0, 0.0]
    assert frames[2]["transform_matrix"][0][:3] == [-1.0, 0.0, 0.0]
    assert frames[3]["transform_matrix"][0][3] == -d
    assert frames[3]["transform_matrix"][1][:3] == [-1.0, 0.0, 0.0]


def test_arbitrary_orbit_yaws_preserve_official_camera_convention() -> None:
    frames = orbit_frames_from_yaws(
        2.0, dict(zip(VIEWS, (0.0, 45.0, 135.0, 225.0)))
    )
    assert tuple(frame["name"] for frame in frames) == VIEWS
    assert frames[0] == canonical_orbit_frames(2.0)[0]
    expected = 2.0 / math.sqrt(2.0)
    assert math.isclose(frames[1]["transform_matrix"][0][3], expected)
    assert math.isclose(frames[1]["transform_matrix"][1][3], -expected)


def test_prepare_manifest_is_sample_and_category_independent(tmp_path: Path) -> None:
    conditions = {}
    for name in VIEWS:
        path = tmp_path / f"source_{name}.png"
        Image.new("RGB", (32, 32), "white").save(path)
        conditions[name] = path
    destination = tmp_path / "mv"
    manifest = prepare_pixal_mv_views(
        conditions,
        destination,
        field_of_view_degrees=38.0,
    )
    payload = json.loads(manifest.read_text())
    assert [frame["file_path"] for frame in payload["frames"]] == [
        "front.png", "side.png", "back.png", "right.png"
    ]
    assert payload["genpc_provenance"]["ground_truth_used"] is False
    assert "sample" not in json.dumps(payload).lower()


def test_native_camera_distance_removes_registered_sim3_scale(tmp_path: Path) -> None:
    manifest = tmp_path / "render_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ordered_similarity": [
                    [2.0, 0.0, 0.0, 4.0],
                    [0.0, 2.0, 0.0, 5.0],
                    [0.0, 0.0, 2.0, 6.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "centre": [1.0, 2.0, 3.0],
                "views": [
                    {
                        "name": "front",
                        "camera_pose": [
                            [1.0, 0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0, 2.0],
                            [0.0, 0.0, 1.0, 9.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
            }
        )
    )
    assert native_camera_distance_from_render_manifest(manifest) == 3.0



def test_camera_lock_normalises_framing_and_rejects_rotation(tmp_path: Path) -> None:
    templates = {}
    conditions = {}
    for name in VIEWS:
        template = tmp_path / f"template_{name}.png"
        condition = tmp_path / f"condition_{name}.png"
        canvas = Image.new("RGB", (120, 100), "white")
        for x in range(48, 72):
            for y in range(15, 85):
                canvas.putpixel((x, y), (20, 30, 40))
        canvas.save(template)
        canvas.resize((180, 150)).save(condition)
        templates[name] = template
        conditions[name] = condition

    prepared = {}
    for name in VIEWS:
        output = tmp_path / f"prepared_{name}.png"
        normalise_condition_to_camera_template(
            conditions[name], templates[name], output
        )
        prepared[name] = output
    assert audit_camera_locked_views(prepared, templates)["passed"] is True

    rotated = Image.new("RGB", (120, 100), "white")
    for x in range(20, 100):
        for y in range(40, 60):
            rotated.putpixel((x, y), (20, 30, 40))
    rotated.save(prepared["right"])
    report = audit_camera_locked_views(prepared, templates)
    assert report["passed"] is False
    assert report["views"]["right"]["principal_axis_drift_degrees"] > 80.0
