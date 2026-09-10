"""Camera-contract helpers for calibrated Pixal3D multi-view inference.

The image editor produces three object-centred orbit views. This module maps
those views to Pixal3D's public NeRF/Blender camera convention without using a
sample ID, category-specific geometry, a point-cloud metric, or ground truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
from PIL import Image


VIEWS = ("front", "side", "back", "right")


def camera_distance_for_fov(
    field_of_view_degrees: float,
    *,
    framing_margin: float = 1.1,
) -> float:
    """Return the canonical camera radius used by Pixal3D training renders.

    Pixal's latent carrier occupies a unit box. Half its image-plane extent is
    therefore 0.5; the framing margin leaves a category-independent border.
    The default reproduces the official 20-degree example radius (3.119205).
    """
    fov = float(field_of_view_degrees)
    margin = float(framing_margin)
    if not 1.0 < fov < 179.0:
        raise ValueError(f"field of view must be in (1, 179), got {fov}")
    if margin < 1.0:
        raise ValueError(f"framing margin must be >= 1, got {margin}")
    return 0.5 * margin / math.tan(math.radians(fov) / 2.0)


def native_camera_distance_from_render_manifest(manifest: Path) -> float:
    """Recover the render radius in the source Pixal carrier coordinates."""
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    transform = payload["ordered_similarity"]
    column_norms = [
        math.sqrt(sum(float(transform[row][column]) ** 2 for row in range(3)))
        for column in range(3)
    ]
    scale = sum(column_norms) / 3.0
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid Sim(3) scale in {manifest}: {scale}")
    centre = [float(value) for value in payload["centre"]]
    front = next(view for view in payload["views"] if view["name"] == "front")
    camera_position = [
        float(front["camera_pose"][row][3]) for row in range(3)
    ]
    world_distance = math.sqrt(
        sum((camera_position[axis] - centre[axis]) ** 2 for axis in range(3))
    )
    distance = world_distance / scale
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"invalid native camera distance in {manifest}: {distance}")
    return distance


def canonical_orbit_frames(distance: float) -> list[dict[str, object]]:
    """Build Pixal3D's canonical four-view orbit in official view order."""
    return orbit_frames_from_yaws(
        distance, dict(zip(VIEWS, (0.0, 90.0, 180.0, 270.0)))
    )


def orbit_frames_from_yaws(
    distance: float,
    yaw_degrees: Mapping[str, float],
) -> list[dict[str, object]]:
    """Build official Pixal camera matrices at arbitrary horizontal yaws."""
    d = float(distance)
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError(f"camera distance must be positive, got {d}")
    missing = [name for name in VIEWS if name not in yaw_degrees]
    if missing:
        raise KeyError(f"missing orbit yaws for views: {missing}")
    frames = []
    for name in VIEWS:
        yaw = math.radians(float(yaw_degrees[name]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        cosine = 0.0 if abs(cosine) < 1e-12 else cosine
        sine = 0.0 if abs(sine) < 1e-12 else sine
        frames.append({
            "file_path": f"{name}.png",
            "name": name,
            "transform_matrix": [
                [cosine, 0.0, sine, d * sine],
                [sine, 0.0, -cosine, -d * cosine],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })
    return frames


def prepare_pixal_mv_views(
    conditions: Mapping[str, Path],
    output_dir: Path,
    *,
    field_of_view_degrees: float,
    framing_margin: float = 1.1,
    camera_distance: float | None = None,
    source_manifest: Path | None = None,
    camera_templates: Mapping[str, Path] | None = None,
    orbit_yaws_degrees: Mapping[str, float] | None = None,
) -> Path:
    """Materialize four images and the official Pixal3D MV camera manifest."""
    output = Path(output_dir)
    missing = [name for name in VIEWS if name not in conditions]
    if missing:
        raise KeyError(f"missing required views: {missing}")
    if camera_templates is not None:
        missing_templates = [name for name in VIEWS if name not in camera_templates]
        if missing_templates:
            raise KeyError(f"missing camera templates: {missing_templates}")
    output.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}
    for name in VIEWS:
        source = Path(conditions[name]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / f"{name}.png"
        if camera_templates is None:
            shutil.copy2(source, destination)
        else:
            normalise_condition_to_camera_template(
                source, Path(camera_templates[name]).resolve(), destination
            )
        prepared[name] = destination

    distance = (
        camera_distance_for_fov(field_of_view_degrees, framing_margin=framing_margin)
        if camera_distance is None else float(camera_distance)
    )
    camera_lock_audit = None
    if camera_templates is not None:
        camera_lock_audit = audit_camera_locked_views(prepared, camera_templates)
        audit_path = output / "camera_lock_audit.json"
        audit_path.write_text(
            json.dumps(camera_lock_audit, indent=2) + "\n", encoding="utf-8"
        )
        if not camera_lock_audit["passed"]:
            raise ValueError(f"camera-lock audit failed; inspect {audit_path}")

    yaws = (
        dict(zip(VIEWS, (0.0, 90.0, 180.0, 270.0)))
        if orbit_yaws_degrees is None
        else {name: float(orbit_yaws_degrees[name]) for name in VIEWS}
    )
    payload = {
        "camera_angle_x": math.radians(float(field_of_view_degrees)),
        "mesh_scale": 1.0,
        "frames": orbit_frames_from_yaws(distance, yaws),
        "genpc_provenance": {
            "method": "camera1_gauged_canonical_four_view_orbit",
            "frame_0_is_camera1_gauge": True,
            "view_order": list(VIEWS),
            "native_camera_distance": distance,
            "framing_margin": float(framing_margin),
            "source_manifest": str(Path(source_manifest).resolve())
            if source_manifest is not None else None,
            "ground_truth_used": False,
            "camera_locked": camera_templates is not None,
            "camera_lock_audit": camera_lock_audit,
            "orbit_yaws_degrees": yaws,
        },
    }
    destination = output / "transforms.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def read_render_fov(manifest: Path) -> float:
    """Read the common horizontal FOV saved by the multi-view renderer."""
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    value = float(payload["field_of_view_degrees"])
    if not 1.0 < value < 179.0:
        raise ValueError(f"invalid field_of_view_degrees in {manifest}: {value}")
    return value


def read_render_yaws(manifest: Path) -> dict[str, float]:
    """Read actual orbit yaws, falling back to the historical orthogonal orbit."""
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    defaults = dict(zip(VIEWS, (0.0, 90.0, 180.0, 270.0)))
    records = {str(item["name"]): item for item in payload.get("views", [])}
    return {
        name: float(records.get(name, {}).get("orbit_yaw_degrees", defaults[name]))
        for name in VIEWS
    }



def _foreground_stats(path: Path, white_threshold: int = 245) -> dict[str, object]:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    mask = np.any(array < int(white_threshold), axis=2)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError(f"no foreground detected in {path}")
    centred = np.stack(
        [columns - columns.mean(), rows - rows.mean()], axis=1
    ).astype(np.float64)
    covariance = centred.T @ centred / max(len(centred), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    anisotropy = 1.0 - float(eigenvalues.min()) / max(float(eigenvalues.max()), 1e-12)
    return {
        "size": [int(image.width), int(image.height)],
        "centre": [
            float(columns.mean() / max(image.width, 1)),
            float(rows.mean() / max(image.height, 1)),
        ],
        "bbox": [
            float(columns.min() / max(image.width, 1)),
            float(rows.min() / max(image.height, 1)),
            float((columns.max() + 1) / max(image.width, 1)),
            float((rows.max() + 1) / max(image.height, 1)),
        ],
        "principal_axis_degrees": float(
            np.degrees(np.arctan2(major[1], major[0])) % 180.0
        ),
        "anisotropy": anisotropy,
        "foreground_fraction": float(mask.mean()),
    }


def normalise_condition_to_camera_template(
    edited_path: Path,
    template_path: Path,
    output_path: Path,
    *,
    white_threshold: int = 245,
) -> dict[str, object]:
    """Restore foreground centre and isotropic framing without warping shape."""
    edited = Image.open(edited_path).convert("RGB")
    template = Image.open(template_path).convert("RGB")
    edit = _foreground_stats(edited_path, white_threshold)
    reference = _foreground_stats(template_path, white_threshold)
    edit_box = np.asarray(edit["bbox"], dtype=np.float64)
    ref_box = np.asarray(reference["bbox"], dtype=np.float64)
    edit_extent = (edit_box[2:] - edit_box[:2]) * np.asarray(edited.size)
    ref_extent = (ref_box[2:] - ref_box[:2]) * np.asarray(template.size)
    scale = float(np.min(ref_extent / np.maximum(edit_extent, 1.0)))
    resized = edited.resize(
        (
            max(1, int(round(edited.width * scale))),
            max(1, int(round(edited.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    edit_centre = np.asarray(edit["centre"]) * np.asarray(edited.size) * scale
    ref_centre = np.asarray(reference["centre"]) * np.asarray(template.size)
    offset = np.rint(ref_centre - edit_centre).astype(int)
    canvas = Image.new("RGB", template.size, "white")
    canvas.paste(resized, tuple(offset))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return {
        "edited": str(Path(edited_path).resolve()),
        "template": str(Path(template_path).resolve()),
        "output": str(destination.resolve()),
        "isotropic_scale": scale,
        "paste_offset": offset.tolist(),
    }


def audit_camera_locked_views(
    conditions: Mapping[str, Path],
    camera_templates: Mapping[str, Path],
    *,
    max_centre_drift_ratio: float = 0.03,
    max_axis_drift_degrees: float = 15.0,
    min_axis_anisotropy: float = 0.2,
) -> dict[str, object]:
    """Reject gross camera drift while allowing supported silhouette changes."""
    records: dict[str, object] = {}
    passed = True
    for name in VIEWS:
        condition = _foreground_stats(Path(conditions[name]))
        template = _foreground_stats(Path(camera_templates[name]))
        centre_error = float(
            np.linalg.norm(
                np.asarray(condition["centre"]) - np.asarray(template["centre"])
            )
        )
        angle_error = abs(
            float(condition["principal_axis_degrees"])
            - float(template["principal_axis_degrees"])
        )
        angle_error = min(angle_error, 180.0 - angle_error)
        axis_observable = (
            float(condition["anisotropy"]) >= min_axis_anisotropy
            and float(template["anisotropy"]) >= min_axis_anisotropy
        )
        view_passed = centre_error <= max_centre_drift_ratio and (
            not axis_observable or angle_error <= max_axis_drift_degrees
        )
        records[name] = {
            "passed": bool(view_passed),
            "centre_drift_ratio": centre_error,
            "axis_observable": bool(axis_observable),
            "principal_axis_drift_degrees": angle_error,
            "condition": condition,
            "camera_template": template,
        }
        passed = passed and view_passed
    return {
        "method": "camera_locked_foreground_moment_audit",
        "passed": bool(passed),
        "thresholds": {
            "max_centre_drift_ratio": float(max_centre_drift_ratio),
            "max_axis_drift_degrees": float(max_axis_drift_degrees),
            "min_axis_anisotropy": float(min_axis_anisotropy),
        },
        "views": records,
    }
