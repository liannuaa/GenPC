"""Saved-camera assets for scene MoGe instance partials.

The scene wrapper never rasterises a depth image.  It only needs the original
scene camera as projective metadata so that the frozen 2D+3D registration code
can compare a direct GPT semantic image with the instance partial.  This small
camera class is deliberately serialisable with :func:`torch.save` and exposes
the same ``transform`` interface as the historical Kaolin camera.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.scene_completion.contracts import SceneManifest
from src.scene_completion.io import load_colored_points
from src.scene_completion.scene_moge import SceneMoGeObservation


class SceneMoGeCamera:
    """Perspective camera for points already expressed in a MoGe scene frame.

    MoGe points follow the usual camera convention: ``x / z`` grows right and
    ``y / z`` grows down in the source RGB image.  The saved-camera projector
    used by the frozen mainline flips its vertical coordinate at rasterisation,
    hence this transform returns ``-y / z`` to preserve the original scene
    image orientation exactly.
    """

    def __init__(self, intrinsics: np.ndarray):
        value = np.asarray(intrinsics, dtype=np.float64)
        if value.shape != (3, 3):
            raise ValueError(f"scene MoGe intrinsics must be [3,3], got {value.shape}")
        self.intrinsics = value

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"expected [N,3] points, got {tuple(points.shape)}")
        depth = points[:, 2]
        epsilon = torch.finfo(points.dtype).eps if points.is_floating_point() else 1e-8
        safe_depth = torch.where(depth.abs() >= epsilon, depth, torch.full_like(depth, epsilon))
        return torch.stack((points[:, 0] / safe_depth, -points[:, 1] / safe_depth, depth), dim=1)


def _point_uv(partial: np.ndarray, camera: SceneMoGeCamera, *, padding: float) -> np.ndarray:
    if not 0.0 <= float(padding) < 0.5:
        raise ValueError("scene camera padding must be in [0, .5)")
    points = torch.as_tensor(np.asarray(partial), dtype=torch.float32)
    with torch.no_grad():
        camera_points = camera.transform(points).cpu().numpy()
    xy = camera_points[:, :2]
    center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    scale = max(float(np.ptp(xy, axis=0).max()), 1e-8)
    return (xy - center) / scale * (1.0 - 2.0 * float(padding)) + 0.5


def write_scene_camera_assets(
    manifest: SceneManifest,
    observation: SceneMoGeObservation,
    *,
    output_root: Path,
    padding: float = 0.15,
) -> dict:
    """Materialise only Camera-1 metadata for direct scene-object semantics.

    The partial PLYs must be the mask-indexed scene MoGe subsets written by
    :func:`extract_scene_instances`; they remain in the original shared camera
    frame.  No depth raster or depth-conditioned semantic image is produced.
    """
    output_root = Path(output_root).resolve()
    records: list[dict] = []
    for item in manifest.instances:
        partial_path = output_root / "inputs" / "partial" / f"{item.instance_id}.ply"
        partial, _ = load_colored_points(partial_path)
        camera_dir = output_root / "inputs" / "camera" / item.instance_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        camera = SceneMoGeCamera(observation.intrinsics)
        torch.save(camera, camera_dir / "camera.pth")
        uv = _point_uv(partial, camera, padding=padding)
        np.save(camera_dir / "point_uv.npy", uv)
        record = {
            "id": item.instance_id,
            "partial": str(partial_path.resolve()),
            "camera": str((camera_dir / "camera.pth").resolve()),
            "point_uv": str((camera_dir / "point_uv.npy").resolve()),
            "point_count": int(len(partial)),
            "padding": float(padding),
            "coordinate_frame": "shared_pixal_moge_scene_camera",
            "projection": "perspective x/z, -y/z; SavedCameraProjector restores image-down y",
            "semantic_generation": "direct_gpt_instance_completion_no_depth_raster",
        }
        (camera_dir / "scene_camera_info.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
    summary = {
        "method": "scene_moge_partial_camera_metadata_only",
        "source_image": str(observation.image_path),
        "depth_to_semantic_used": False,
        "camera_role": "registration_and_mesh_placement_metadata_only",
        "instances": records,
    }
    (output_root / "scene_camera_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
