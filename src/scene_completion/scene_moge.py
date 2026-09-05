"""Shared-camera scene observation from the Pixal MoGe-2 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.moge_pixel_bridge import run_moge_with_pixels
from src.scene_completion.io import write_colored_points


SCENE_MOGE_CACHE = "pixal_moge_scene_observation.npz"


@dataclass(frozen=True)
class SceneMoGeObservation:
    """Visible scene points, RGB and their source-image pixels in one frame."""

    image_path: Path
    points: np.ndarray
    colors: np.ndarray
    pixel_xy: np.ndarray
    intrinsics: np.ndarray
    image_hw: tuple[int, int]
    metadata: dict


def _intrinsics_from_info(info: dict) -> np.ndarray:
    value = info.get("output_keys", {}).get("intrinsics")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            "Pixal MoGe observation did not provide a 3x3 intrinsics matrix; "
            f"got {matrix.shape}"
        )
    return matrix


def infer_scene_moge(
    image_path: Path,
    *,
    moge_model: Path,
    output_dir: Path,
    device: str = "cuda",
    fp16: bool = True,
) -> SceneMoGeObservation:
    """Infer a scene depth cloud using Pixal's own MoGe-2 tensor contract.

    This intentionally calls the shared ``run_moge_with_pixels`` utility, whose
    preprocessing is exactly the one used for Pixal native observations:
    ``PIL RGB -> float/255 CHW -> MoGeModel.infer``.  The output coordinate
    frame is therefore the single camera frame into which all object priors
    are eventually registered.
    """
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    points, colors, pixel_xy, info = run_moge_with_pixels(
        image_path=image_path,
        pretrained=Path(moge_model),
        device=device,
        fp16=fp16,
    )
    intrinsics = _intrinsics_from_info(info)
    image_hw = tuple(int(value) for value in info["image_hw"])
    if points.shape != colors.shape or pixel_xy.shape != (len(points), 2):
        raise ValueError("Pixal MoGe points, colors and pixels must have matching [N,3]/[N,2] lengths")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / SCENE_MOGE_CACHE
    np.savez_compressed(
        cache,
        schema_version=np.asarray(1, dtype=np.int64),
        source_image=np.asarray(str(image_path)),
        points=points.astype(np.float64),
        colors=colors.astype(np.float64),
        pixel_xy=pixel_xy.astype(np.float64),
        intrinsics=intrinsics.astype(np.float64),
        image_hw=np.asarray(image_hw, dtype=np.int64),
    )
    write_colored_points(output_dir / "pixal_moge_scene_visible.ply", points, colors)
    metadata = {
        "method": "pixal_moge2_scene_visible_observation",
        "coordinate_frame": "Pixal MoGe camera coordinates for the unmodified RGB scene",
        "source_image": str(image_path),
        "moge_model": str(Path(moge_model).resolve()),
        "fp16": bool(fp16),
        "device": str(device),
        "point_count": int(len(points)),
        "image_hw": list(image_hw),
        "intrinsics": intrinsics.tolist(),
        "cache": str(cache),
        "pixal_moge_contract": (
            "PIL RGB -> float/255 CHW -> MoGeModel.infer; same MoGe-2 checkpoint "
            "and input convention as Pixal native registration"
        ),
        "moge_output": info,
    }
    (output_dir / "pixal_moge_scene_info.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return SceneMoGeObservation(
        image_path=image_path,
        points=points,
        colors=colors,
        pixel_xy=pixel_xy,
        intrinsics=intrinsics,
        image_hw=image_hw,
        metadata=metadata,
    )


def load_scene_moge(output_dir: Path) -> SceneMoGeObservation:
    """Load a previously frozen scene observation without rerunning MoGe."""
    output_dir = Path(output_dir).resolve()
    cache = output_dir / SCENE_MOGE_CACHE
    info_path = output_dir / "pixal_moge_scene_info.json"
    if not cache.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"missing scene MoGe cache under {output_dir}")
    with np.load(cache, allow_pickle=False) as archive:
        if int(archive["schema_version"]) != 1:
            raise ValueError(f"unsupported scene MoGe cache schema: {cache}")
        points = np.asarray(archive["points"], dtype=np.float64)
        colors = np.asarray(archive["colors"], dtype=np.float64)
        pixel_xy = np.asarray(archive["pixel_xy"], dtype=np.float64)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
        image_hw = tuple(int(value) for value in archive["image_hw"])
        image_path = Path(str(archive["source_image"].item())).resolve()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"invalid scene MoGe point cache: {cache}")
    if colors.shape != points.shape or pixel_xy.shape != (len(points), 2):
        raise ValueError(f"inconsistent scene MoGe cache arrays: {cache}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"invalid scene MoGe intrinsics: {cache}")
    return SceneMoGeObservation(
        image_path=image_path,
        points=points,
        colors=colors,
        pixel_xy=pixel_xy,
        intrinsics=intrinsics,
        image_hw=image_hw,
        metadata=json.loads(info_path.read_text(encoding="utf-8")),
    )
