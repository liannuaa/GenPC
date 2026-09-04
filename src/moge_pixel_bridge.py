"""MoGe inference, object masking, and pixel-index correspondences.

This module is deliberately independent of the registration runners.  It is
the only bridge needed between a semantic image, its MoGe reconstruction, and
the saved-view partial pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PixelIndexResult:
    partial_to_moge: np.ndarray
    pixel_distances: np.ndarray
    matched_partial_indices: np.ndarray
    matched_moge_indices: np.ndarray


@dataclass
class MogeObjectPoints:
    points: np.ndarray
    colors: np.ndarray
    pixel_xy: np.ndarray
    original_indices: np.ndarray


def point_uv_to_pixel_xy(point_uv, image_size: int, *, flip_y: bool = False):
    uv = np.asarray(point_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"point_uv must have shape [N, 2], got {uv.shape}")
    uv = uv.copy()
    if flip_y:
        uv[:, 1] = 1.0 - uv[:, 1]
    pixel_xy = np.rint(uv * (int(image_size) - 1)).astype(np.int64)
    valid = np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0)
    valid &= (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
    return pixel_xy, valid


def build_partial_to_moge_index(
    point_uv,
    image_size: int,
    moge_pixel_xy,
    max_pixel_distance: float,
    *,
    flip_y: bool = False,
) -> PixelIndexResult:
    pixel_xy, valid_partial = point_uv_to_pixel_xy(point_uv, image_size, flip_y=flip_y)
    moge_pixel_xy = np.asarray(moge_pixel_xy, dtype=np.float64)
    if moge_pixel_xy.ndim != 2 or moge_pixel_xy.shape[1] != 2:
        raise ValueError(f"moge_pixel_xy must have shape [M, 2], got {moge_pixel_xy.shape}")
    partial_to_moge = np.full(len(pixel_xy), -1, dtype=np.int64)
    pixel_distances = np.full(len(pixel_xy), np.inf, dtype=np.float64)
    if not len(moge_pixel_xy) or not valid_partial.any():
        return PixelIndexResult(
            partial_to_moge, pixel_distances,
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        )
    valid_indices = np.where(valid_partial)[0]
    distances, nearest = cKDTree(moge_pixel_xy).query(pixel_xy[valid_indices], k=1)
    keep = distances <= float(max_pixel_distance)
    matched_partial = valid_indices[keep].astype(np.int64)
    matched_moge = nearest[keep].astype(np.int64)
    partial_to_moge[matched_partial] = matched_moge
    pixel_distances[matched_partial] = distances[keep]
    return PixelIndexResult(partial_to_moge, pixel_distances, matched_partial, matched_moge)


def colors_for_moge_hits(num_points: int, hit_indices) -> np.ndarray:
    colors = np.full((int(num_points), 3), 0.55, dtype=np.float64)
    hit_indices = np.asarray(hit_indices, dtype=np.int64)
    hit_indices = hit_indices[(hit_indices >= 0) & (hit_indices < int(num_points))]
    if len(hit_indices):
        colors[np.unique(hit_indices)] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return colors


def prepare_object_mask(object_mask, alpha_threshold: int, erode_pixels: int = 0) -> np.ndarray:
    object_mask = np.asarray(object_mask)
    if object_mask.ndim == 3:
        object_mask = object_mask[..., -1]
    if object_mask.ndim != 2:
        raise ValueError(f"object_mask must be 2D or RGBA-like, got {object_mask.shape}")
    binary = (object_mask >= int(alpha_threshold)).astype(np.uint8) * 255
    if int(erode_pixels) > 0:
        import cv2

        kernel_size = int(erode_pixels) * 2 + 1
        binary = cv2.erode(binary, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    return binary


def filter_moge_points_by_object_mask(
    points, colors, pixel_xy, object_mask, alpha_threshold: int, erode_pixels: int = 0,
) -> MogeObjectPoints:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    object_mask = prepare_object_mask(object_mask, alpha_threshold, erode_pixels)
    if len(points) != len(colors) or len(points) != len(pixel_xy):
        raise ValueError("points, colors, and pixel_xy must share the first dimension")
    xy = np.rint(pixel_xy).astype(np.int64)
    height, width = object_mask.shape
    in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    keep = np.zeros(len(xy), dtype=bool)
    valid = np.where(in_bounds)[0]
    keep[valid] = object_mask[xy[valid, 1], xy[valid, 0]] > 0
    return MogeObjectPoints(points[keep], colors[keep], pixel_xy[keep], np.where(keep)[0].astype(np.int64))


def load_image_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)[..., :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_alpha_mask(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return image
    if image.shape[-1] == 4:
        return image[..., 3]
    raise ValueError(f"Expected RMBG output with alpha channel, got shape {image.shape}")


def save_mask_png(path: Path, mask) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8))


def run_rmbg_mask(image_path: Path, output_rgba_path: Path, model_path: Path) -> np.ndarray:
    from tools.RMBG import RMBG_pred

    RMBG_pred(str(image_path), str(output_rgba_path), model_path=str(model_path))
    return load_alpha_mask(output_rgba_path)


def _serialize_output_value(value):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 16 else {"shape": list(value.shape), "dtype": str(value.dtype)}
    return value.item() if np.isscalar(value) else str(type(value).__name__)


def run_moge_with_pixels(image_path: Path, pretrained: Path, device: str, fp16: bool):
    import torch
    from moge.model.v2 import MoGeModel

    pretrained = Path(pretrained)
    if pretrained.is_dir():
        pretrained = pretrained / "model.pt"
    image_rgb = load_image_rgb(image_path)
    image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    model = MoGeModel.from_pretrained(str(pretrained)).to(device).eval()
    with torch.no_grad():
        if fp16 and str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.infer(image_tensor)
        else:
            output = model.infer(image_tensor)
    points = output["points"].detach().float().cpu().numpy()
    valid = output["mask"].detach().cpu().numpy().astype(bool)
    valid &= np.isfinite(points).all(axis=-1)
    valid &= np.linalg.norm(points, axis=-1) > 1e-8
    ys, xs = np.where(valid)
    info = {
        "pretrained": str(pretrained), "image_path": str(image_path),
        "image_hw": list(image_rgb.shape[:2]), "valid_points": int(valid.sum()),
        "output_keys": {key: _serialize_output_value(value) for key, value in output.items() if key not in {"points", "mask"}},
        "camera2_frame": "moge_camera_coordinates_identity",
    }
    return (points[valid].astype(np.float64),
            (image_rgb[valid].astype(np.float64) / 255.0).clip(0, 1),
            np.stack([xs, ys], axis=1).astype(np.float64), info)
