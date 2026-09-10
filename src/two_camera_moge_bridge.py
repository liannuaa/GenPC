"""Build a scan-to-Pixal MoGe bridge from two calibrated image planes.

The partial scan is tied to the Redwood depth image (camera 1) through its
saved ``point_uv`` coordinates.  Pixal, in contrast, estimates a camera from
its own foreground-normalized semantic input (camera 2).  Those images retain
the same object pose and silhouette, but have different image resolutions and
object-normalization crops.  This module explicitly estimates that *image
preprocessing* map from foreground supports, then uses it to transfer the
partial pixels onto Pixal-input MoGe pixels.

It intentionally does not search 3-D orientations.  The resulting indexed
correspondences are sufficient to estimate a proper Sim(3) from native MoGe
coordinates to the partial frame.  Pixal-to-native-MoGe is handled separately
by the camera-anchored analytic registration.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 3:
        value = value[..., -1]
    if value.ndim != 2:
        raise ValueError(f"foreground mask must be 2-D, got {value.shape}")
    return value > 0


def foreground_bbox(mask: np.ndarray) -> np.ndarray:
    """Return ``[xmin, ymin, xmax, ymax]`` for a non-empty foreground mask."""
    foreground = _binary_mask(mask)
    y, x = np.where(foreground)
    if len(x) < 8:
        raise ValueError("foreground mask has fewer than eight pixels")
    return np.array((x.min(), y.min(), x.max(), y.max()), dtype=np.float64)


def bbox_affine(source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Estimate the explicit camera-1 to camera-2 crop/resize affine map.

    This is deliberately only an image preprocessing transform: independent
    x/y scale is permitted because the two raster processing paths may resize
    to different integer dimensions.  It is not an anisotropic 3-D alignment.
    """
    source, target = foreground_bbox(source_mask), foreground_bbox(target_mask)
    source_extent = source[2:] - source[:2]
    target_extent = target[2:] - target[:2]
    if np.any(source_extent < 1.) or np.any(target_extent < 1.):
        raise ValueError("foreground support is degenerate")
    scale = target_extent / source_extent
    affine = np.eye(3, dtype=np.float64)
    affine[0, 0], affine[1, 1] = scale
    affine[:2, 2] = target[:2] - scale * source[:2]
    return affine


def apply_image_affine(pixel_xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 2-D affine to ``(x, y)`` image coordinates."""
    pixels = np.asarray(pixel_xy, dtype=np.float64)
    matrix = np.asarray(affine, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixel_xy must be shaped (N, 2)")
    if matrix.shape != (3, 3):
        raise ValueError("affine must be 3x3 homogeneous")
    homogeneous = np.concatenate((pixels, np.ones((len(pixels), 1))), axis=1)
    moved = homogeneous @ matrix.T
    return moved[:, :2] / np.clip(moved[:, 2:3], 1e-12, None)


def partial_uv_to_image_pixels(
    point_uv: np.ndarray,
    image_hw: tuple[int, int],
    *,
    flip_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized saved partial UV into top-left raster coordinates."""
    uv = np.asarray(point_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("point_uv must be shaped (N, 2)")
    height, width = map(int, image_hw)
    valid = np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0.) & (uv[:, 0] <= 1.)
    valid &= (uv[:, 1] >= 0.) & (uv[:, 1] <= 1.)
    pixels = uv.copy()
    if flip_y:
        pixels[:, 1] = 1. - pixels[:, 1]
    pixels[:, 0] *= max(width - 1, 1)
    pixels[:, 1] *= max(height - 1, 1)
    return pixels, valid


def infer_point_uv_flip_y(
    point_uv: np.ndarray,
    source_mask: np.ndarray,
    *,
    dilation_pixels: int = 2,
) -> tuple[bool, dict[str, float]]:
    """Infer whether normalized Camera-1 UV needs a vertical flip.

    Kaolin/DepthPrompting assets use a bottom-left origin, while standard
    pinhole rasters use a top-left origin.  Select the convention whose scan
    samples receive more Camera-1 foreground support.  This relies only on
    saved observation evidence and has no dataset/category-specific branch.
    """
    foreground = _binary_mask(source_mask)
    if int(dilation_pixels) > 0:
        foreground = binary_dilation(
            foreground,
            structure=np.ones((2 * int(dilation_pixels) + 1,) * 2, dtype=bool),
        )
    scores: dict[str, float] = {}
    for name, flip in (("top_left", False), ("bottom_left", True)):
        pixels, valid = partial_uv_to_image_pixels(
            point_uv, foreground.shape, flip_y=flip,
        )
        rounded = np.rint(pixels).astype(np.int64)
        inside = (
            valid
            & (rounded[:, 0] >= 0) & (rounded[:, 0] < foreground.shape[1])
            & (rounded[:, 1] >= 0) & (rounded[:, 1] < foreground.shape[0])
        )
        hits = np.zeros(len(rounded), dtype=bool)
        hits[inside] = foreground[rounded[inside, 1], rounded[inside, 0]]
        scores[name] = float(hits.sum() / max(int(valid.sum()), 1))
    # Preserve the historical convention if a symmetric object causes a tie.
    flip_y = scores["bottom_left"] >= scores["top_left"]
    return bool(flip_y), scores


def transferred_partial_to_moge_matches(
    point_uv: np.ndarray,
    source_image_hw: tuple[int, int],
    camera1_to_camera2: np.ndarray,
    target_moge_pixels: np.ndarray,
    *,
    max_pixel_distance: float,
    flip_y: bool = True,
) -> tuple[np.ndarray, dict]:
    """Create pixel-indexed partial/MoGe pairs through the two-camera map.

    Returns rows compatible with :func:`robust_moge_to_partial`:
    ``[partial_index, moge_index, moge_index, pixel_distance]``.
    """
    moge_pixels = np.asarray(target_moge_pixels, dtype=np.float64)
    if moge_pixels.ndim != 2 or moge_pixels.shape[1] != 2:
        raise ValueError("target_moge_pixels must be shaped (N, 2)")
    pixels, valid = partial_uv_to_image_pixels(point_uv, source_image_hw, flip_y=flip_y)
    mapped = apply_image_affine(pixels, camera1_to_camera2)
    if not valid.any() or len(moge_pixels) == 0:
        return np.empty((0, 4), dtype=np.float64), {
            "valid_partial_pixels": int(valid.sum()), "matched_partial_pixels": 0,
            "match_ratio": 0., "max_pixel_distance": float(max_pixel_distance),
            "mapped_partial_pixels": mapped,
        }
    valid_ids = np.flatnonzero(valid)
    distance, nearest = cKDTree(moge_pixels).query(mapped[valid_ids], k=1)
    keep = distance <= float(max_pixel_distance)
    matched_partial = valid_ids[keep]
    matched_moge = nearest[keep].astype(np.int64)
    matches = np.column_stack((matched_partial, matched_moge, matched_moge, distance[keep]))
    return matches.astype(np.float64), {
        "valid_partial_pixels": int(valid.sum()), "matched_partial_pixels": int(len(matches)),
        "match_ratio": float(len(matches) / max(int(valid.sum()), 1)),
        "max_pixel_distance": float(max_pixel_distance), "mapped_partial_pixels": mapped,
    }


def affine_mask_iou(source_mask: np.ndarray, target_mask: np.ndarray, affine: np.ndarray) -> float:
    """Report support IoU after applying camera-1-to-camera-2 affine mapping."""
    import cv2

    source = _binary_mask(source_mask).astype(np.uint8)
    target = _binary_mask(target_mask)
    warped = cv2.warpAffine(
        source, np.asarray(affine, dtype=np.float32)[:2],
        (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST,
    ).astype(bool)
    return float(np.count_nonzero(warped & target) / max(np.count_nonzero(warped | target), 1))


def conjugate_partial_residual_to_native(
    partial_residual: np.ndarray,
    native_moge_to_partial: np.ndarray,
) -> np.ndarray:
    """Express a small partial-frame correction in native MoGe coordinates.

    If ``T`` maps native MoGe into the partial frame and a residual ``D`` is
    applied after it, the equivalent native correction is ``T^-1 D T``.  This
    lets a saved-view partial correction be checked against Pixal--MoGe's
    native camera evidence before it is accepted.
    """
    from src.bidirectional_cycle_registration import invert_proper_sim3

    residual, bridge = np.asarray(partial_residual, dtype=np.float64), np.asarray(native_moge_to_partial, dtype=np.float64)
    if residual.shape != (4, 4) or bridge.shape != (4, 4):
        raise ValueError("partial residual and native bridge must be 4x4")
    return invert_proper_sim3(bridge) @ residual @ bridge


def conjugate_native_residual_to_partial(
    native_residual: np.ndarray,
    native_moge_to_partial: np.ndarray,
) -> np.ndarray:
    """Express a small native-MoGe correction in the partial coordinate frame."""
    from src.bidirectional_cycle_registration import invert_proper_sim3

    residual, bridge = np.asarray(native_residual, dtype=np.float64), np.asarray(native_moge_to_partial, dtype=np.float64)
    if residual.shape != (4, 4) or bridge.shape != (4, 4):
        raise ValueError("native residual and bridge must be 4x4")
    return bridge @ residual @ invert_proper_sim3(bridge)
