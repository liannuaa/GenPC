"""Camera-locked 2-D control image from a compact partial/prior residual.

Text alone can be under-expressed by an image editor when the input is already
photorealistic. This module exposes the same measured residual as a small
image-space warp: only a compact visible support receives motion, while the
complete-prior image remains unchanged elsewhere. It never uses labels, ground
truth, or a learned optical-flow model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.hierarchical_residual_registration import visible_residual_components


@dataclass(frozen=True)
class ResidualWarpEvidence:
    eligible: bool
    actionable: bool
    component_fraction: float
    controls: int
    median_pixel_displacement: float
    max_pixel_displacement: float
    gain: float
    support_radius_pixels: float

    def to_dict(self) -> dict:
        return asdict(self)


def _select_controls(source_uv: np.ndarray, displacement: np.ndarray,
                     maximum: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically thin many projected point correspondences."""
    if len(source_uv) <= maximum:
        return source_uv, displacement
    order = np.lexsort((source_uv[:, 1], source_uv[:, 0]))
    ids = order[np.linspace(0, len(order) - 1, int(maximum), dtype=np.int64)]
    return source_uv[ids], displacement[ids]


def warp_residual_support(image: np.ndarray, partial: np.ndarray, prior: np.ndarray,
                          projector, *, diagonal: float, gain: float,
                          maximum_controls: int = 192,
                          support_radius_pixels: float = 48.) -> tuple[np.ndarray, ResidualWarpEvidence]:
    """Move the top connected saved-view residual support in a source image."""
    if not 0. < float(gain) <= 1.:
        raise ValueError("gain must lie in (0, 1]")
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected RGB image")
    components, _ = visible_residual_components(
        partial, prior, projector, diagonal=diagonal, pixel_radius=8.,
        residual_quantile=.70, residual_min_ratio=.018, residual_max_ratio=.14,
        min_points=48, max_components=8)
    if not components:
        return rgb.copy(), ResidualWarpEvidence(False, False, 0., 0, 0., 0., float(gain),
                                                 float(support_radius_pixels))
    (source, target), info = components[0]
    source_uv, _ = projector.project(source)
    target_uv, _ = projector.project(target)
    height, width = rgb.shape[:2]
    valid = (np.isfinite(source_uv).all(axis=1) & np.isfinite(target_uv).all(axis=1)
             & (source_uv[:, 0] >= 0) & (source_uv[:, 0] < width)
             & (source_uv[:, 1] >= 0) & (source_uv[:, 1] < height)
             & (target_uv[:, 0] >= 0) & (target_uv[:, 0] < width)
             & (target_uv[:, 1] >= 0) & (target_uv[:, 1] < height))
    source_uv, target_uv = source_uv[valid], target_uv[valid]
    if len(source_uv) < 8:
        return rgb.copy(), ResidualWarpEvidence(False, False, float(info["component_fraction"]),
                                                 int(len(source_uv)), 0., 0., float(gain),
                                                 float(support_radius_pixels))
    displacement = (target_uv - source_uv) * float(gain)
    source_uv, displacement = _select_controls(source_uv, displacement, maximum_controls)
    pad = int(np.ceil(float(support_radius_pixels)))
    lo = np.maximum(np.floor(source_uv.min(axis=0) - pad).astype(int), 0)
    hi = np.minimum(np.ceil(source_uv.max(axis=0) + pad).astype(int),
                    np.asarray((width - 1, height - 1)))
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    query = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    count = min(12, len(source_uv))
    distance, ids = cKDTree(source_uv).query(query, k=count)
    if count == 1:
        distance, ids = distance[:, None], ids[:, None]
    radius = max(float(support_radius_pixels), 1.)
    weights = np.exp(-.5 * (distance / radius) ** 2)
    weights[distance > radius] = 0.
    total = weights.sum(axis=1, keepdims=True)
    local = np.zeros((len(query), 2), dtype=np.float64)
    nonzero = total[:, 0] > 1e-12
    local[nonzero] = (weights[nonzero, :, None] * displacement[ids[nonzero]]).sum(axis=1) / total[nonzero]
    field = np.zeros((height, width, 2), dtype=np.float32)
    field[yy, xx] = local.reshape(yy.shape + (2,)).astype(np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    warped = cv2.remap(rgb, grid_x - field[..., 0], grid_y - field[..., 1],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lengths = np.linalg.norm(displacement, axis=1)
    median = float(np.median(lengths))
    return warped, ResidualWarpEvidence(
        True, bool(median >= 2.), float(info["component_fraction"]), int(len(source_uv)),
        median, float(np.max(lengths)), float(gain), float(support_radius_pixels))
