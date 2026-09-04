"""GT-free saved-view surfel posterior for a complete point-cloud prior.

The representation is deliberately discrete: all Gaussian/surfel centres are
existing prior or partial points.  The saved semantic image and partial scan
only assign opacity; the decoder chooses a spatially uniform subset.  It never
regenerates, deforms, or deletes hidden geometry as an optimization variable.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from scripts.run_render_to_moge_sim3 import zbuffer_depth_with_indices


def semantic_target(path, size: int):
    image = Image.open(path).convert("RGB").resize((int(size), int(size)), Image.Resampling.LANCZOS)
    u8 = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    chroma = u8.max(axis=2).astype(np.int16) - u8.min(axis=2).astype(np.int16)
    foreground = (gray < 242) | (chroma > 14)
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    edge = cv2.Canny(foreground.astype(np.uint8) * 255, 40, 120).astype(np.float64) / 255.0
    return foreground.astype(np.float64), edge


def _sample(values: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    h, w = values.shape
    xy = np.rint(pixels).astype(np.int64)
    valid = ((xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h))
    result = np.zeros(len(pixels), dtype=np.float64)
    result[valid] = values[xy[valid, 1], xy[valid, 0]]
    return result


def _visible_ids(projector, points: np.ndarray, size: int) -> np.ndarray:
    pixel, depth = projector.project(points)
    sx = float(size - 1) / max(projector.image_shape[1] - 1, 1)
    sy = float(size - 1) / max(projector.image_shape[0] - 1, 1)
    _z, _mask, indices = zbuffer_depth_with_indices(
        pixel * np.asarray((sx, sy)), depth, (size, size), splat_radius=1
    )
    return np.unique(indices[indices >= 0])


def weighted_voxel_decode(points: np.ndarray, opacity: np.ndarray, *, target_points: int,
                          diagonal: float, binary_steps: int = 18):
    """One representative per voxel, preferring high-confidence surfels."""
    points = np.asarray(points, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    target = int(np.clip(target_points, 1024, len(points)))
    origin = points.min(axis=0)

    def representatives(size: float):
        keys = np.floor((points - origin) / max(float(size), diagonal * 1e-8)).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        groups = int(inverse.max()) + 1
        centroid = np.zeros((groups, 3), dtype=np.float64)
        np.add.at(centroid, inverse, points)
        centroid /= np.bincount(inverse, minlength=groups)[:, None]
        distance2 = np.sum((points - centroid[inverse]) ** 2, axis=1)
        order = np.lexsort((np.arange(len(points)), distance2, -opacity, inverse))
        ordered_groups = inverse[order]
        return order[np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]]

    low, high = diagonal * 1e-6, diagonal
    best = np.arange(len(points), dtype=np.int64)
    best_size, best_gap = low, abs(len(best) - target)
    for _ in range(int(binary_steps)):
        size = .5 * (low + high)
        ids = representatives(size)
        gap = abs(len(ids) - target)
        if gap < best_gap:
            best, best_size, best_gap = ids, size, gap
        if len(ids) > target:
            low = size
        else:
            high = size
    if len(best) > target:
        best = best[np.linspace(0, len(best) - 1, target, dtype=np.int64)]
    return best, {"decoded_points": int(len(best)), "target_points": int(target),
                  "voxel_size": float(best_size), "gap": int(best_gap)}


def coverage(source: np.ndarray, decoded: np.ndarray, diagonal: float):
    distance = cKDTree(decoded).query(source, k=1, workers=-1)[0] / max(float(diagonal), 1e-12)
    return {"q95_ratio": float(np.quantile(distance, .95)),
            "q99_ratio": float(np.quantile(distance, .99))}


def semantic_projection(projector, points: np.ndarray, target_mask: np.ndarray):
    from scripts.run_pixal_pca_sim3_ttt_v2 import mask_metrics, render_mask
    return mask_metrics(target_mask.astype(bool), render_mask(projector, points, target_mask.shape[0], splat=1))


def build_view_conditioned_surfel_field(body: np.ndarray, partial: np.ndarray, projector,
                                        semantic_mask: np.ndarray, semantic_edge: np.ndarray,
                                        diagonal: float, support_ratio: float):
    centres = np.concatenate((body, partial), axis=0)
    is_partial = np.r_[np.zeros(len(body), dtype=bool), np.ones(len(partial), dtype=bool)]
    pixels, depth = projector.project(centres)
    mask = _sample(semantic_mask, pixels)
    edge = _sample(semantic_edge, pixels)
    body_distance = cKDTree(partial).query(body, k=1, workers=-1)[0]
    partial_distance = cKDTree(body).query(partial, k=1, workers=-1)[0]
    support_distance = np.concatenate((body_distance, partial_distance))
    support = np.exp(-support_distance / max(float(support_ratio) * diagonal, 1e-12))
    visible = np.zeros(len(centres), dtype=np.float64)
    visible[_visible_ids(projector, body, semantic_mask.shape[0])] = 1.0
    opacity = (.05 + .38 * mask + .12 * edge + .25 * support + .15 * visible
               + .25 * is_partial.astype(np.float64))
    opacity *= np.where(np.isfinite(depth) & (depth > 1e-8), 1.0, .05)
    opacity = np.clip(opacity, .01, 1.0)
    return centres, opacity, {
        "mean_opacity": float(opacity.mean()),
        "mean_prior_opacity": float(opacity[~is_partial].mean()),
        "mean_partial_opacity": float(opacity[is_partial].mean()),
        "visible_prior_centres": int(visible[:len(body)].sum()),
        "surface_supported_prior_fraction": float((support[:len(body)] > .5).mean()),
        "partial_centres": int(is_partial.sum()),
    }
