"""Positive-only virtual-view correspondence for registered point populations.

A partial scan remains geometrically informative after a viewpoint change:
its visible subset is incomplete, but every rendered partial/Pixal overlap is
still valid positive evidence. This module creates no loss for unoccupied
partial pixels; it returns only mutual z-buffered overlap pairs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(points - centre, full_matrices=False)
    return centre, right.T


def _zbuffer_ids(coordinates: np.ndarray, *, depth_axis: int, sign: int,
                 lower: np.ndarray, extent: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    planar_axes = [axis for axis in range(3) if axis != depth_axis]
    plane = coordinates[:, planar_axes]
    uv = np.floor((plane - lower) / extent * (int(resolution) - 1)).astype(np.int64)
    uv = np.clip(uv, 0, int(resolution) - 1)
    key = uv[:, 0] * int(resolution) + uv[:, 1]
    depth = int(sign) * coordinates[:, depth_axis]
    order = np.lexsort((np.arange(len(coordinates)), depth, key))
    sorted_key = key[order]
    first = np.concatenate(([True], sorted_key[1:] != sorted_key[:-1]))
    ids = order[first]
    return ids.astype(np.int64), uv[ids].astype(np.float64)


def virtual_view_positive_pairs(
    partial: np.ndarray,
    prior: np.ndarray,
    *,
    frame_points: np.ndarray | None = None,
    views: int = 6,
    resolution: int = 384,
    max_pixel_distance: float = 2.,
) -> tuple[np.ndarray, dict]:
    """Return mutual overlap pairs from deterministic signed-PCA virtual views.

    Each row follows ``[partial_id, prior_id, prior_id, pixel_distance]``. The
    same registered frame is used for both populations. Missing partial pixels
    remain unconstrained rather than becoming empty-space supervision.
    """
    partial = np.asarray(partial, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if partial.ndim != 2 or prior.ndim != 2 or partial.shape[1] != 3 or prior.shape[1] != 3:
        raise ValueError("partial and prior must be (N, 3)")
    if min(len(partial), len(prior)) == 0:
        raise ValueError("partial and prior must be nonempty")
    if not (0 <= int(views) <= 6) or int(resolution) < 8 or float(max_pixel_distance) < 0.:
        raise ValueError("views must lie in [0, 6], resolution must be >= 8, and pixel distance nonnegative")
    if views == 0:
        return np.empty((0, 4), dtype=np.float64), {
            "virtual_views": 0, "virtual_mutual_pairs": 0, "per_view": [],
        }
    frame_points = prior if frame_points is None else np.asarray(frame_points, dtype=np.float64)
    if frame_points.ndim != 2 or frame_points.shape[1] != 3 or len(frame_points) < 3:
        raise ValueError("frame_points must be a (N, 3) array with at least three points")
    centre, basis = _pca_frame(frame_points)
    partial_coordinates = (partial - centre) @ basis
    prior_coordinates = (prior - centre) @ basis
    combined = np.concatenate((partial_coordinates, prior_coordinates), axis=0)
    rows: list[np.ndarray] = []
    per_view: list[dict] = []
    directions = [(axis, sign) for axis in range(3) for sign in (-1, 1)][:int(views)]
    for view_id, (depth_axis, sign) in enumerate(directions):
        planar_axes = [axis for axis in range(3) if axis != depth_axis]
        plane = combined[:, planar_axes]
        lower = plane.min(axis=0)
        extent = np.maximum(plane.max(axis=0) - lower, 1e-9)
        partial_ids, partial_uv = _zbuffer_ids(
            partial_coordinates, depth_axis=depth_axis, sign=sign,
            lower=lower, extent=extent, resolution=int(resolution),
        )
        prior_ids, prior_uv = _zbuffer_ids(
            prior_coordinates, depth_axis=depth_axis, sign=sign,
            lower=lower, extent=extent, resolution=int(resolution),
        )
        partial_tree, prior_tree = cKDTree(partial_uv), cKDTree(prior_uv)
        forward_distance, forward_local = prior_tree.query(partial_uv, k=1, workers=-1)
        _, reverse_local = partial_tree.query(prior_uv, k=1, workers=-1)
        candidate = np.arange(len(partial_ids))
        reciprocal = reverse_local[forward_local] == candidate
        keep = reciprocal & (forward_distance <= float(max_pixel_distance))
        if np.any(keep):
            selected_partial = partial_ids[candidate[keep]]
            selected_prior = prior_ids[forward_local[keep]]
            rows.append(np.column_stack((selected_partial, selected_prior, selected_prior,
                                         forward_distance[keep])))
        per_view.append({
            "view": int(view_id), "pca_depth_axis": int(depth_axis), "depth_sign": int(sign),
            "partial_visible_points": int(len(partial_ids)), "prior_visible_points": int(len(prior_ids)),
            "mutual_pairs": int(keep.sum()),
        })
    pairs = np.concatenate(rows, axis=0) if rows else np.empty((0, 4), dtype=np.float64)
    return pairs.astype(np.float64), {
        "virtual_views": int(views), "virtual_resolution": int(resolution),
        "virtual_max_pixel_distance": float(max_pixel_distance),
        "virtual_mutual_pairs": int(len(pairs)), "per_view": per_view,
        "evidence_policy": "positive_overlap_only; missing_partial_pixels_are_unconstrained",
    }


def concatenate_positive_pairs(saved_pairs: np.ndarray, virtual_pairs: np.ndarray) -> np.ndarray:
    """Join saved-camera and virtual positive pairs in one standard layout."""
    saved_pairs = np.asarray(saved_pairs, dtype=np.float64)
    virtual_pairs = np.asarray(virtual_pairs, dtype=np.float64)
    if saved_pairs.ndim != 2 or virtual_pairs.ndim != 2:
        raise ValueError("pair arrays must be two-dimensional")
    if saved_pairs.shape[1] < 4 or virtual_pairs.shape[1] < 4:
        raise ValueError("pair arrays must have at least four columns")
    return np.concatenate((saved_pairs[:, :4], virtual_pairs[:, :4]), axis=0)


def save_virtual_overlay_board(
    path,
    partial: np.ndarray,
    prior: np.ndarray,
    *,
    frame_points: np.ndarray | None = None,
    views: int = 6,
    resolution: int = 256,
) -> None:
    """Write a compact partial-gray/Pixal-red board for virtual-view review."""
    partial = np.asarray(partial, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    frame_points = prior if frame_points is None else np.asarray(frame_points, dtype=np.float64)
    if frame_points.ndim != 2 or frame_points.shape[1] != 3 or len(frame_points) < 3:
        raise ValueError("frame_points must be a (N, 3) array with at least three points")
    centre, basis = _pca_frame(frame_points)
    partial_coordinates = (partial - centre) @ basis
    prior_coordinates = (prior - centre) @ basis
    combined = np.concatenate((partial_coordinates, prior_coordinates), axis=0)
    columns, rows = 3, max(1, int(np.ceil(int(views) / 3)))
    board = Image.new("RGB", (columns * int(resolution), rows * int(resolution)), (28, 28, 28))
    directions = [(axis, sign) for axis in range(3) for sign in (-1, 1)][:int(views)]
    for view_id, (depth_axis, sign) in enumerate(directions):
        planar_axes = [axis for axis in range(3) if axis != depth_axis]
        plane = combined[:, planar_axes]
        lower = plane.min(axis=0)
        extent = np.maximum(plane.max(axis=0) - lower, 1e-9)
        partial_ids, partial_uv = _zbuffer_ids(
            partial_coordinates, depth_axis=depth_axis, sign=sign,
            lower=lower, extent=extent, resolution=int(resolution),
        )
        prior_ids, prior_uv = _zbuffer_ids(
            prior_coordinates, depth_axis=depth_axis, sign=sign,
            lower=lower, extent=extent, resolution=int(resolution),
        )
        del partial_ids, prior_ids
        panel = Image.new("RGB", (int(resolution), int(resolution)), (28, 28, 28))
        draw = ImageDraw.Draw(panel)
        for x, y in prior_uv:
            draw.point((int(x), int(y)), fill=(218, 42, 42))
        for x, y in partial_uv:
            draw.point((int(x), int(y)), fill=(185, 185, 185))
        draw.text((7, 7), f"PCA{depth_axis} {'+' if sign > 0 else '-'}", fill=(238, 238, 238))
        board.paste(panel, ((view_id % columns) * int(resolution), (view_id // columns) * int(resolution)))
    Image.Image.save(board, path)
