"""Small deterministic z-buffer utilities shared by the mainline."""

from __future__ import annotations

import numpy as np


def frontmost_pixel_indices(uv, depth, image_shape, *, splat_radius: int = 0,
                            circular_splat: bool = False):
    """Vectorised deterministic front-most rasterisation.

    Candidate splats are sorted by pixel, depth, and the exact historical
    loop order.  The final key retains the old strict-``<`` tie behaviour,
    while eliminating Python-level work over every point/pixel pair.
    """
    height, width = (int(image_shape[0]), int(image_shape[1]))
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    rendered = np.full((height, width), np.inf, dtype=np.float64)
    indices = np.full((height, width), -1, dtype=np.int64)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
    if not valid.any():
        return rendered, np.zeros((height, width), dtype=bool), indices

    source_ids = np.flatnonzero(valid)
    xy = np.rint(uv[valid]).astype(np.int64)
    values = depth[valid]
    radius = max(0, int(splat_radius))
    offsets = np.asarray([
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not circular_splat or dx * dx + dy * dy <= radius * radius
    ], dtype=np.int64)
    xx = xy[:, None, 0] + offsets[None, :, 0]
    yy = xy[:, None, 1] + offsets[None, :, 1]
    in_bounds = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    if not in_bounds.any():
        return rendered, np.zeros((height, width), dtype=bool), indices

    flat = (yy * width + xx)[in_bounds]
    candidate_depth = np.broadcast_to(values[:, None], xx.shape)[in_bounds]
    candidate_source = np.broadcast_to(source_ids[:, None], xx.shape)[in_bounds]
    # The previous nested loops visit every offset before starting the next;
    # retain this as a deterministic depth-tie rule.
    offset_order = np.broadcast_to(np.arange(len(offsets), dtype=np.int64), xx.shape)[in_bounds]
    order = np.lexsort((candidate_source, offset_order, candidate_depth, flat))
    sorted_flat = flat[order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    winners = order[first]
    winner_flat = flat[winners]
    rendered.reshape(-1)[winner_flat] = candidate_depth[winners]
    indices.reshape(-1)[winner_flat] = candidate_source[winners]
    return rendered, np.isfinite(rendered), indices


def zbuffer_depth_with_indices(uv, depth, image_shape, splat_radius: int = 1):
    """Render front-most depth and its source index at each image pixel."""
    rendered, mask, indices = frontmost_pixel_indices(
        uv, depth, image_shape, splat_radius=splat_radius, circular_splat=True,
    )
    rendered[~mask] = 0.0
    return rendered, mask, indices
