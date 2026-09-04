"""Small deterministic z-buffer utilities shared by the mainline."""

from __future__ import annotations

import numpy as np


def zbuffer_depth_with_indices(uv, depth, image_shape, splat_radius: int = 1):
    """Render front-most depth and its source index at each image pixel."""
    height, width = (int(image_shape[0]), int(image_shape[1]))
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    rendered = np.full((height, width), np.inf, dtype=np.float64)
    indices = np.full((height, width), -1, dtype=np.int64)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
    if not valid.any():
        return rendered, np.zeros((height, width), dtype=bool), indices

    source_indices = np.where(valid)[0]
    xy = np.rint(uv[valid]).astype(np.int64)
    z = depth[valid]
    radius = max(0, int(splat_radius))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            xx, yy = xy[:, 0] + dx, xy[:, 1] + dy
            keep = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            for source_index, x, y, value in zip(
                source_indices[keep], xx[keep], yy[keep], z[keep], strict=False,
            ):
                if value < rendered[y, x]:
                    rendered[y, x], indices[y, x] = value, source_index
    mask = np.isfinite(rendered)
    rendered[~mask] = 0.0
    return rendered, mask, indices
