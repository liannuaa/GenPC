#!/usr/bin/env python3
"""Add GT-free same-ray visible-depth evidence to all-orientation Sim(3) TTT.

The v2 implementation remains an explicit silhouette/surface-only ablation.
This thin layer changes its shared candidate objective before any hypothesis is
optimized or selected, and writes to a separate derivative root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scripts.run_pixal_pca_sim3_ttt_v2 as base


ORIGINAL_EVALUATE = base.evaluate
DEPTH_CACHE = {}


def render_depth_mask(projector, points, size, splat=1):
    pixel, depth = projector.project(points)
    height, width = projector.image_shape
    pixel[:, 0] *= (size - 1) / max(width - 1, 1)
    pixel[:, 1] *= (size - 1) / max(height - 1, 1)
    rendered_depth, mask, _ = base.zbuffer_depth_with_indices(
        pixel, depth, (size, size), splat_radius=splat)
    return rendered_depth, mask


def target_depth_data(projector, partial, render_size):
    key = (id(projector), id(partial), int(render_size))
    if key not in DEPTH_CACHE:
        depth, mask = render_depth_mask(projector, partial, render_size)
        values = depth[mask]
        span = max(float(np.quantile(values, .99) - np.quantile(values, .01)), 1e-8)
        DEPTH_CACHE[key] = depth, mask, span
    return DEPTH_CACHE[key]


def depth_aware_evaluate(source, partial, rotation, scale, translation, projector,
                         target_mask, diagonal, render_size):
    result = ORIGINAL_EVALUATE(
        source, partial, rotation, scale, translation, projector,
        target_mask, diagonal, render_size)
    target_depth, target_depth_mask, depth_span = target_depth_data(
        projector, partial, render_size)
    moved = base.apply_sim3(source, rotation, scale, translation)
    predicted_depth, predicted_mask = render_depth_mask(projector, moved, render_size)
    overlap = target_depth_mask & predicted_mask
    if overlap.any():
        error = np.abs(predicted_depth[overlap] - target_depth[overlap])
        q90 = float(np.quantile(error, .90))
        trimmed = error[error <= q90]
        mean = float(trimmed.mean()) if len(trimmed) else float("inf")
    else:
        mean = q90 = float("inf")
    normalized = mean / depth_span
    result.update({
        "visible_depth_trim90_mean": mean,
        "visible_depth_q90": q90,
        "visible_depth_robust_span": depth_span,
        "normalized_visible_depth": normalized,
        "score_without_depth": result["score"],
    })
    result["score"] = float(result["score"] - .20 * normalized)
    return result


if __name__ == "__main__":
    base.evaluate = depth_aware_evaluate
    base.OUTPUT_ROOT = (
        Path(base.ROOT) / "gpt_version/_pixal_pca_depth_sim3_ttt_v3_20260822"
    )
    base.main()
