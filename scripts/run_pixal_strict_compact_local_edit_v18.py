#!/usr/bin/env python3
"""Strict point-level compact-support diagnostic built on residual v17."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import scripts.run_pixal_partial_core_local_edit_v16 as v16
import scripts.run_pixal_residual_local_edit_v17 as v17


OUTPUT_ROOT = (
    v17.ROOT / "gpt_version/_pixal_strict_compact_local_edit_v18_20260822")
_deform_v16 = v16.deform_points
_optimize_v16 = v16.optimize_local_edit
_handle_points = None
_partial_diagonal = 1.0


def compact_deform_points(points, graph, translations, point_nodes=None,
                          point_weights=None):
    """Force displacement to zero beyond 7% of partial-shape diagonal."""
    points = np.asarray(points, dtype=np.float64)
    raw = _deform_v16(points, graph, translations, point_nodes, point_weights)
    if _handle_points is None or not len(_handle_points):
        return points.copy()
    distance = cKDTree(_handle_points).query(points, k=1)[0] / _partial_diagonal
    inner, outer = .025, .070
    u = np.clip((outer - distance) / (outer - inner), 0., 1.)
    weight = u * u * (3. - 2. * u)
    return points + weight[:, None] * (raw - points)


def optimize_strict_compact(generated, partial, partial_normals, projector, args):
    global _handle_points, _partial_diagonal
    _partial_diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    generated_normals = v16.estimate_normals(
        generated, radius=.025 * _partial_diagonal)
    _, partial_ids, generated_ids = v16.evaluate_raw_partial_gate(
        partial, generated, projector, bbox_diagonal=_partial_diagonal,
        partial_normals=partial_normals, complete_normals=generated_normals,
        distance_ratio=args.correspondence_distance_ratio)
    source_ids, _ = v16.aggregate_pairs(
        partial, generated, partial_ids, generated_ids)
    _handle_points = generated[source_ids].copy()
    return _optimize_v16(
        generated, partial, partial_normals, projector, args)


def main():
    v17.OUTPUT_ROOT = OUTPUT_ROOT
    v16.deform_points = compact_deform_points
    v16.optimize_local_edit = optimize_strict_compact
    v17.main()


if __name__ == "__main__":
    main()
