"""Compact saved-view depth editing for partial-anchored 3D Gaussian fields.

The registered complete prior provides the global object shape.  A partial
scan is only authoritative where it is visible in the saved camera.  This
module therefore estimates a small, image-grid depth residual from the two
z-buffers and moves only the corresponding *editable* Gaussian means along
camera rays.  Locked partial anchors and prior Gaussians outside that visible
support receive exactly zero displacement.

This deliberately replaces a point-level deformation graph: there is one
global Sim(3) before this action, one low-dimensional residual field here, and
one common no-GT gate after it.
"""

from __future__ import annotations

import numpy as np
import torch

from src.ray_consistent_registration import zbuffer_indices


def camera_depth_direction(projector) -> np.ndarray:
    """Return the world displacement that increases saved-camera depth by one.

    The saved camera exposes only a point-transform API.  Finite probing
    recovers its affine linear part without depending on private camera
    attributes or a particular camera implementation.
    """
    origin = torch.zeros((1, 3), dtype=torch.float32, device=projector.device)
    basis = torch.eye(3, dtype=torch.float32, device=projector.device)
    with torch.no_grad():
        camera_origin = projector.camera.transform(origin).detach().cpu().numpy()[0]
        camera_basis = projector.camera.transform(basis).detach().cpu().numpy() - camera_origin
    # Points are row vectors: camera_delta = world_delta @ camera_basis.
    direction = np.linalg.solve(camera_basis.T, np.array((0., 0., 1.)))
    if not np.all(np.isfinite(direction)):
        raise ValueError("saved camera has no finite depth direction")
    return direction


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a deterministic weighted median without duplicating samples."""
    order = np.argsort(values, kind="stable")
    values, weights = values[order], np.clip(weights[order], 0., None)
    if float(weights.sum()) <= 1e-12:
        return float(np.median(values))
    return float(values[np.searchsorted(np.cumsum(weights), .5 * weights.sum(), side="left")])


def _median_grid(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, *, grid_size: int,
                 image_shape: tuple[int, int], min_samples: int,
                 weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Robustly pool residuals into a fixed image grid."""
    height, width = map(int, image_shape)
    grid_y = np.clip((rows * grid_size / max(height, 1)).astype(np.int64), 0, grid_size - 1)
    grid_x = np.clip((cols * grid_size / max(width, 1)).astype(np.int64), 0, grid_size - 1)
    field = np.zeros((grid_size, grid_size), dtype=np.float64)
    active = np.zeros((grid_size, grid_size), dtype=bool)
    for y in range(grid_size):
        for x in range(grid_size):
            selected = (grid_y == y) & (grid_x == x)
            if int(selected.sum()) >= int(min_samples):
                local_weights = np.ones(int(selected.sum()), dtype=np.float64) if weights is None else weights[selected]
                field[y, x] = _weighted_median(values[selected], local_weights)
                active[y, x] = True
    return field, active


def _edge_aware_grid_smooth(field: np.ndarray, active: np.ndarray, depth: np.ndarray, *,
                            iterations: int, depth_threshold: float) -> np.ndarray:
    """Smooth only across compatible prior-depth cells, preserving boundaries."""
    result = field.copy()
    cell_depth = np.zeros_like(field)
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            values = depth[y::field.shape[0], x::field.shape[1]]
            finite = values[np.isfinite(values)]
            cell_depth[y, x] = np.median(finite) if len(finite) else np.nan
    for _ in range(int(iterations)):
        updated = result.copy()
        for y, x in zip(*np.where(active)):
            values = [result[y, x]]
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < field.shape[0] and 0 <= xx < field.shape[1] and active[yy, xx]):
                    continue
                if (np.isfinite(cell_depth[y, x]) and np.isfinite(cell_depth[yy, xx])
                        and abs(cell_depth[y, x] - cell_depth[yy, xx]) <= depth_threshold):
                    values.append(result[yy, xx])
            updated[y, x] = float(np.mean(values))
        result = updated
    return result


def saved_view_ray_edit(
    partial: np.ndarray,
    prior_means: np.ndarray,
    projector,
    *,
    soft_points: np.ndarray | None = None,
    soft_confidence: np.ndarray | None = None,
    soft_weight: float = .25,
    grid_size: int = 32,
    min_cell_samples: int = 2,
    smoothing_iterations: int = 2,
    max_depth_ratio: float = .035,
    diagonal: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Estimate a bounded editable-prior depth correction from one saved view.

    The function operates on the editable Gaussian means only.  It makes no
    assumptions about object category, mesh topology, texture, GT, or a
    particular 3DGS trainer.
    """
    partial = np.asarray(partial, dtype=np.float64)
    prior_means = np.asarray(prior_means, dtype=np.float64)
    if partial.ndim != 2 or prior_means.ndim != 2 or partial.shape[1] != 3 or prior_means.shape[1] != 3:
        raise ValueError("partial and prior_means must be (N, 3)")
    if diagonal is None:
        diagonal = float(np.linalg.norm(np.ptp(partial, axis=0)))
    diagonal = max(float(diagonal), 1e-8)
    partial_uv, partial_depth = projector.project(partial)
    prior_uv, prior_depth = projector.project(prior_means)
    partial_z, partial_mask, _ = zbuffer_indices(partial_uv, partial_depth, projector.image_shape)
    prior_z, prior_mask, prior_index = zbuffer_indices(prior_uv, prior_depth, projector.image_shape)
    soft_z = np.full(projector.image_shape, np.inf, dtype=np.float64)
    soft_mask = np.zeros(projector.image_shape, dtype=bool)
    soft_index = np.full(projector.image_shape, -1, dtype=np.int64)
    if soft_points is not None:
        soft_points = np.asarray(soft_points, dtype=np.float64)
        if soft_points.ndim != 2 or soft_points.shape[1] != 3:
            raise ValueError("soft_points must have shape (N, 3)")
        if soft_confidence is None:
            soft_confidence = np.ones(len(soft_points), dtype=np.float64)
        soft_confidence = np.asarray(soft_confidence, dtype=np.float64)
        if soft_confidence.shape != (len(soft_points),):
            raise ValueError("soft_confidence must align with soft_points")
        soft_uv, soft_depth = projector.project(soft_points)
        soft_z, soft_mask, soft_index = zbuffer_indices(soft_uv, soft_depth, projector.image_shape)
    # Hard scan depth always wins.  Soft MoGe may only densify empty pixels in
    # the already scan-supported image region passed by its own gate.
    target_mask = partial_mask | (~partial_mask & soft_mask)
    target_z = np.where(partial_mask, partial_z, soft_z)
    shared = target_mask & prior_mask & np.isfinite(target_z) & np.isfinite(prior_z)
    rows, cols = np.where(shared)
    if len(rows) < max(16, int(min_cell_samples)):
        return prior_means.copy(), {"active": False, "reason": "insufficient_shared_saved_view_support",
                                    "shared_pixels": int(len(rows)), "edited_gaussians": 0}
    direction = camera_depth_direction(projector)
    # Saved-camera depth can be normalized/scaled.  Convert the shared metric
    # trust region into its camera-depth equivalent before clipping residuals;
    # otherwise a numerically small rendered-depth delta can become a large
    # world-space Gaussian jump.
    metric_cap = float(max_depth_ratio) * diagonal
    camera_cap = metric_cap / max(float(np.linalg.norm(direction)), 1e-12)
    raw_residual = target_z[rows, cols] - prior_z[rows, cols]
    raw_residual = np.clip(raw_residual, -camera_cap, camera_cap)
    hard_rows = partial_mask[rows, cols]
    sample_weight = np.ones(len(rows), dtype=np.float64)
    if soft_points is not None:
        soft_rows = ~hard_rows
        sample_weight[soft_rows] = float(soft_weight) * np.clip(
            soft_confidence[soft_index[rows[soft_rows], cols[soft_rows]]], 0., 1.)
    field, active = _median_grid(raw_residual, rows, cols, grid_size=int(grid_size),
                                 image_shape=projector.image_shape, min_samples=int(min_cell_samples),
                                 weights=sample_weight)
    field = _edge_aware_grid_smooth(field, active, prior_z, iterations=int(smoothing_iterations),
                                    depth_threshold=max(.015 * diagonal, 1e-8))
    height, width = projector.image_shape
    cell_y = np.clip((rows * grid_size / max(height, 1)).astype(np.int64), 0, grid_size - 1)
    cell_x = np.clip((cols * grid_size / max(width, 1)).astype(np.int64), 0, grid_size - 1)
    point_ids = prior_index[rows, cols]
    residual = field[cell_y, cell_x]
    # One editable Gaussian may represent adjacent saved-view pixels.  Use a
    # median so splat density cannot overweight it.
    unique, inverse = np.unique(point_ids, return_inverse=True)
    pooled = np.zeros(len(unique), dtype=np.float64)
    for idx in range(len(unique)):
        pooled[idx] = _weighted_median(residual[inverse == idx], sample_weight[inverse == idx])
    moved = prior_means.copy()
    moved[unique] += pooled[:, None] * direction[None]
    metric_displacement = np.abs(pooled) * float(np.linalg.norm(direction))
    return moved, {
        "active": True, "shared_pixels": int(len(rows)), "hard_shared_pixels": int(hard_rows.sum()),
        "soft_shared_pixels": int((~hard_rows).sum()), "soft_weight": float(soft_weight) if soft_points is not None else 0.,
        "active_grid_cells": int(active.sum()),
        "edited_gaussians": int(len(unique)), "grid_size": int(grid_size),
        "max_metric_displacement": float(metric_displacement.max(initial=0.)),
        "mean_metric_displacement": float(metric_displacement.mean()) if len(pooled) else 0.,
        "max_metric_cap": metric_cap, "max_camera_depth_cap": camera_cap,
        "camera_depth_direction": direction,
    }


def interpolate_ray_edit(prior_means: np.ndarray, corrected_means: np.ndarray, fraction: float) -> np.ndarray:
    """Form a conservative agent candidate without changing edit support."""
    fraction = float(fraction)
    if not 0. <= fraction <= 1.:
        raise ValueError("fraction must lie in [0, 1]")
    prior_means, corrected_means = np.asarray(prior_means), np.asarray(corrected_means)
    if prior_means.shape != corrected_means.shape:
        raise ValueError("prior_means and corrected_means must have matching shapes")
    return prior_means + fraction * (corrected_means - prior_means)
