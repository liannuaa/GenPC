"""Compact saved-camera ray correspondence primitives for Agent GenPC+."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def zbuffer_indices(pixel_xy, depth, image_shape, splat_radius=0):
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    height, width = map(int, image_shape)
    zbuffer = np.full((height, width), np.inf, dtype=np.float64)
    indices = np.full((height, width), -1, dtype=np.int64)
    rounded = np.rint(pixel_xy).astype(np.int64)
    finite = np.isfinite(pixel_xy).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
    for source_id in np.flatnonzero(finite):
        x, y = rounded[source_id]
        for offset_y in range(-int(splat_radius), int(splat_radius) + 1):
            yy = y + offset_y
            if yy < 0 or yy >= height:
                continue
            for offset_x in range(-int(splat_radius), int(splat_radius) + 1):
                xx = x + offset_x
                if 0 <= xx < width and depth[source_id] < zbuffer[yy, xx]:
                    zbuffer[yy, xx] = depth[source_id]
                    indices[yy, xx] = source_id
    return zbuffer, indices >= 0, indices


def _grid_coverage(pixel_xy, matched, image_shape, grid_size=8):
    height, width = map(int, image_shape)
    if len(pixel_xy) == 0:
        return 0.0
    gx = np.clip((pixel_xy[:, 0] * grid_size / max(width, 1)).astype(int), 0, grid_size - 1)
    gy = np.clip((pixel_xy[:, 1] * grid_size / max(height, 1)).astype(int), 0, grid_size - 1)
    occupied = set(zip(gx.tolist(), gy.tolist()))
    covered = set(zip(gx[matched].tolist(), gy[matched].tolist()))
    return float(len(covered) / max(len(occupied), 1))


def soft_ray_correspondences(partial_points, generated_points, projector, *,
                             pixel_radius=10.0, splat_radius=1,
                             trim_quantile=.70, max_distance_ratio=.20,
                             bbox_diagonal=None):
    """Match z-buffer-visible points by saved-image rays and robust 3-D distance."""
    partial_points = np.asarray(partial_points, dtype=np.float64)
    generated_points = np.asarray(generated_points, dtype=np.float64)
    partial_uv, partial_depth = projector.project(partial_points)
    generated_uv, generated_depth = projector.project(generated_points)
    _, partial_mask, partial_index = zbuffer_indices(partial_uv, partial_depth, projector.image_shape)
    _, generated_mask, generated_index = zbuffer_indices(
        generated_uv, generated_depth, projector.image_shape, splat_radius=splat_radius)
    py, px = np.where(partial_mask)
    gy, gx = np.where(generated_mask)
    if not len(px) or not len(gx):
        return {"partial_ids": np.empty(0, dtype=np.int64),
                "generated_ids": np.empty(0, dtype=np.int64),
                "pixel_coverage": 0., "grid_coverage": 0., "matched_coverage": 0.,
                "distance_mean": float("inf"), "distance_p95": float("inf"),
                "depth_mean": float("inf"), "objective": float("inf"),
                "visible_partial_count": int(len(px)), "matched_count": 0}
    pixel_distance, nearest = cKDTree(np.c_[gx, gy]).query(np.c_[px, py], k=1)
    near = pixel_distance <= float(pixel_radius)
    partial_ids_all = partial_index[py, px]
    generated_ids_all = generated_index[gy[nearest], gx[nearest]]
    point_distance = np.linalg.norm(
        partial_points[partial_ids_all] - generated_points[generated_ids_all], axis=1)
    depth_distance = np.abs(partial_depth[partial_ids_all] - generated_depth[generated_ids_all])
    if bbox_diagonal is None:
        joined = np.concatenate([partial_points, generated_points], axis=0)
        bbox_diagonal = float(np.linalg.norm(joined.max(axis=0) - joined.min(axis=0)))
    diagonal = max(float(bbox_diagonal), 1e-8)
    near_distances = point_distance[near]
    robust_limit = min(float(max_distance_ratio) * diagonal,
                       max(float(np.quantile(near_distances, trim_quantile)), .01 * diagonal)) if len(near_distances) else 0.
    matched = near & (point_distance <= robust_limit)
    if matched.any():
        distances, depths = point_distance[matched], depth_distance[matched]
        distance_mean = float(np.mean(distances))
        distance_p95 = float(np.quantile(distances, .95))
        depth_mean = float(np.mean(depths))
    else:
        distance_mean = distance_p95 = depth_mean = float("inf")
    pixel_coverage = float(near.mean())
    grid_coverage = _grid_coverage(np.c_[px, py], near, projector.image_shape)
    objective = (distance_mean + .35 * distance_p95 + .35 * depth_mean
                 + .12 * diagonal * (1. - pixel_coverage)
                 + .12 * diagonal * (1. - grid_coverage))
    return {"partial_ids": partial_ids_all[matched], "generated_ids": generated_ids_all[matched],
            "pixel_coverage": pixel_coverage, "grid_coverage": grid_coverage,
            "matched_coverage": float(matched.mean()), "distance_mean": distance_mean,
            "distance_p95": distance_p95, "depth_mean": depth_mean, "objective": float(objective),
            "robust_distance_limit": float(robust_limit), "visible_partial_count": int(len(px)),
            "matched_count": int(matched.sum())}


def fit_similarity_umeyama(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 3 or len(target) != len(source):
        raise ValueError("At least three paired points are required")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_center, target - target_center
    u, singular, vt = np.linalg.svd(target_zero.T @ source_zero / len(source))
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_center - scale * (rotation @ source_center)
    return transform


def bounded_delta_sim3(transform, *, max_rotation_deg, scale_bounds, max_translation):
    transform = np.asarray(transform, dtype=np.float64)
    scale = float(np.cbrt(max(np.linalg.det(transform[:3, :3]), 1e-12)))
    rotvec = Rotation.from_matrix(transform[:3, :3] / max(scale, 1e-12)).as_rotvec()
    angle, max_angle = float(np.linalg.norm(rotvec)), math.radians(float(max_rotation_deg))
    if angle > max_angle:
        rotvec *= max_angle / angle
    scale = float(np.clip(scale, scale_bounds[0], scale_bounds[1]))
    translation = transform[:3, 3].copy()
    norm = float(np.linalg.norm(translation))
    if norm > float(max_translation):
        translation *= float(max_translation) / norm
    bounded = np.eye(4, dtype=np.float64)
    bounded[:3, :3] = scale * Rotation.from_rotvec(rotvec).as_matrix()
    bounded[:3, 3] = translation
    return bounded


def apply_transform(points, transform):
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def smooth_observation_absorption(generated_points, partial_points, projector, *,
                                  bbox_diagonal, pixel_radius=5.0,
                                  influence_ratio=.08, max_displacement_ratio=.05,
                                  anchor_limit=4096, seed=6145):
    """Apply only a compact, evidence-gated visible-surface correction."""
    generated_points = np.asarray(generated_points, dtype=np.float64)
    partial_points = np.asarray(partial_points, dtype=np.float64)
    diagonal = max(float(bbox_diagonal), 1e-8)
    before = soft_ray_correspondences(
        partial_points, generated_points, projector, pixel_radius=pixel_radius,
        trim_quantile=.60, max_distance_ratio=.12, bbox_diagonal=diagonal)
    if before["matched_count"] < 64:
        return generated_points.copy(), {"accepted": False, "reason": "insufficient_visible_anchors",
                                         "before": before, "after": before, "anchor_count": 0}
    unique_ids, inverse = np.unique(before["generated_ids"], return_inverse=True)
    residuals = partial_points[before["partial_ids"]] - generated_points[before["generated_ids"]]
    sums, counts = np.zeros((len(unique_ids), 3)), np.zeros(len(unique_ids))
    np.add.at(sums, inverse, residuals)
    np.add.at(counts, inverse, 1.)
    anchor_points = generated_points[unique_ids]
    anchor_displacement = sums / np.maximum(counts[:, None], 1.)
    lengths = np.linalg.norm(anchor_displacement, axis=1)
    robust_limit = min(float(max_displacement_ratio) * diagonal, float(np.quantile(lengths, .80)))
    keep = lengths <= max(robust_limit, .005 * diagonal)
    anchor_points, anchor_displacement = anchor_points[keep], anchor_displacement[keep]
    if len(anchor_points) > int(anchor_limit):
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(len(anchor_points), int(anchor_limit), replace=False))
        anchor_points, anchor_displacement = anchor_points[chosen], anchor_displacement[chosen]
    if len(anchor_points) < 32:
        return generated_points.copy(), {"accepted": False, "reason": "insufficient_robust_anchors",
                                         "before": before, "after": before, "anchor_count": int(len(anchor_points))}
    influence = max(float(influence_ratio) * diagonal, 1e-8)
    distances, neighbors = cKDTree(anchor_points).query(generated_points, k=min(8, len(anchor_points)))
    if distances.ndim == 1:
        distances, neighbors = distances[:, None], neighbors[:, None]
    weights = np.square(np.clip(1. - distances / influence, 0., 1.)) * (distances < influence)
    weight_sum = weights.sum(axis=1)
    active = weight_sum > 1e-10
    displacement = np.zeros_like(generated_points)
    displacement[active] = (weights[active, :, None] * anchor_displacement[neighbors[active]]).sum(axis=1) / weight_sum[active, None]
    displacement *= np.clip(weight_sum / (weight_sum + .75), 0., 1.)[:, None]
    length = np.linalg.norm(displacement, axis=1)
    max_displacement = float(max_displacement_ratio) * diagonal
    over = length > max_displacement
    displacement[over] *= max_displacement / length[over, None]
    moved = generated_points + displacement
    after = soft_ray_correspondences(
        partial_points, moved, projector, pixel_radius=pixel_radius,
        trim_quantile=.60, max_distance_ratio=.12, bbox_diagonal=diagonal)
    accepted = bool(np.isfinite(after["objective"]) and after["objective"] < before["objective"] * .995
                    and after["grid_coverage"] >= before["grid_coverage"] - .03)
    return (moved if accepted else generated_points.copy()), {
        "accepted": accepted, "reason": "visible_surface_absorbed" if accepted else "absorption_rejected",
        "before": before, "after": after, "anchor_count": int(len(anchor_points)),
        "active_point_count": int(active.sum()), "max_displacement": float(length.max(initial=0.)),
        "mean_active_displacement": float(length[active].mean()) if active.any() else 0.}
