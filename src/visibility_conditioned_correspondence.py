"""Category-free visible structural patch correspondences.

Correspondence is deliberately not based on a named primitive.  Each visible
surface point receives a multi-scale local geometry descriptor, then candidate
matches are restricted to a saved-view pixel neighbourhood and made one-to-one
by mutual minimum cost.  This prevents many partial samples on one sofa panel
or tail segment from collapsing onto a single prior point.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.ray_consistent_registration import zbuffer_indices
from src.structural_patch_tto import estimate_normals_planarity


def local_shape_descriptor(points: np.ndarray, *, neighbours: int = 20,
                           batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Build rotation-invariant covariance-spectrum descriptors and normals."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("points must have shape (N, 3) with N >= 4")
    count = min(max(int(neighbours), 4), len(points))
    tree = cKDTree(points)
    descriptor = np.empty((len(points), 4), dtype=np.float64)
    for start in range(0, len(points), int(batch_size)):
        stop = min(start + int(batch_size), len(points))
        distance, ids = tree.query(points[start:stop], k=count, workers=-1)
        local = points[ids] - points[ids].mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", local, local) / max(count - 1, 1)
        values = np.linalg.eigvalsh(covariance)
        values = np.clip(values, 0., None)
        normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
        # Spectrum captures point-/line-/surface-likeness; neighbourhood scale
        # distinguishes a broad sofa cushion from a narrow support at one unit.
        descriptor[start:stop, :3] = normalized
        descriptor[start:stop, 3] = np.log(np.maximum(distance[:, -1], 1e-8))
    descriptor = (descriptor - descriptor.mean(axis=0)) / np.maximum(descriptor.std(axis=0), 1e-6)
    normals, _ = estimate_normals_planarity(points, neighbours=count, batch_size=batch_size)
    return descriptor, normals


def visible_structural_correspondences(
    partial: np.ndarray,
    prior: np.ndarray,
    projector,
    *,
    pixel_radius: float = 10.,
    candidate_count: int = 8,
    neighbours: int = 20,
    diagonal: float | None = None,
) -> dict:
    """Return mutual visible patch matches with geometry/normal/image costs."""
    partial = np.asarray(partial, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if diagonal is None:
        joined = np.concatenate((partial, prior), axis=0)
        diagonal = float(np.linalg.norm(joined.max(axis=0) - joined.min(axis=0)))
    diagonal = max(float(diagonal), 1e-8)
    p_desc, p_normal = local_shape_descriptor(partial, neighbours=neighbours)
    q_desc, q_normal = local_shape_descriptor(prior, neighbours=neighbours)
    p_uv, p_z = projector.project(partial); q_uv, q_z = projector.project(prior)
    _, p_mask, p_index = zbuffer_indices(p_uv, p_z, projector.image_shape)
    _, q_mask, q_index = zbuffer_indices(q_uv, q_z, projector.image_shape, splat_radius=1)
    py, px = np.where(p_mask); qy, qx = np.where(q_mask)
    if not len(px) or not len(qx):
        return {"partial_ids": np.empty(0, dtype=np.int64), "prior_ids": np.empty(0, dtype=np.int64),
                "cost": np.empty(0), "confidence": np.empty(0), "visible_partial": int(len(px))}
    p_ids = p_index[py, px]; q_ids = q_index[qy, qx]
    query_count = min(int(candidate_count), len(q_ids))
    pixel_distance, local_ids = cKDTree(np.c_[qx, qy]).query(np.c_[px, py], k=query_count)
    if query_count == 1:
        pixel_distance, local_ids = pixel_distance[:, None], local_ids[:, None]
    candidate_ids = q_ids[local_ids]
    descriptor_cost = np.linalg.norm(p_desc[p_ids, None] - q_desc[candidate_ids], axis=2) / np.sqrt(p_desc.shape[1])
    normal_cost = 1. - np.abs(np.sum(p_normal[p_ids, None] * q_normal[candidate_ids], axis=2))
    depth_cost = np.minimum(np.abs(p_z[p_ids, None] - q_z[candidate_ids]) / diagonal, .20)
    image_cost = pixel_distance / max(float(pixel_radius), 1e-8)
    cost = image_cost + .45 * descriptor_cost + .25 * normal_cost + .30 * depth_cost
    best_local = np.argmin(cost, axis=1)
    chosen_prior = candidate_ids[np.arange(len(p_ids)), best_local]
    chosen_cost = cost[np.arange(len(p_ids)), best_local]
    valid = pixel_distance[np.arange(len(p_ids)), best_local] <= float(pixel_radius)
    p_ids, chosen_prior, chosen_cost = p_ids[valid], chosen_prior[valid], chosen_cost[valid]
    # Mutual reduction: retain the best partial claimant for each prior point.
    order = np.lexsort((p_ids, chosen_cost, chosen_prior))
    first = np.r_[True, chosen_prior[order][1:] != chosen_prior[order][:-1]]
    selected = order[first]
    selected_cost = chosen_cost[selected]
    return {
        "partial_ids": p_ids[selected], "prior_ids": chosen_prior[selected],
        "cost": selected_cost, "confidence": np.exp(-selected_cost),
        "visible_partial": int(len(px)), "candidate_pairs": int(len(p_ids)),
        "mutual_pairs": int(len(selected)),
    }
