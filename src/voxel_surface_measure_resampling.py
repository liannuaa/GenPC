"""Support-aware voxel resampling for a uniform completed surface measure."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _nearest_neighbor_spacing(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64)
    return cKDTree(points).query(points, k=2, workers=-1)[0][:, 1]


def _voxel_representatives(
    points: np.ndarray,
    observed: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Return original-point representatives, preferring observations per cell."""
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / max(float(voxel_size), 1e-12)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    groups = int(inverse.max()) + 1
    centroids = np.zeros((groups, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=groups).astype(np.float64)
    np.add.at(centroids, inverse, points)
    centroids /= counts[:, None]
    distance2 = np.sum((points - centroids[inverse]) ** 2, axis=1)
    # An observed point wins its cell whenever available; distance then chooses
    # the least biased original sample nearest the voxel centroid.
    source_rank = (~observed).astype(np.int8)
    order = np.lexsort((distance2, source_rank, inverse))
    sorted_groups = inverse[order]
    first = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    return order[first]


def support_aware_voxel_resample(
    points,
    *,
    observation_count: int,
    target_points: int = 32768,
    outlier_k: int = 8,
    outlier_mad_scale: float = 4.0,
    support_ratio: float = 0.02,
    binary_steps: int = 18,
):
    """Denoise unsupported observation outliers and voxel-uniformize a surface.

    The last ``observation_count`` input points are the FPS observation mass
    inserted by the preceding surface posterior. Returned points are a subset
    of the input, so the operation cannot hallucinate or geometrically warp the
    complete body.
    """
    points = np.asarray(points, dtype=np.float64)
    observation_count = int(np.clip(observation_count, 0, len(points)))
    prior_count = len(points) - observation_count
    prior = points[:prior_count]
    observation = points[prior_count:]
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-12)

    removed = np.zeros(len(observation), dtype=bool)
    if len(observation) > max(outlier_k + 1, 32) and len(prior) > 0:
        k = min(int(outlier_k) + 1, len(observation))
        isolation = cKDTree(observation).query(
            observation, k=k, workers=-1)[0][:, -1]
        median = float(np.median(isolation))
        mad = float(np.median(np.abs(isolation - median)))
        isolation_limit = median + float(outlier_mad_scale) * 1.4826 * max(mad, 1e-12)
        support = cKDTree(prior).query(observation, k=1, workers=-1)[0]
        removed = ((isolation > isolation_limit)
                   & (support > float(support_ratio) * diagonal))
        observation = observation[~removed]

    merged = np.concatenate((prior, observation), axis=0)
    observed = np.r_[np.zeros(len(prior), dtype=bool),
                     np.ones(len(observation), dtype=bool)]
    target = int(np.clip(target_points, 1024, len(merged)))

    low = diagonal * 1e-6
    high = diagonal
    best_ids = np.arange(len(merged), dtype=np.int64)
    best_gap = abs(len(best_ids) - target)
    best_size = low
    for _ in range(int(binary_steps)):
        size = (low + high) * .5
        ids = _voxel_representatives(merged, observed, size)
        gap = abs(len(ids) - target)
        if gap < best_gap:
            best_ids, best_gap, best_size = ids, gap, size
        if len(ids) > target:
            low = size
        else:
            high = size

    result = merged[best_ids]
    selected_observed = observed[best_ids]
    before_spacing = _nearest_neighbor_spacing(merged) / diagonal
    after_spacing = _nearest_neighbor_spacing(result) / diagonal
    prior_spacing = (_nearest_neighbor_spacing(prior) / diagonal
                     if len(prior) else np.empty(0))
    observation_spacing = (_nearest_neighbor_spacing(observation) / diagonal
                           if len(observation) else np.empty(0))
    return result, {
        "input_points": int(len(points)),
        "output_points": int(len(result)),
        "target_points": int(target),
        "voxel_size": float(best_size),
        "voxel_size_ratio": float(best_size / diagonal),
        "input_prior_points": int(len(prior)),
        "input_observation_points": int(observation_count),
        "observation_outliers_removed": int(removed.sum()),
        "selected_prior_points": int((~selected_observed).sum()),
        "selected_observation_points": int(selected_observed.sum()),
        "selected_observation_fraction": float(selected_observed.mean()),
        "prior_knn_median_before": float(np.median(prior_spacing)) if len(prior_spacing) else None,
        "observation_knn_median_before": float(np.median(observation_spacing)) if len(observation_spacing) else None,
        "global_knn_median_before": float(np.median(before_spacing)),
        "global_knn_median_after": float(np.median(after_spacing)),
        "global_knn_cv_before": float(np.std(before_spacing) / max(np.mean(before_spacing), 1e-12)),
        "global_knn_cv_after": float(np.std(after_spacing) / max(np.mean(after_spacing), 1e-12)),
        "subset_of_input": True,
        "new_geometry_created": False,
    }
