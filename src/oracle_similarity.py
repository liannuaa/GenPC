"""Offline GT-only similarity oracle for registration diagnostics."""

from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import cKDTree

from src.complete_prior_bridge import _pca_basis
from src.ray_consistent_registration import apply_transform, fit_similarity_umeyama


def _robust_radius(points: np.ndarray, centre: np.ndarray) -> float:
    return float(np.quantile(np.linalg.norm(points - centre, axis=1), 0.75))


def _trimmed_symmetric_score(source: np.ndarray, target: np.ndarray, fraction: float) -> float:
    forward = cKDTree(target).query(source, workers=-1)[0]
    reverse = cKDTree(source).query(target, workers=-1)[0]
    nf = max(1, int(round(len(forward) * fraction)))
    nr = max(1, int(round(len(reverse) * fraction)))
    return float(
        np.mean(np.partition(forward, nf - 1)[:nf])
        + np.mean(np.partition(reverse, nr - 1)[:nr])
    )


def gt_oracle_similarity(
    source: np.ndarray,
    ground_truth: np.ndarray,
    *,
    sample_points: int = 20_000,
    trim_fraction: float = 1.0,
    iterations: int = 30,
    seed: int = 6145,
) -> tuple[np.ndarray, dict]:
    """Fit a complete-to-complete proper Sim(3), explicitly using GT."""
    source = np.asarray(source, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    rng = np.random.default_rng(seed)
    src = source[rng.choice(len(source), min(sample_points, len(source)), replace=False)]
    tgt = ground_truth[rng.choice(len(ground_truth), min(sample_points, len(ground_truth)), replace=False)]
    src_centre, tgt_centre = np.median(src, axis=0), np.median(tgt, axis=0)
    scale = _robust_radius(tgt, tgt_centre) / max(_robust_radius(src, src_centre), 1e-12)
    src_basis, tgt_basis = _pca_basis(src), _pca_basis(tgt)
    records = []
    signed_permutations = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            signed_permutations.append(base @ np.diag(signs))
    basin = 0
    for signed_permutation in signed_permutations:
        rotation = tgt_basis @ signed_permutation @ src_basis.T
        if np.linalg.det(rotation) <= 0.0:
            continue
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = scale * rotation
        transform[:3, 3] = tgt_centre - scale * rotation @ src_centre
        current = apply_transform(src, transform)
        for _ in range(iterations):
            target_tree = cKDTree(tgt)
            forward_dist, forward_ids = target_tree.query(current, workers=-1)
            source_tree = cKDTree(current)
            reverse_dist, reverse_ids = source_tree.query(tgt, workers=-1)
            forward_count = max(16, int(round(len(forward_dist) * trim_fraction)))
            reverse_count = max(16, int(round(len(reverse_dist) * trim_fraction)))
            forward_keep = np.argpartition(forward_dist, forward_count - 1)[:forward_count]
            reverse_keep = np.argpartition(reverse_dist, reverse_count - 1)[:reverse_count]
            moving = np.concatenate([current[forward_keep], current[reverse_ids[reverse_keep]]], axis=0)
            fixed = np.concatenate([tgt[forward_ids[forward_keep]], tgt[reverse_keep]], axis=0)
            step = fit_similarity_umeyama(moving, fixed)
            current = apply_transform(current, step)
            transform = step @ transform
        score = _trimmed_symmetric_score(current, tgt, trim_fraction)
        records.append({"basin": basin, "score": score, "transform": transform})
        basin += 1
    selected = min(records, key=lambda item: item["score"])
    result = apply_transform(source, selected["transform"])
    linear = selected["transform"][:3, :3]
    selected_scale = float(np.cbrt(np.linalg.det(linear)))
    selected_rotation = linear / selected_scale
    return result, {
        "method": "offline_gt_complete_to_complete_surface_sim3_oracle",
        "ground_truth_used": True,
        "inference_eligible": False,
        "sample_points": int(min(sample_points, len(src), len(tgt))),
        "trim_fraction": float(trim_fraction),
        "iterations": int(iterations),
        "selected_basin": int(selected["basin"]),
        "trimmed_symmetric_score": float(selected["score"]),
        "scale": selected_scale,
        "rotation": selected_rotation.tolist(),
        "translation": selected["transform"][:3, 3].tolist(),
        "transform": selected["transform"].tolist(),
        "basins": [{"basin": int(item["basin"]), "score": float(item["score"])} for item in records],
    }
