"""MoGe-assisted coarse Sim(3), followed by a real-scan-only refinement.

MoGe is used here only to enlarge the initialization basin.  It never enters
the final saved-view objective: the caller must refine and accept solely using
the hard partial scan after this bootstrap transform.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from scripts.run_pixal_pca_sim3_ttt_v2 import proper_pca_rotations
from src.ray_consistent_registration import apply_transform, bounded_delta_sim3
from src.structural_similarity_registration import weighted_umeyama


def _subset(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= int(count):
        return np.arange(len(values), dtype=np.int64)
    return np.linspace(0, len(values) - 1, int(count), dtype=np.int64)


def _metric_diagonal(points: np.ndarray) -> float:
    return max(float(np.linalg.norm(np.ptp(np.asarray(points), axis=0))), 1e-8)


def _coarse_score(moved: np.ndarray, target: np.ndarray, confidence: np.ndarray, diagonal: float) -> dict:
    target_tree, moved_tree = cKDTree(target), cKDTree(moved)
    forward, forward_ids = target_tree.query(moved, k=1, workers=-1)
    reverse, _ = moved_tree.query(target, k=1, workers=-1)
    weights = np.clip(confidence, 1e-4, None)
    forward_trim = forward <= np.quantile(forward, .65)
    reverse_trim = reverse <= np.quantile(reverse, .75)
    forward_value = float(np.mean(forward[forward_trim])) if np.any(forward_trim) else float("inf")
    reverse_value = (float(np.sum(weights[reverse_trim] * reverse[reverse_trim])
                           / np.sum(weights[reverse_trim])) if np.any(reverse_trim) else float("inf"))
    # The soft MoGe observation is incomplete. Target coverage is therefore
    # dominant, while prior-to-MoGe is trimmed to avoid penalizing hidden body.
    return {"objective": float((reverse_value + .30 * forward_value) / diagonal),
            "moge_to_prior_trimmed": reverse_value / diagonal,
            "prior_to_moge_trimmed": forward_value / diagonal,
            "target_coverage": float(np.mean(reverse <= .04 * diagonal)),
            "prior_trimmed_support": float(np.mean(forward_trim))}


def moge_coarse_proper_sim3(
    prior: np.ndarray,
    moge: np.ndarray,
    confidence: np.ndarray,
    *,
    prior_points: int = 24000,
    moge_points: int = 24000,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select a PCA-seeded, soft-MoGe-weighted coarse proper Sim(3)."""
    prior, moge, confidence = (np.asarray(prior, dtype=np.float64), np.asarray(moge, dtype=np.float64),
                               np.asarray(confidence, dtype=np.float64))
    if prior.ndim != 2 or moge.ndim != 2 or prior.shape[1] != 3 or moge.shape[1] != 3:
        raise ValueError("prior and moge must be (N, 3)")
    if confidence.shape != (len(moge),) or len(prior) < 6 or len(moge) < 6:
        raise ValueError("nontrivial, confidence-aligned point sets are required")
    source_ids, target_ids = _subset(prior, prior_points), _subset(moge, moge_points)
    source, target, target_confidence = prior[source_ids], moge[target_ids], confidence[target_ids]
    diagonal = _metric_diagonal(target)
    target_tree = cKDTree(target)
    candidates = []
    for pca_id, item in enumerate(proper_pca_rotations(source, target)):
        rotation = item["rotation"]
        source_center, target_center = np.median(source, axis=0), np.median(target, axis=0)
        rotated = source @ rotation.T
        scale = _metric_diagonal(target) / _metric_diagonal(rotated)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = scale * rotation
        transform[:3, 3] = target_center - scale * rotation @ source_center
        for _ in range(int(iterations)):
            moved = apply_transform(source, transform)
            distance, nearest = target_tree.query(moved, k=1, workers=-1)
            keep = distance <= np.quantile(distance, .65)
            if int(keep.sum()) < 6:
                break
            weights = target_confidence[nearest[keep]]
            delta = weighted_umeyama(moved[keep], target[nearest[keep]], weights.copy())
            delta = bounded_delta_sim3(delta, max_rotation_deg=10., scale_bounds=(.85, 1.15),
                                       max_translation=.12 * diagonal)
            transform = delta @ transform
        moved = apply_transform(source, transform)
        score = _coarse_score(moved, target, target_confidence, diagonal)
        candidates.append({"pca_id": pca_id, "transform": transform, "score": score})
    selected = min(candidates, key=lambda item: item["score"]["objective"])
    return apply_transform(prior, selected["transform"]), selected["transform"], {
        "method": "confidence_weighted_moge_visible_surface_pca_sim3_bootstrap",
        "coarse_target": "aligned semantic-image MoGe soft observation",
        "selected_pca_id": int(selected["pca_id"]), "selected_score": selected["score"],
        "candidate_scores": [{"pca_id": int(item["pca_id"]), "score": item["score"]} for item in candidates],
        "source_points": int(len(source)), "moge_points": int(len(target)),
    }
