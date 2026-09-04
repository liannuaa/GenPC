"""Pixel-indexed correspondences and robust proper Sim(3) fitting."""

from __future__ import annotations

import numpy as np

from src.ray_consistent_registration import apply_transform
from src.robust_similarity import weighted_umeyama


def unique_pixel_matches(matches: np.ndarray) -> np.ndarray:
    """Keep the closest partial claimant for each target pixel-point."""
    matches = np.asarray(matches, dtype=np.float64)
    if matches.ndim != 2 or matches.shape[1] < 4:
        raise ValueError("matches must be (N, 4): partial, target, original, pixel distance")
    order = np.lexsort((matches[:, 0], matches[:, 3], matches[:, 1]))
    ordered = matches[order]
    return ordered[np.r_[True, ordered[1:, 1] != ordered[:-1, 1]]]


def robust_indexed_sim3(
    source_points: np.ndarray,
    target_points: np.ndarray,
    matches: np.ndarray,
    *,
    diagonal: float,
    trials: int = 384,
    seed: int = 6145,
) -> tuple[np.ndarray, dict]:
    """Fit an unbounded proper Sim(3) from mutually indexed visible points."""
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    pairs = unique_pixel_matches(matches)
    target_ids, source_ids = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    if (len(pairs) < 6 or np.any(target_ids < 0) or np.any(target_ids >= len(target_points))
            or np.any(source_ids < 0) or np.any(source_ids >= len(source_points))):
        raise ValueError("insufficient or invalid indexed pixel matches")
    source, target = source_points[source_ids], target_points[target_ids]
    confidence = np.exp(-np.clip(pairs[:, 3], 0., None) / 2.)
    initial = weighted_umeyama(source, target, confidence.copy())
    initial_residual = np.linalg.norm(apply_transform(source, initial) - target, axis=1)
    threshold = float(np.clip(np.quantile(initial_residual, .70), .012 * diagonal, .060 * diagonal))
    rng = np.random.default_rng(int(seed))
    probability = confidence / confidence.sum()
    best = {"transform": initial, "inlier": initial_residual <= threshold,
            "score": float(np.sum(confidence[initial_residual <= threshold]))}
    for _ in range(int(trials)):
        ids = rng.choice(len(source), 3, replace=False, p=probability)
        if np.linalg.matrix_rank(source[ids] - source[ids].mean(axis=0)) < 2:
            continue
        try:
            transform = weighted_umeyama(source[ids], target[ids], confidence[ids].copy())
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
        inlier = residual <= threshold
        score = float(np.sum(confidence[inlier] * (1. - residual[inlier] / threshold)))
        if score > best["score"]:
            best = {"transform": transform, "inlier": inlier, "score": score}
    transform = best["transform"]
    for _ in range(3):
        residual = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
        inlier = residual <= threshold
        if int(inlier.sum()) < 6:
            break
        robust = confidence[inlier] * np.clip(1. - residual[inlier] / threshold, .05, 1.)
        transform = weighted_umeyama(source[inlier], target[inlier], robust.copy())
    residual = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    return transform, {
        "unique_pixel_pairs": int(len(pairs)), "inlier_threshold": threshold,
        "inliers": int((residual <= threshold).sum()), "inlier_ratio": float((residual <= threshold).mean()),
        "mean_residual": float(residual.mean()), "p90_residual": float(np.quantile(residual, .90)),
        "scale": scale,
    }
