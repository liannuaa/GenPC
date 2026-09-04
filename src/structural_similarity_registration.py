"""Robust proper Sim(3) from category-free visible structural matches."""

from __future__ import annotations

import numpy as np

from src.ray_consistent_registration import apply_transform, bounded_delta_sim3


def weighted_umeyama(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted proper isotropic similarity transform from source to target."""
    source, target, weights = (np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64),
                               np.asarray(weights, dtype=np.float64))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source and target must be aligned (N, 3) arrays with N >= 3")
    weights = np.clip(weights, 0., None)
    weights /= max(float(weights.sum()), 1e-12)
    source_center = (weights[:, None] * source).sum(axis=0)
    target_center = (weights[:, None] * target).sum(axis=0)
    src = source - source_center; tgt = target - target_center
    covariance = (weights[:, None] * tgt).T @ src
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0.:
        sign[-1] = -1.
    rotation = u @ np.diag(sign) @ vt
    variance = float((weights * np.sum(src * src, axis=1)).sum())
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    transform = np.eye(4); transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_center - scale * rotation @ source_center
    return transform


def robust_structural_similarity(
    source: np.ndarray, target: np.ndarray, confidence: np.ndarray, *, diagonal: float,
    trials: int = 512, inlier_ratio: float = .045, seed: int = 6145,
    max_rotation_deg: float = 8., scale_bounds=(.90, 1.10), max_translation_ratio: float = .08,
) -> tuple[np.ndarray, dict]:
    """RANSAC + confidence-weighted proper Sim(3), bounded in one shared trust region."""
    source, target, confidence = (np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64),
                                  np.asarray(confidence, dtype=np.float64))
    if len(source) < 6 or len(source) != len(target) or len(source) != len(confidence):
        raise ValueError("at least six aligned structural matches are required")
    diagonal = max(float(diagonal), 1e-8); threshold = float(inlier_ratio) * diagonal
    probability = np.clip(confidence, 1e-6, None); probability /= probability.sum()
    rng = np.random.default_rng(int(seed)); best = None
    for _ in range(int(trials)):
        ids = rng.choice(len(source), 3, replace=False, p=probability)
        if np.linalg.matrix_rank(source[ids] - source[ids].mean(axis=0)) < 2:
            continue
        try:
            transform = weighted_umeyama(source[ids], target[ids], confidence[ids])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
        inlier = residual <= threshold
        score = float(np.sum(confidence[inlier] * (1. - residual[inlier] / threshold)))
        if best is None or score > best["score"]:
            best = {"transform": transform, "inlier": inlier, "score": score}
    if best is None or best["inlier"].sum() < 6:
        raise RuntimeError("no stable structural similarity hypothesis")
    transform = best["transform"]
    for _ in range(3):
        residual = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
        inlier = residual <= threshold
        if inlier.sum() < 6:
            break
        robust = confidence[inlier] * np.clip(1. - residual[inlier] / threshold, .05, 1.)
        transform = weighted_umeyama(source[inlier], target[inlier], robust)
    bounded = bounded_delta_sim3(transform, max_rotation_deg=max_rotation_deg,
                                 scale_bounds=scale_bounds, max_translation=max_translation_ratio * diagonal)
    final = np.linalg.norm(apply_transform(source, bounded) - target, axis=1)
    return bounded, {"trials": int(trials), "inlier_threshold": threshold,
                     "raw_inliers": int(best["inlier"].sum()), "bounded_inliers": int((final <= threshold).sum()),
                     "residual_before_mean": float(np.linalg.norm(source - target, axis=1).mean()),
                     "residual_after_mean": float(final.mean())}
