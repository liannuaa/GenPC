"""Convert MoGe into a confidence-gated soft extension of a real partial scan.

MoGe is useful only after its monocular coordinate system is metrically aligned
to the scan.  The real partial remains the hard observation; the aligned MoGe
surface can add low-confidence samples *only* inside image cells already
supported by stable real-scan/MoGe correspondences.  It never becomes a final
completion by point-cloud union.
"""

from __future__ import annotations

import numpy as np

from src.ray_consistent_registration import apply_transform
from src.robust_similarity import weighted_umeyama


def _unique_pairs(matches: np.ndarray) -> np.ndarray:
    """Keep the closest pixel claimant for every MoGe point."""
    matches = np.asarray(matches, dtype=np.float64)
    if matches.ndim != 2 or matches.shape[1] < 4:
        raise ValueError("matches must be (N, 4): partial, MoGe, original, pixel distance")
    order = np.lexsort((matches[:, 0], matches[:, 3], matches[:, 1]))
    ordered = matches[order]
    first = np.r_[True, ordered[1:, 1] != ordered[:-1, 1]]
    return ordered[first]


def robust_moge_to_partial(
    moge_points: np.ndarray,
    partial: np.ndarray,
    matches: np.ndarray,
    *,
    diagonal: float,
    trials: int = 384,
    seed: int = 6145,
) -> tuple[np.ndarray, dict]:
    """Estimate a proper unbounded Sim(3) from pixel-indexed scan matches."""
    moge_points, partial = np.asarray(moge_points, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    pairs = _unique_pairs(matches)
    partial_ids, moge_ids = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    if (len(pairs) < 6 or np.any(partial_ids < 0) or np.any(partial_ids >= len(partial))
            or np.any(moge_ids < 0) or np.any(moge_ids >= len(moge_points))):
        raise ValueError("insufficient or invalid MoGe/partial pixel matches")
    source, target = moge_points[moge_ids], partial[partial_ids]
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


def confidence_gated_moge_extension(
    aligned_moge: np.ndarray,
    moge_pixel_xy: np.ndarray,
    matches: np.ndarray,
    matched_residual: np.ndarray,
    *,
    image_size: int = 512,
    grid_size: int = 32,
    min_cell_matches: int = 8,
    residual_ratio: float = .035,
    diagonal: float,
    soft_confidence: float = .25,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep MoGe only in real-observation-supported, low-residual image cells."""
    aligned_moge, moge_pixel_xy = np.asarray(aligned_moge, dtype=np.float64), np.asarray(moge_pixel_xy, dtype=np.float64)
    pairs, residual = _unique_pairs(matches), np.asarray(matched_residual, dtype=np.float64)
    if len(residual) != len(pairs):
        raise ValueError("matched_residual must align with unique MoGe pixel pairs")
    if moge_pixel_xy.shape != (len(aligned_moge), 2):
        raise ValueError("moge_pixel_xy must align with aligned_moge")
    threshold = float(residual_ratio) * max(float(diagonal), 1e-8)
    gx = np.clip((moge_pixel_xy[:, 0] * grid_size / image_size).astype(np.int64), 0, grid_size - 1)
    gy = np.clip((moge_pixel_xy[:, 1] * grid_size / image_size).astype(np.int64), 0, grid_size - 1)
    pair_ids = pairs[:, 1].astype(np.int64)
    cell_count = np.zeros((grid_size, grid_size), dtype=np.int64)
    cell_residual = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    for y in range(grid_size):
        for x in range(grid_size):
            ids = pair_ids[(gx[pair_ids] == x) & (gy[pair_ids] == y)]
            if len(ids) == 0:
                continue
            pair_mask = np.isin(pair_ids, ids, assume_unique=False)
            cell_count[y, x] = int(pair_mask.sum())
            cell_residual[y, x] = float(np.median(residual[pair_mask]))
    support = (cell_count >= int(min_cell_matches)) & (cell_residual <= threshold)
    keep = support[gy, gx]
    confidence = np.zeros(len(aligned_moge), dtype=np.float32)
    confidence[keep] = (float(soft_confidence)
                        * np.exp(-cell_residual[gy[keep], gx[keep]] / max(threshold, 1e-8))).astype(np.float32)
    return aligned_moge[keep], confidence[keep], {
        "grid_size": int(grid_size), "supported_cells": int(support.sum()),
        "kept_points": int(keep.sum()), "soft_confidence_max": float(confidence.max(initial=0.)),
        "cell_residual_threshold": threshold,
    }


def moge_visible_soft_completion(
    aligned_moge: np.ndarray,
    moge_pixel_xy: np.ndarray,
    matches: np.ndarray,
    matched_residual: np.ndarray,
    *,
    image_size: int = 512,
    grid_size: int = 32,
    min_cell_matches: int = 8,
    residual_ratio: float = .035,
    diagonal: float,
    reliable_confidence: float = .25,
    fallback_confidence: float = .05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the complete MoGe foreground as a two-level soft observation.

    Every masked MoGe point is retained as the semantic-view-visible soft
    layer.  Cells calibrated by stable real-scan matches receive a higher
    confidence; all other visible MoGe points are explicitly weak.  The caller
    must still keep the real partial as a separate hard layer and give it
    z-buffer priority on overlapping image rays.
    """
    aligned_moge, pixels = np.asarray(aligned_moge, dtype=np.float64), np.asarray(moge_pixel_xy, dtype=np.float64)
    pairs, residual = _unique_pairs(matches), np.asarray(matched_residual, dtype=np.float64)
    if len(residual) != len(pairs) or pixels.shape != (len(aligned_moge), 2):
        raise ValueError("MoGe points, pixels, and residuals must align")
    threshold = float(residual_ratio) * max(float(diagonal), 1e-8)
    gx = np.clip((pixels[:, 0] * grid_size / image_size).astype(np.int64), 0, grid_size - 1)
    gy = np.clip((pixels[:, 1] * grid_size / image_size).astype(np.int64), 0, grid_size - 1)
    pair_ids = pairs[:, 1].astype(np.int64)
    count = np.zeros((grid_size, grid_size), dtype=np.int64)
    median = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    for y in range(grid_size):
        for x in range(grid_size):
            selected = (gx[pair_ids] == x) & (gy[pair_ids] == y)
            if int(selected.sum()) > 0:
                count[y, x] = int(selected.sum())
                median[y, x] = float(np.median(residual[selected]))
    reliable = (count >= int(min_cell_matches)) & (median <= threshold)
    confidence = np.full(len(aligned_moge), float(fallback_confidence), dtype=np.float32)
    high = reliable[gy, gx]
    confidence[high] = (float(reliable_confidence)
                        * np.exp(-median[gy[high], gx[high]] / max(threshold, 1e-8))).astype(np.float32)
    return aligned_moge.copy(), confidence, {
        "visible_moge_points": int(len(aligned_moge)), "reliable_cells": int(reliable.sum()),
        "reliable_points": int(high.sum()), "fallback_points": int((~high).sum()),
        "reliable_confidence_max": float(confidence.max(initial=0.)),
        "fallback_confidence": float(fallback_confidence),
        "cell_residual_threshold": threshold,
        "composition": "MoGe-visible soft layer plus independent real-partial hard layer",
    }
