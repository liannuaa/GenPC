"""Decode a partial-anchored Gaussian field without raw point-cloud union."""

from __future__ import annotations

import numpy as np


def collision_free_anchor_pairs(pixel_pairs: np.ndarray, partial: np.ndarray, prior: np.ndarray, *,
                                max_residual: float) -> tuple[np.ndarray, dict]:
    """Keep at most one hard partial anchor for each editable prior Gaussian.

    Rows of ``pixel_pairs`` follow the shared indexed-pixel convention
    ``[partial_id, prior_id, prior_id, pixel_distance]``.  Pixel visibility
    makes pair creation camera-aware; this function then performs a stable
    one-to-one reduction so decoding never increases density or deletes an
    unobserved portion of the complete prior.
    """
    pairs = np.asarray(pixel_pairs, dtype=np.float64)
    partial, prior = np.asarray(partial, dtype=np.float64), np.asarray(prior, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] < 4:
        raise ValueError("pixel_pairs must be shaped (N, >=4)")
    if max_residual <= 0.:
        raise ValueError("max_residual must be positive")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64), {
            "pixel_pairs": 0, "within_metric_limit": 0, "collision_free_pairs": 0,
            "max_residual": float(max_residual),
        }
    partial_ids, prior_ids = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    valid = ((partial_ids >= 0) & (partial_ids < len(partial))
             & (prior_ids >= 0) & (prior_ids < len(prior)))
    partial_ids, prior_ids, pixel_distance = partial_ids[valid], prior_ids[valid], pairs[valid, 3]
    residual = np.linalg.norm(partial[partial_ids] - prior[prior_ids], axis=1)
    keep = np.isfinite(residual) & (residual <= float(max_residual))
    partial_ids, prior_ids = partial_ids[keep], prior_ids[keep]
    residual, pixel_distance = residual[keep], pixel_distance[keep]
    # Greedy by geometric agreement, then image agreement, then stable source
    # ids.  A pair claims both its partial anchor and its prior Gaussian.
    order = np.lexsort((prior_ids, partial_ids, pixel_distance, residual))
    used_partial, used_prior, rows = set(), set(), []
    for row in order:
        partial_id, prior_id = int(partial_ids[row]), int(prior_ids[row])
        if partial_id in used_partial or prior_id in used_prior:
            continue
        used_partial.add(partial_id); used_prior.add(prior_id)
        rows.append((partial_id, prior_id))
    selected = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    return selected, {
        "pixel_pairs": int(len(pairs)), "within_metric_limit": int(keep.sum()),
        "collision_free_pairs": int(len(selected)), "max_residual": float(max_residual),
        "residual_median": float(np.median(residual)) if len(residual) else float("inf"),
        "residual_p90": float(np.quantile(residual, .90)) if len(residual) else float("inf"),
    }


def decode_partial_anchored_gaussians(
    edited_prior: np.ndarray,
    partial: np.ndarray,
    pixel_pairs: np.ndarray,
    *,
    max_residual: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return exactly the complete prior count with collision-free hard anchors.

    The returned cloud has one row for every editable complete-prior Gaussian.
    A selected hard anchor replaces the position of its corresponding Gaussian;
    all other Gaussian centres, including the entire unobserved body, are
    copied verbatim.
    """
    edited_prior = np.asarray(edited_prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if edited_prior.ndim != 2 or edited_prior.shape[1] != 3:
        raise ValueError("edited_prior must be (N, 3)")
    selected, info = collision_free_anchor_pairs(
        pixel_pairs, partial, edited_prior, max_residual=max_residual,
    )
    decoded = edited_prior.copy()
    if len(selected):
        decoded[selected[:, 1]] = partial[selected[:, 0]]
    info.update({
        "prior_points": int(len(edited_prior)), "decoded_points": int(len(decoded)),
        "all_prior_slots_preserved": bool(len(decoded) == len(edited_prior)),
        "partial_anchor_fraction": float(len(selected) / max(len(decoded), 1)),
        "decoder": "collision_free_saved_view_hard_anchor_replacement",
    })
    return decoded, selected, info
