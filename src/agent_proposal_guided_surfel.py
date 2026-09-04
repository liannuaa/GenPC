"""Proposal-guided local surfel editing with complete-prior preservation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.hierarchical_residual_registration import (
    remove_similarity_modes,
    visible_residual_components,
)


def _aggregate(indices: np.ndarray, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[int, list[np.ndarray]] = {}
    for index, delta in zip(indices, displacement):
        groups.setdefault(int(index), []).append(delta)
    ids = np.asarray(sorted(groups), dtype=np.int64)
    values = np.stack([np.median(np.stack(groups[int(index)]), axis=0) for index in ids])
    return ids, values


def proposal_guided_surfel_candidates(
    anchor: np.ndarray,
    guidance: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    diagonal: float,
    transfer_ratio: float = .025,
    sigma_ratio: float = .035,
    support_ratio: float = .11,
    max_displacement_ratio: float = .028,
    handle_neighbors: int = 8,
    continuation: tuple[float, ...] = (1., .75, .50, .35, .25, .15, .10, .05),
) -> tuple[list[tuple[float, np.ndarray]], dict]:
    """Transfer only local residual vectors, never generated surfels.

    The editor proposal supplies camera-visible vectors from its surface to
    partial observations.  Those vectors are attached to nearby anchor
    surfels, smoothed within a shared partial-support tube, and projected away
    from global Sim(3) modes.  Every output point descends from ``anchor``.
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    guidance = np.asarray(guidance, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    components, summary = visible_residual_components(
        partial, guidance, projector, diagonal=diagonal)
    if not components:
        return [], {**summary, "accepted_input": False}
    source, target = components[0][0]
    component = components[0][1]
    anchor_tree = cKDTree(anchor)
    transfer_distance, anchor_ids = anchor_tree.query(source, k=1, workers=-1)
    transferable = transfer_distance <= float(transfer_ratio) * diagonal
    if int(transferable.sum()) < 8:
        return [], {**summary, **component, "accepted_input": False,
                    "reason": "insufficient_nearby_generated_vectors",
                    "transferable_vectors": int(transferable.sum())}
    handle_ids, handle_delta = _aggregate(
        anchor_ids[transferable], target[transferable] - source[transferable])
    if len(handle_ids) < 6:
        return [], {**summary, **component, "accepted_input": False,
                    "reason": "insufficient_unique_surfel_handles",
                    "transferable_vectors": int(transferable.sum()),
                    "handles": int(len(handle_ids))}
    handle_points = anchor[handle_ids]
    distance, nearest = cKDTree(handle_points).query(
        anchor, k=min(int(handle_neighbors), len(handle_points)), workers=-1)
    if distance.ndim == 1:
        distance, nearest = distance[:, None], nearest[:, None]
    sigma = max(float(sigma_ratio) * diagonal, 1e-9)
    weights = np.exp(-.5 * (distance / sigma) ** 2)
    numerator = (weights[..., None] * handle_delta[nearest]).sum(axis=1)
    denominator = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    displacement = numerator / denominator
    partial_distance = cKDTree(partial).query(anchor, k=1, workers=-1)[0]
    support = np.clip(1. - partial_distance / max(float(support_ratio) * diagonal, 1e-9), 0., 1.)
    displacement *= support[:, None]
    displacement, removed = remove_similarity_modes(anchor, displacement, support)
    displacement *= support[:, None]
    length = np.linalg.norm(displacement, axis=1)
    cap = float(max_displacement_ratio) * diagonal
    displacement *= np.minimum(1., cap / np.maximum(length, 1e-12))[:, None]
    candidates = [(float(fraction), anchor + float(fraction) * displacement)
                  for fraction in continuation]
    return candidates, {
        **summary, **component, "accepted_input": True,
        "transferable_vectors": int(transferable.sum()), "handles": int(len(handle_ids)),
        "support_fraction": float(np.mean(support > 0.)),
        "transfer_ratio": float(transfer_ratio), "sigma_ratio": float(sigma_ratio),
        "support_ratio": float(support_ratio),
        "max_displacement_ratio": float(max_displacement_ratio),
        "displacement_p99_ratio": float(np.quantile(length, .99) / diagonal),
        "similarity_modes_removed": removed,
    }
