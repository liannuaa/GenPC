"""GT-free orthographic evidence for bounded completion-agent actions.

The saved camera is the principal observation, but an edit can look better in
that view while widening an unobserved side of the object.  This module adds
three deterministic PCA-frame silhouette checks.  The frame and canvas are
fixed by the observed partial and trusted anchor before a proposal is scored,
so a proposal cannot improve merely by changing its own normalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OrthographicReference:
    """Object-independent PCA frame and fixed raster ranges."""

    centre: np.ndarray
    frame: np.ndarray
    limits: np.ndarray  # [3, 2], PCA-coordinate lower/upper limits
    size: int


@dataclass(frozen=True)
class MultiViewEvidence:
    """Mean three-view support metrics plus an out-of-canvas safety signal."""

    iou: float
    coverage: float
    leakage: float
    outside_prior_ratio: float
    per_view: tuple[dict[str, float], ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _pca_frame(points: np.ndarray) -> np.ndarray:
    centred = np.asarray(points, dtype=np.float64) - np.median(points, axis=0)
    _, axes = np.linalg.eigh(centred.T @ centred / max(len(centred) - 1, 1))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def make_orthographic_reference(
    partial: np.ndarray,
    anchor: np.ndarray,
    *,
    size: int = 256,
    padding: float = .12,
) -> OrthographicReference:
    """Freeze a broad PCA canvas using only partial and trusted prior geometry."""
    partial = np.asarray(partial, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if len(partial) < 16 or len(anchor) < 16:
        raise ValueError("multi-view evidence needs nontrivial point sets")
    centre = np.median(partial, axis=0)
    frame = _pca_frame(partial)
    values = np.concatenate(((partial - centre) @ frame, (anchor - centre) @ frame), axis=0)
    low, high = np.quantile(values, (.005, .995), axis=0)
    span = np.maximum(high - low, 1e-8)
    limits = np.stack((low - padding * span, high + padding * span), axis=1)
    return OrthographicReference(centre=centre, frame=frame, limits=limits, size=int(size))


def _dilate(mask: np.ndarray) -> np.ndarray:
    """One-pixel zero-padded dilation, avoiding a scipy runtime dependency."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for row in range(3):
        for col in range(3):
            out |= padded[row:row + mask.shape[0], col:col + mask.shape[1]]
    return out


def _mask(values: np.ndarray, axes: tuple[int, int], reference: OrthographicReference) -> tuple[np.ndarray, float]:
    limits = reference.limits[np.asarray(axes)]
    xy = values[:, axes]
    uv = (xy - limits[:, 0]) / np.maximum(limits[:, 1] - limits[:, 0], 1e-8)
    pixels = np.floor(uv * (reference.size - 1)).astype(np.int64)
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < reference.size)
             & (pixels[:, 1] >= 0) & (pixels[:, 1] < reference.size))
    mask = np.zeros((reference.size, reference.size), dtype=bool)
    selected = pixels[valid]
    mask[reference.size - 1 - selected[:, 1], selected[:, 0]] = True
    return _dilate(mask), float(1.0 - np.mean(valid))


def measure_multiview_evidence(
    partial: np.ndarray,
    prior: np.ndarray,
    reference: OrthographicReference,
) -> MultiViewEvidence:
    """Measure partial support and prior spill in the fixed three-view frame."""
    partial_values = (np.asarray(partial, dtype=np.float64) - reference.centre) @ reference.frame
    prior_values = (np.asarray(prior, dtype=np.float64) - reference.centre) @ reference.frame
    rows: list[dict[str, float]] = []
    outside = []
    for axes in ((0, 1), (0, 2), (1, 2)):
        partial_mask, _ = _mask(partial_values, axes, reference)
        prior_mask, prior_outside = _mask(prior_values, axes, reference)
        overlap = partial_mask & prior_mask
        union = partial_mask | prior_mask
        partial_count = max(int(partial_mask.sum()), 1)
        prior_count = max(int(prior_mask.sum()), 1)
        rows.append({
            "iou": float(overlap.sum() / max(int(union.sum()), 1)),
            "coverage": float(overlap.sum() / partial_count),
            "leakage": float((prior_mask & ~partial_mask).sum() / prior_count),
            "outside_prior_ratio": prior_outside,
        })
        outside.append(prior_outside)
    return MultiViewEvidence(
        iou=float(np.mean([row["iou"] for row in rows])),
        coverage=float(np.mean([row["coverage"] for row in rows])),
        leakage=float(np.mean([row["leakage"] for row in rows])),
        outside_prior_ratio=float(np.mean(outside)),
        per_view=tuple(rows),
    )


def accept_multiview_no_harm(
    anchor: MultiViewEvidence,
    proposal: MultiViewEvidence,
    *,
    maximum_iou_loss: float = .01,
    maximum_coverage_loss: float = .01,
    maximum_leakage_increase: float = .025,
    maximum_outside_increase: float = .01,
) -> bool:
    """Reject edits that trade one saved-view gain for a global silhouette loss."""
    return bool(
        proposal.iou >= anchor.iou - maximum_iou_loss
        and proposal.coverage >= anchor.coverage - maximum_coverage_loss
        and proposal.leakage <= anchor.leakage + maximum_leakage_increase
        and proposal.outside_prior_ratio <= anchor.outside_prior_ratio + maximum_outside_increase
    )
