"""Partial-anchored dual Gaussian-field state for a completion agent.

The complete prior and observed partial are represented as distinct Gaussian
populations in one method frame.  Only the prior population is a candidate for
global Sim(3) or local generative editing; partial Gaussians are fixed
geometric anchors.  Keeping the populations separate is crucial: direct point
union silently lets dense generated geometry overwhelm sparse observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DualGaussianField:
    """Compact geometry-only Gaussian state with explicit edit permissions."""

    means: np.ndarray              # [N, 3], method/partial coordinate frame
    log_scales: np.ndarray         # [N, 3], isotropic initialization retained per axis
    colors: np.ndarray             # [N, 3], uint8, semantic cue only
    opacity_logits: np.ndarray     # [N], float32
    observed_anchor: np.ndarray    # [N], bool; never moved by an edit action
    confidence: np.ndarray         # [N], partial=1, prior<1

    @property
    def prior_mask(self) -> np.ndarray:
        return ~self.observed_anchor

    def to_npz(self, path) -> None:
        np.savez_compressed(path, means=self.means, log_scales=self.log_scales,
                            colors=self.colors, opacity_logits=self.opacity_logits,
                            observed_anchor=self.observed_anchor, confidence=self.confidence)


def _initial_log_scales(points: np.ndarray, *, factor: float = .65) -> np.ndarray:
    """Set local isotropic splat radii from observed sampling density."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("Gaussian initialization needs at least three points per field")
    distance, _ = cKDTree(points).query(points, k=min(5, len(points)))
    spacing = np.median(distance[:, 1:], axis=1)
    floor = max(float(np.quantile(spacing, .01)) * .2, 1e-7)
    scales = np.maximum(spacing * factor, floor)
    return np.log(scales[:, None].repeat(3, axis=1)).astype(np.float32)


def make_dual_gaussian_field(
    prior: np.ndarray,
    partial: np.ndarray,
    *,
    prior_colors: np.ndarray | None = None,
    partial_color: tuple[int, int, int] = (170, 170, 170),
    prior_confidence: float = .35,
) -> DualGaussianField:
    """Initialize editable prior and immutable partial-anchor Gaussian sets."""
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != 3 or partial.ndim != 2 or partial.shape[1] != 3:
        raise ValueError("both fields must be [N, 3]")
    if not (0.0 < prior_confidence < 1.0):
        raise ValueError("prior confidence must lie strictly between zero and one")
    if prior_colors is None:
        prior_colors = np.tile(np.array((210, 210, 210), dtype=np.uint8), (len(prior), 1))
    prior_colors = np.asarray(prior_colors, dtype=np.uint8)
    if prior_colors.shape != (len(prior), 3):
        raise ValueError("prior_colors must match the prior point count")
    partial_colors = np.tile(np.asarray(partial_color, dtype=np.uint8), (len(partial), 1))
    means = np.concatenate((prior, partial), axis=0).astype(np.float32)
    log_scales = np.concatenate((_initial_log_scales(prior), _initial_log_scales(partial)), axis=0)
    anchors = np.concatenate((np.zeros(len(prior), dtype=bool), np.ones(len(partial), dtype=bool)))
    confidence = np.concatenate((np.full(len(prior), prior_confidence, dtype=np.float32),
                                 np.ones(len(partial), dtype=np.float32)))
    # The logistic values are intentionally conservative.  The later renderer
    # learns appearance/opacity but has no permission to translate anchors.
    opacity = np.concatenate((np.full(len(prior), -1.0, dtype=np.float32),
                              np.full(len(partial), 1.5, dtype=np.float32)))
    return DualGaussianField(means, log_scales, np.concatenate((prior_colors, partial_colors)),
                             opacity, anchors, confidence)


def apply_prior_sim3(field: DualGaussianField, transform: np.ndarray) -> DualGaussianField:
    """Move only editable prior Gaussians with a validated proper Sim(3)."""
    from src.prior_backend_contract import validate_proper_sim3

    transform = np.asarray(transform, dtype=np.float64)
    scale = validate_proper_sim3(transform)
    means = field.means.copy()
    prior = field.prior_mask
    means[prior] = means[prior] @ transform[:3, :3].T + transform[:3, 3]
    log_scales = field.log_scales.copy()
    log_scales[prior] += np.log(scale)
    return DualGaussianField(means, log_scales, field.colors.copy(), field.opacity_logits.copy(),
                             field.observed_anchor.copy(), field.confidence.copy())


def observed_anchor_loss(rendered_depth: np.ndarray, observed_depth: np.ndarray,
                         observed_mask: np.ndarray, *, truncation: float) -> float:
    """Robust depth loss on visible observed support, independent of RGB edits."""
    rendered_depth = np.asarray(rendered_depth, dtype=np.float64)
    observed_depth = np.asarray(observed_depth, dtype=np.float64)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    valid = observed_mask & np.isfinite(rendered_depth) & np.isfinite(observed_depth)
    if not np.any(valid):
        return float("inf")
    residual = np.abs(rendered_depth[valid] - observed_depth[valid])
    return float(np.mean(np.minimum(residual, truncation)))
