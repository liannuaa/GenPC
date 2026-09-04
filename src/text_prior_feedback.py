"""Convert geometric residuals into a bounded, category-free text edit.

The agent does not ask a language model to diagnose a shape.  It measures the
registered partial-to-prior discrepancy itself, then writes a short prompt for
a text-conditioned *variant* model.  All directions are expressed in the
principal frame of the observed scan, so the policy is shared across objects
and datasets and needs neither labels nor ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TextEditFeedback:
    """A compact geometric diagnosis and its controlled natural-language form."""

    eligible: bool
    residual_q70_ratio: float
    residual_q90_ratio: float
    principal_axis: int
    observed_extent_ratio: float
    correction: str
    strength: str
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centred = np.asarray(points, dtype=np.float64) - np.median(points, axis=0)
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def _extent(values: np.ndarray) -> float:
    low, high = np.quantile(values, (.05, .95))
    return float(max(high - low, 1e-12))


def build_text_edit_feedback(
    partial: np.ndarray,
    registered_prior: np.ndarray,
    *,
    minimum_q90_ratio: float = .018,
    extent_change_ratio: float = .035,
) -> TextEditFeedback:
    """Describe only a supported visible-shape correction.

    A partial scan cannot identify global shape extent.  We therefore compare
    it solely with the prior points that are nearest to observed points, and
    only activate a text proposal when the unmatched visible residual is
    meaningful relative to the scan diagonal.  The returned prompt is generic
    by design: object identity and hidden geometry remain anchored by the
    supplied base mesh to ``run_variant``.
    """
    partial = np.asarray(partial, dtype=np.float64)
    prior = np.asarray(registered_prior, dtype=np.float64)
    if len(partial) < 16 or len(prior) < 16:
        raise ValueError("text feedback requires nontrivial partial and prior point sets")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    distance, nearest = cKDTree(prior).query(partial, k=1, workers=-1)
    q70 = float(np.quantile(distance, .70) / diagonal)
    q90 = float(np.quantile(distance, .90) / diagonal)
    axes = _principal_axes(partial)
    observed = partial @ axes
    supported_prior = prior[np.asarray(nearest, dtype=np.int64)] @ axes
    observed_extents = np.asarray([_extent(observed[:, axis]) for axis in range(3)])
    prior_extents = np.asarray([_extent(supported_prior[:, axis]) for axis in range(3)])
    ratios = observed_extents / np.maximum(prior_extents, diagonal * 1e-8)
    axis = int(np.argmax(np.abs(np.log(np.maximum(ratios, 1e-8)))))
    extent_ratio = float(ratios[axis])
    if extent_ratio > 1.0 + extent_change_ratio:
        correction = "extend"
    elif extent_ratio < 1.0 - extent_change_ratio:
        correction = "contract"
    else:
        correction = "conform"
    strength = "slightly" if q90 < .055 else "moderately"
    eligible = bool(q90 >= float(minimum_q90_ratio))
    axis_names = ("long", "middle", "short")
    if correction == "extend":
        adjustment = (
            f"{strength} extend the camera-visible observed region along its "
            f"{axis_names[axis]} principal axis so that it covers the scan"
        )
    elif correction == "contract":
        adjustment = (
            f"{strength} contract the camera-visible observed region along its "
            f"{axis_names[axis]} principal axis so that it follows the scan"
        )
    else:
        adjustment = (
            "make a small camera-visible surface correction so that the observed "
            "surface conforms more closely to the scan"
        )
    prompt = (
        "A complete coherent 3D object. Preserve the input asset's identity, "
        "global orientation, global proportions, structural parts, and all hidden "
        "geometry. Do not replace the object or remove components. "
        f"Only {adjustment}. Keep the correction local and smooth."
    )
    return TextEditFeedback(
        eligible=eligible,
        residual_q70_ratio=q70,
        residual_q90_ratio=q90,
        principal_axis=axis,
        observed_extent_ratio=extent_ratio,
        correction=correction,
        strength=strength,
        prompt=prompt,
    )
