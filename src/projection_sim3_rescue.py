"""Evidence-only global Sim(3) recovery for an empty Camera-1 overlap.

This tool is intentionally narrow: it does not estimate a new orientation or
invent 3-D correspondences.  It assumes the native/bridge route already
provides the orientation, and repairs only a large monocular gauge error in
translation and isotropic scale.  Candidate ranking uses rendered Camera-1
silhouette, coverage, leakage, and visible depth; no GT or category signal is
available.
"""

from __future__ import annotations

import numpy as np

from src.bidirectional_cycle_registration import projection_metrics
from src.ray_consistent_registration import apply_transform


def projection_energy(metrics: dict[str, float]) -> float:
    """Finite, dimensionless score for recovery before 3-D pairs exist."""
    depth = min(float(metrics["visible_depth_normalized"]), .20)
    return float(
        .60 * (1.0 - float(metrics["iou"]))
        + .20 * (1.0 - float(metrics["coverage"]))
        + .15 * float(metrics["leakage"])
        + .05 * depth
    )


def _pca_axes(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.median(points, axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return vectors


def center_scale_projection_rescue(partial, prior, projector):
    """Return the best bounded center/scale proposal and its rendered evidence.

    The central candidate aligns robust centres after an isotropic bbox-diagonal
    correction.  Six signed PCA translations cover only a small residual
    fraction of the partial diagonal; they are symmetric, fixed, and therefore
    independent of category, object frame, or ground truth.
    """
    partial = np.asarray(partial, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    partial_diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    prior_diagonal = max(float(np.linalg.norm(np.ptp(prior, axis=0))), 1e-8)
    nominal_scale = float(np.clip(partial_diagonal / prior_diagonal, .45, 2.40))
    partial_center = np.median(partial, axis=0)
    prior_center = np.median(prior, axis=0)
    before = projection_metrics(partial, prior, projector, partial_diagonal)
    before_energy = projection_energy(before)
    axes = _pca_axes(partial)
    candidates = []
    for scale_multiplier in (.85, 1.0, 1.15):
        scale = float(np.clip(nominal_scale * scale_multiplier, .45, 2.40))
        base_translation = partial_center - scale * prior_center
        shifts = [("center", np.zeros(3, dtype=np.float64))]
        for axis, vector in enumerate(axes):
            for sign in (-1.0, 1.0):
                shifts.append((f"pca{axis}_{sign:+.0f}", sign * .12 * partial_diagonal * vector))
        for name, shift in shifts:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] *= scale
            transform[:3, 3] = base_translation + shift
            transformed = apply_transform(prior, transform)
            metrics = projection_metrics(partial, transformed, projector, partial_diagonal)
            candidates.append({
                "name": f"s{scale_multiplier:.2f}_{name}", "transform": transform,
                "points": transformed, "metrics": metrics, "energy": projection_energy(metrics),
            })
    selected = min(candidates, key=lambda item: item["energy"])
    return selected["points"], selected["transform"], {
        "method": "camera1_projection_only_center_scale_proper_sim3",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "orientation_source": "unchanged_native_bridge_transform",
        "nominal_scale": nominal_scale,
        "scale_multipliers": [.85, 1.0, 1.15],
        "signed_pca_translation_ratio": .12,
        "before": {"metrics": before, "energy": before_energy},
        "selected": {
            "name": selected["name"], "metrics": selected["metrics"], "energy": selected["energy"],
            "projection_improved": bool(selected["energy"] < before_energy * .99),
        },
        "candidate_count": len(candidates),
    }
