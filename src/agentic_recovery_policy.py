"""Sample-agnostic verifier policy for bounded agentic recovery.

The policy deliberately consumes only the compact no-GT diagnostic emitted by
the probe runner.  It is not a learned category router: a finite visible
surface estimate is the sole condition that separates the unchanged fast path
from the more expensive global recovery tool.
"""

from __future__ import annotations

import math
from typing import Any


MIN_VISIBLE_PAIRS = 6
WEAK_GLOBAL_ENERGY = .40
WEAK_GLOBAL_SILHOUETTE = .30
WEAK_GLOBAL_COVERAGE = .70


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def visible_evidence(diagnostics: dict[str, Any]) -> dict[str, float | int]:
    """Extract the cross-dataset verifier fields with safe missing defaults."""
    visible = diagnostics.get("registered_prior_vs_partial") or {}
    return {
        "pairs": int(visible.get("visible_pair_count", 0) or 0),
        "energy": float(diagnostics.get("verifier_energy", math.inf)),
        "silhouette": float(visible.get("silhouette_iou", 0.0) or 0.0),
        "coverage": float(visible.get("coverage", 0.0) or 0.0),
        "leakage": float(visible.get("leakage", 1.0) or 1.0),
    }


def needs_global_recovery(diagnostics: dict[str, Any]) -> bool:
    """Return true only when the normal route has no usable visible evidence."""
    evidence = visible_evidence(diagnostics)
    return evidence["pairs"] < MIN_VISIBLE_PAIRS or not _finite(evidence["energy"])


def has_usable_visible_evidence(diagnostics: dict[str, Any]) -> bool:
    return not needs_global_recovery(diagnostics)


def needs_global_rescue_probe(diagnostics: dict[str, Any]) -> bool:
    """Decide whether a fixed global rescue should be *evaluated* as a rival.

    Besides true zero-overlap failures, a proposal with simultaneously high
    normalized verifier energy and a clearly unsupported silhouette/coverage
    is allowed to request one global candidate.  This does not force a pose
    change: the resulting candidate must still beat the original diagnostic
    before the residual solver is called.  The thresholds live in a
    dimensionless verifier space and are shared across all datasets.
    """
    if needs_global_recovery(diagnostics):
        return True
    evidence = visible_evidence(diagnostics)
    return bool(
        evidence["energy"] >= WEAK_GLOBAL_ENERGY
        and (evidence["silhouette"] <= WEAK_GLOBAL_SILHOUETTE
             or evidence["coverage"] <= WEAK_GLOBAL_COVERAGE)
    )


def evidence_quality(diagnostics: dict[str, Any]) -> tuple[float, int, float, float, float]:
    """Rank finite candidates using only the fixed verifier observations.

    Lower is better.  Pair count, silhouette and coverage break near-ties in
    favour of the candidate with more direct observed support; leakage is the
    final penalty.  This tuple is intentionally independent of class, GT and
    completion-model confidence.
    """
    evidence = visible_evidence(diagnostics)
    energy = evidence["energy"] if _finite(evidence["energy"]) else math.inf
    return (
        float(energy),
        -int(evidence["pairs"]),
        -float(evidence["silhouette"]),
        -float(evidence["coverage"]),
        float(evidence["leakage"]),
    )


def prefer_rescue_over_refinement(
    rescue_diagnostics: dict[str, Any], refinement_diagnostics: dict[str, Any],
) -> bool:
    """Reject a residual refinement if it loses the rescue's visible support."""
    if not has_usable_visible_evidence(refinement_diagnostics):
        return has_usable_visible_evidence(rescue_diagnostics)
    return evidence_quality(rescue_diagnostics) < evidence_quality(refinement_diagnostics)
