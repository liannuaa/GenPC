"""GT-free guard for choosing between two complete-prior registrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorEvidence:
    objective: float
    scale: float
    accepted: bool


def choose_prior(
    current: PriorEvidence,
    fallback: PriorEvidence,
    *,
    min_scale_disagreement: float = 0.06,
    objective_ratio: float = 0.80,
) -> tuple[str, str]:
    """Prefer current unless observable shape/scale conflict supports fallback.

    Opposite scale directions expose a shape-scale ambiguity: one prior must
    shrink while the other must grow to explain the same partial observation.
    A fallback is allowed only when it also passes its registration guard and
    improves the shared visible objective by a substantial margin.
    """
    scale_disagreement = abs(fallback.scale / current.scale - 1.0)
    opposite_directions = (current.scale - 1.0) * (fallback.scale - 1.0) < 0.0
    fallback_supported = (
        fallback.accepted
        and not current.accepted
        and fallback.objective <= objective_ratio * current.objective
    )
    if (
        opposite_directions
        and scale_disagreement >= min_scale_disagreement
        and fallback_supported
    ):
        return "fallback", "observable_shape_scale_conflict"
    return "current", "current_prior_default"
