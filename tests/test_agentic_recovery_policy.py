"""Tests for the no-GT recovery trigger and candidate selection."""

from __future__ import annotations

import math

from src.agentic_recovery_policy import (
    has_usable_visible_evidence,
    needs_global_recovery,
    needs_global_rescue_probe,
    prefer_rescue_over_refinement,
)


def _diagnostic(*, pairs: int, energy: float, silhouette: float = .4, coverage: float = .8, leakage: float = .3):
    return {
        "verifier_energy": energy,
        "registered_prior_vs_partial": {
            "visible_pair_count": pairs,
            "silhouette_iou": silhouette,
            "coverage": coverage,
            "leakage": leakage,
        },
    }


def test_zero_pair_or_nonfinite_route_triggers_recovery():
    assert needs_global_recovery(_diagnostic(pairs=0, energy=math.inf))
    assert needs_global_recovery(_diagnostic(pairs=5, energy=.2))


def test_finite_visible_route_is_left_unchanged():
    diagnostic = _diagnostic(pairs=6, energy=.2)
    assert has_usable_visible_evidence(diagnostic)
    assert not needs_global_recovery(diagnostic)
    assert not needs_global_rescue_probe(diagnostic)


def test_weak_global_evidence_only_probes_a_competing_candidate():
    weak = _diagnostic(pairs=900, energy=.45, silhouette=.12, coverage=.58, leakage=.85)
    assert has_usable_visible_evidence(weak)
    assert needs_global_rescue_probe(weak)


def test_refinement_is_not_kept_when_it_loses_visible_evidence():
    rescue = _diagnostic(pairs=1200, energy=.24)
    refined = _diagnostic(pairs=800, energy=.31)
    assert prefer_rescue_over_refinement(rescue, refined)
    assert not prefer_rescue_over_refinement(refined, rescue)
