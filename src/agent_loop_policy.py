"""Cross-round state policy for the bounded GenPC+ agent loop.

Per-action gates protect a single transition, but a sequence of individually
tolerated moves can still drift.  This module compares every successor to the
immutable initial state in the same saved and PCA-frame evidence, then selects
the best safe state by its observed geometric residual.  It contains no model,
dataset, category, or ground-truth dependency.
"""

from __future__ import annotations

from typing import Mapping

from src.agent_completion_policy import accept_registration_refinement
from src.multiview_agent_feedback import MultiViewEvidence, accept_multiview_no_harm


def relative_geometric_gain(before: Mapping, after: Mapping) -> float:
    """Positive when the observed 3-D residual decreases."""
    base = max(float(before["geometric"]["objective"]), 1e-12)
    return (base - float(after["geometric"]["objective"])) / base


def accepted_against_initial(
    initial_visible: Mapping,
    candidate_visible: Mapping,
    initial_multiview: MultiViewEvidence,
    candidate_multiview: MultiViewEvidence,
) -> bool:
    """Prevent cumulative action drift by always comparing to round zero."""
    return bool(
        accept_registration_refinement(initial_visible, candidate_visible)
        and accept_multiview_no_harm(initial_multiview, candidate_multiview)
    )


def prefers_candidate(current_best: Mapping, candidate: Mapping) -> bool:
    """Rank safe states by 3-D observed conformance, then saved-view score."""
    current_key = (float(current_best["geometric"]["objective"]),
                   float(current_best["objective"]))
    candidate_key = (float(candidate["geometric"]["objective"]),
                     float(candidate["objective"]))
    return candidate_key < current_key


def should_probe_generative_action(trace: list[Mapping]) -> bool:
    """Permit one high-risk text/3-D probe only after a safe local phase.

    The controller's per-round local action deliberately has priority over a
    global re-decode.  This predicate makes the hand-off explicit: a text/3-D
    edit is useful as a *different* action after at least one accepted,
    re-observed intrinsic correction, rather than as a competing edit in the
    same causal step.  It is independent of object identity and metrics.
    """
    return any(
        bool(state.get("action_accepted"))
        and state.get("route") == "accepted_intrinsic_local_geometry"
        for state in trace
    )
