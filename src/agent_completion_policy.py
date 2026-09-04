"""Small, auditable action policy for zero-shot completion agents.

The controller intentionally has a tiny action space.  It does not use a
language model, GT shape, category labels, CD, or EMD to route a case.  A
generated proposal becomes eligible only after its saved-view evidence is at
least as good as the trusted anchor within a shared tolerance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from src.multiview_agent_feedback import MultiViewEvidence, accept_multiview_no_harm


@dataclass(frozen=True)
class ViewEvidence:
    iou: float
    coverage: float
    leakage: float
    depth_error: float

    @property
    def score(self) -> float:
        return (float(self.iou) + .15 * float(self.coverage)
                - .45 * float(self.leakage) - .20 * float(self.depth_error))


@dataclass(frozen=True)
class AgentDecision:
    action: str
    accepted: bool
    reason: str
    anchor_score: float
    proposal_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def accept_registration_refinement(before: Mapping, after: Mapping, *,
                                  objective_ratio: float = 1.002,
                                  maximum_leakage_increase: float = .025) -> bool:
    """Shared no-harm gate for an agent's bounded registration action.

    The saved partial view must not lose silhouette support while the combined
    2D+3D objective stays within a small numerical tolerance of its anchor.
    """
    return bool(
        float(after["objective"]) <= objective_ratio * float(before["objective"])
        and float(after["projection"]["coverage"])
            >= float(before["projection"]["coverage"]) - .01
        and float(after["projection"]["iou"])
            >= float(before["projection"]["iou"]) - .01
        and float(after["projection"]["leakage"])
            <= float(before["projection"]["leakage"]) + maximum_leakage_increase
    )


def accept_agent_proposal(
    before: Mapping,
    after: Mapping,
    *,
    anchor_multiview: MultiViewEvidence | None = None,
    proposal_multiview: MultiViewEvidence | None = None,
) -> bool:
    """Shared gate for a generative action re-registered into a trusted frame.

    If the caller provides a fixed three-view audit, both the saved-camera and
    orthographic evidence must pass.  Keeping the latter optional preserves
    compatibility for purely rigid screen refinements, which cannot create a
    new shape outside the saved view.
    """
    if not accept_registration_refinement(before, after):
        return False
    if (anchor_multiview is None) != (proposal_multiview is None):
        raise ValueError("multi-view evidence must be supplied for both anchor and proposal")
    return (anchor_multiview is None or accept_multiview_no_harm(
        anchor_multiview, proposal_multiview
    ))


def decide_prior_action(
    anchor: ViewEvidence,
    proposal: ViewEvidence | None,
    *,
    score_tolerance: float = .005,
    minimum_coverage: float = .90,
    maximum_leakage: float = .08,
) -> AgentDecision:
    """Route between preserve and regenerate with a shared no-harm guard."""
    anchor_score = anchor.score
    if proposal is None:
        return AgentDecision(
            action="preserve_prior", accepted=True,
            reason="no_regeneration_proposal", anchor_score=anchor_score,
            proposal_score=None,
        )
    proposal_score = proposal.score
    eligible = bool(
        proposal.coverage >= minimum_coverage
        and proposal.leakage <= maximum_leakage
        and proposal_score >= anchor_score - score_tolerance
    )
    return AgentDecision(
        action="regenerate_prior" if eligible else "preserve_prior",
        accepted=eligible,
        reason="proposal_passed_saved_view_guard" if eligible else "proposal_failed_saved_view_guard",
        anchor_score=anchor_score,
        proposal_score=proposal_score,
    )
