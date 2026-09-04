from src.agent_completion_policy import (
    ViewEvidence, accept_registration_refinement, decide_prior_action,
)


def test_preserves_anchor_without_proposal():
    anchor = ViewEvidence(.9, .95, .03, .02)
    decision = decide_prior_action(anchor, None)
    assert decision.action == "preserve_prior"
    assert decision.accepted


def test_accepts_only_safe_proposal():
    anchor = ViewEvidence(.9, .95, .03, .02)
    proposal = ViewEvidence(.91, .95, .03, .02)
    assert decide_prior_action(anchor, proposal).action == "regenerate_prior"
    unsafe = ViewEvidence(.99, .70, .20, .01)
    assert decide_prior_action(anchor, unsafe).action == "preserve_prior"


def test_registration_action_requires_saved_view_no_harm():
    before = {"objective": .2, "projection": {"coverage": .9, "iou": .8, "leakage": .1}}
    safe = {"objective": .199, "projection": {"coverage": .895, "iou": .795, "leakage": .115}}
    unsafe = {"objective": .199, "projection": {"coverage": .87, "iou": .795, "leakage": .115}}
    assert accept_registration_refinement(before, safe)
    assert not accept_registration_refinement(before, unsafe)
