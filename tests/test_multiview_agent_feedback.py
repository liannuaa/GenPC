import numpy as np

from src.agent_completion_policy import accept_agent_proposal
from src.multiview_agent_feedback import (
    accept_multiview_no_harm,
    make_orthographic_reference,
    measure_multiview_evidence,
)


def _box_grid() -> np.ndarray:
    values = np.linspace(-1., 1., 13)
    xx, yy, zz = np.meshgrid(values, values, values, indexing="ij")
    shell = (np.isclose(np.abs(xx), 1.) | np.isclose(np.abs(yy), 1.) | np.isclose(np.abs(zz), 1.))
    return np.stack((xx[shell], yy[shell], zz[shell]), axis=1)


def test_fixed_multiview_reference_accepts_identity_and_rejects_large_shift():
    partial = _box_grid()
    anchor = partial.copy()
    reference = make_orthographic_reference(partial, anchor, size=96)
    before = measure_multiview_evidence(partial, anchor, reference)
    same = measure_multiview_evidence(partial, anchor.copy(), reference)
    shifted = measure_multiview_evidence(partial, anchor + np.array([.55, 0., 0.]), reference)
    assert accept_multiview_no_harm(before, same)
    assert not accept_multiview_no_harm(before, shifted)


def test_agent_gate_requires_saved_and_multiview_no_harm():
    score = {"objective": .2, "projection": {"coverage": .9, "iou": .8, "leakage": .1}}
    partial = _box_grid(); reference = make_orthographic_reference(partial, partial, size=96)
    anchor = measure_multiview_evidence(partial, partial, reference)
    shifted = measure_multiview_evidence(partial, partial + np.array([.55, 0., 0.]), reference)
    assert not accept_agent_proposal(score, score, anchor_multiview=anchor, proposal_multiview=shifted)
