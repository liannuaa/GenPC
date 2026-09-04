from src.openai_vlm_feedback import approved_by_vlm


def test_vlm_gate_is_conservative_when_absent_or_negative():
    assert not approved_by_vlm(None)
    assert not approved_by_vlm({"approve_edit": False, "visible_issue": "extent"})
    assert not approved_by_vlm({"approve_edit": True, "visible_issue": "none"})
    assert approved_by_vlm({"approve_edit": True, "visible_issue": "surface"})
