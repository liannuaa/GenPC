from src.multiview_edit_agent import decide_edit_action


def _report(values, support=None):
    names = ("front", "side", "back", "right")
    support = support or (1000, 1000, 1000, 1000)
    return {
        "views": {
            name: {"edited": {
                "coverage_3px": value,
                "support_pixels": count,
                "outside_distance_mean_px": 1.0,
                "outside_distance_p90_px": 4.0,
            }}
            for name, value, count in zip(names, values, support)
        }
    }


def test_rejects_regression_and_requests_targeted_views():
    decision = decide_edit_action(
        incumbent_report=_report((0.99, 0.96, 0.91, 0.91)),
        candidate_report=_report((0.98, 0.95, 0.82, 0.86)),
    )
    assert decision["selected"] == "incumbent"
    assert decision["action"] == "ROLLBACK_AND_REFINE_VIEWS"
    assert decision["failing_views"] == ["side", "back", "right"]


def test_accepts_improved_views_and_rolls_back_regressed_view():
    decision = decide_edit_action(
        incumbent_report=_report((0.99, 0.96, 0.91, 0.91)),
        candidate_report=_report((0.99, 0.97, 0.85, 0.93)),
    )
    assert decision["selected"] == "hybrid"
    assert decision["selected_source_by_view"] == {
        "front": "incumbent",
        "side": "candidate",
        "back": "incumbent",
        "right": "candidate",
    }
    assert decision["action"] == "COMPOSE_HYBRID_AND_REFINE_VIEWS"


def test_accepts_when_each_reliable_view_has_low_residual():
    decision = decide_edit_action(
        incumbent_report=_report((0.99, 0.96, 0.94, 0.93)),
        candidate_report=_report((0.99, 0.98, 0.98, 0.98)),
    )
    assert decision["selected"] == "candidate"
    assert decision["action"] == "ACCEPT"
    assert decision["failing_views"] == []


def test_accepts_small_robust_residual_when_coverage_is_high():
    incumbent = _report((0.99, 0.96, 0.91, 0.91))
    candidate = _report((0.99, 0.96, 0.91, 0.94))
    candidate["views"]["right"]["edited"]["outside_distance_p90_px"] = 2.5
    decision = decide_edit_action(
        incumbent_report=incumbent,
        candidate_report=candidate,
    )
    assert decision["pass_by_view"]["right"] is True


def test_low_support_view_is_uncertain_not_forced_to_match_another_view():
    incumbent = _report((0.99, 0.96, 0.65, 0.94), support=(1000, 900, 200, 800))
    candidate = _report((0.99, 0.96, 0.65, 0.94), support=(1000, 900, 200, 800))
    candidate["views"]["right"]["edited"]["outside_distance_p90_px"] = 2.0
    decision = decide_edit_action(
        incumbent_report=incumbent,
        candidate_report=candidate,
    )
    assert decision["reliable_view"]["back"] is False
    assert "back" in decision["uncertain_low_support_views"]
    assert decision["pass_by_view"]["back"] is True


def test_six_round_budget_keeps_best_so_far():
    decision = decide_edit_action(
        incumbent_report=_report((0.90, 0.90, 0.90, 0.90)),
        candidate_report=_report((0.89, 0.89, 0.89, 0.89)),
        round_index=6,
        max_rounds=6,
    )
    assert decision["selected"] == "incumbent"
    assert decision["action"] == "STOP_BUDGET_KEEP_BEST"
    assert decision["best_so_far_is_per_view"] is True
