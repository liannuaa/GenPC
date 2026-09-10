"""No-GT policy for residual-guided multi-view image refinement.

The agent chooses only discrete actions.  Image generation and geometric
measurement remain external, reproducible tools.  Missing partial pixels are
never interpreted as negative evidence.
"""

from __future__ import annotations

import json
from pathlib import Path


VIEW_ORDER = ("front", "side", "back", "right")


def _coverage(report: dict, view: str) -> float:
    return float(report["views"][view]["edited"]["coverage_3px"])


def _support_pixels(report: dict, view: str) -> int:
    return int(report["views"][view]["edited"].get("support_pixels", 1))


def _p90_distance(report: dict, view: str) -> float:
    return float(
        report["views"][view]["edited"].get("outside_distance_p90_px", float("inf"))
    )


def _mean_distance(report: dict, view: str) -> float:
    return float(
        report["views"][view]["edited"].get("outside_distance_mean_px", float("inf"))
    )


def _residual_score(report: dict, view: str, tolerance_px: float) -> float:
    """One-sided positive-support residual; lower is better."""
    tolerance = max(float(tolerance_px), 1e-8)
    coverage_term = 1.0 - _coverage(report, view)
    mean_term = min(_mean_distance(report, view) / tolerance, 2.0)
    p90_term = min(_p90_distance(report, view) / tolerance, 2.0)
    return float(0.50 * coverage_term + 0.35 * mean_term + 0.15 * p90_term)


def decide_edit_action(
    *,
    incumbent_report: dict,
    candidate_report: dict,
    min_reliable_support_ratio: float = 0.50,
    high_coverage_threshold: float = 0.97,
    residual_p90_tolerance_px: float = 3.0,
    residual_coverage_floor: float = 0.93,
    per_view_regression_tolerance: float = 0.01,
    score_improvement_tolerance: float = 0.002,
    round_index: int = 1,
    max_rounds: int = 6,
) -> dict:
    """Select a no-GT per-view incumbent and the next discrete action.

    Views are judged by their own one-sided observed-support residual.  A view
    with little physical support is recorded as uncertain instead of being
    forced to match a better-observed reference view.
    """
    incumbent = {name: _coverage(incumbent_report, name) for name in VIEW_ORDER}
    candidate = {name: _coverage(candidate_report, name) for name in VIEW_ORDER}
    regressions = {
        name: candidate[name] - incumbent[name] for name in VIEW_ORDER
    }
    incumbent_score = {
        name: _residual_score(incumbent_report, name, residual_p90_tolerance_px)
        for name in VIEW_ORDER
    }
    candidate_score = {
        name: _residual_score(candidate_report, name, residual_p90_tolerance_px)
        for name in VIEW_ORDER
    }
    selected_source_by_view = {"front": "incumbent"}
    for name in VIEW_ORDER[1:]:
        score_gain = incumbent_score[name] - candidate_score[name]
        selected_source_by_view[name] = (
            "candidate"
            if (
                score_gain >= score_improvement_tolerance
                and regressions[name] >= -per_view_regression_tolerance
            )
            else "incumbent"
        )
    selected = {
        name: (
            candidate[name]
            if selected_source_by_view[name] == "candidate"
            else incumbent[name]
        )
        for name in VIEW_ORDER
    }
    accepted_candidate_views = [
        name for name in VIEW_ORDER if selected_source_by_view[name] == "candidate"
    ]
    candidate_selected = len(accepted_candidate_views) == len(VIEW_ORDER) - 1
    max_support = max(
        max(_support_pixels(incumbent_report, name) for name in VIEW_ORDER), 1,
    )
    support_ratio = {
        name: min(1.0, _support_pixels(incumbent_report, name) / max_support)
        for name in VIEW_ORDER
    }
    reliable = {
        name: support_ratio[name] >= min_reliable_support_ratio for name in VIEW_ORDER
    }
    selected_report_by_view = {
        name: (
            candidate_report
            if selected_source_by_view[name] == "candidate"
            else incumbent_report
        )
        for name in VIEW_ORDER
    }
    p90_by_view = {
        name: _p90_distance(selected_report_by_view[name], name)
        for name in VIEW_ORDER
    }
    mean_distance_by_view = {
        name: _mean_distance(selected_report_by_view[name], name)
        for name in VIEW_ORDER
    }
    pass_by_view = {
        name: bool(
            not reliable[name]
            or selected[name] >= high_coverage_threshold
            or (selected[name] >= residual_coverage_floor
                and p90_by_view[name] <= residual_p90_tolerance_px)
        )
        for name in VIEW_ORDER
    }
    failing = [name for name in VIEW_ORDER if reliable[name] and not pass_by_view[name]]

    incumbent_weighted_score = sum(
        support_ratio[name] * incumbent_score[name] for name in VIEW_ORDER
    ) / max(sum(support_ratio.values()), 1e-8)
    selected_score = {
        name: (
            candidate_score[name]
            if selected_source_by_view[name] == "candidate"
            else incumbent_score[name]
        )
        for name in VIEW_ORDER
    }
    selected_weighted_score = sum(
        support_ratio[name] * selected_score[name] for name in VIEW_ORDER
    ) / max(sum(support_ratio.values()), 1e-8)

    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if round_index < 1:
        raise ValueError("round_index must be positive")
    if not failing:
        action = "ACCEPT"
    elif round_index >= max_rounds:
        action = "STOP_BUDGET_KEEP_BEST"
    elif accepted_candidate_views:
        action = "COMPOSE_HYBRID_AND_REFINE_VIEWS"
    else:
        action = "ROLLBACK_AND_REFINE_VIEWS"
    return {
        "policy": "visibility_weighted_one_sided_positive_support_loop",
        "ground_truth_used": False,
        "missing_partial_pixels_are_unknown": True,
        "action": action,
        "selected": (
            "candidate" if candidate_selected else
            "hybrid" if accepted_candidate_views else "incumbent"
        ),
        "selected_source_by_view": selected_source_by_view,
        "accepted_candidate_views": accepted_candidate_views,
        "min_reliable_support_ratio": min_reliable_support_ratio,
        "high_coverage_threshold": high_coverage_threshold,
        "residual_p90_tolerance_px": residual_p90_tolerance_px,
        "residual_coverage_floor": residual_coverage_floor,
        "score_improvement_tolerance": score_improvement_tolerance,
        "round_index": int(round_index),
        "max_rounds": int(max_rounds),
        "best_so_far_is_per_view": True,
        "support_ratio_to_best_observed_view": support_ratio,
        "reliable_view": reliable,
        "uncertain_low_support_views": [
            name for name in VIEW_ORDER if not reliable[name]
        ],
        "selected_outside_distance_p90_px": p90_by_view,
        "selected_outside_distance_mean_px": mean_distance_by_view,
        "pass_by_view": pass_by_view,
        "failing_views": failing,
        "incumbent_coverage_3px": incumbent,
        "candidate_coverage_3px": candidate,
        "selected_coverage_3px": selected,
        "candidate_minus_incumbent": regressions,
        "incumbent_residual_score": incumbent_score,
        "candidate_residual_score": candidate_score,
        "selected_residual_score": selected_score,
        "incumbent_visibility_weighted_residual": incumbent_weighted_score,
        "selected_visibility_weighted_residual": selected_weighted_score,
    }


def decide_from_files(
    incumbent_path: Path,
    candidate_path: Path,
    output_path: Path,
    **kwargs: float,
) -> dict:
    incumbent = json.loads(Path(incumbent_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    decision = decide_edit_action(
        incumbent_report=incumbent,
        candidate_report=candidate,
        **kwargs,
    )
    decision["incumbent_report"] = str(Path(incumbent_path).resolve())
    decision["candidate_report"] = str(Path(candidate_path).resolve())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision
