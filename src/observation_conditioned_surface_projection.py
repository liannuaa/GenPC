"""Observation-conditioned local surface projection with a fixed mass budget."""

from __future__ import annotations

import numpy as np
from fpsample import fps_sampling
from scipy.spatial import cKDTree

from src.bidirectional_cycle_registration import visible_score
from src.ray_consistent_registration import smooth_observation_absorption


def project_visible_surface_mass(body, partial, *, budget_ratio, seed):
    """Replace only prior mass nearest the observation with uniform partial mass."""
    body = np.asarray(body, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    budget = min(len(partial), int(round(float(budget_ratio) * len(body))))
    if budget < 32:
        return body.copy(), {
            "valid": False, "reason": "insufficient_budget", "budget": int(budget)}
    # FPS prevents the scan-line density of the sensor from dominating the
    # completed point distribution.
    target_ids = fps_sampling(
        partial.astype(np.float32), budget,
        start_idx=int(seed) % len(partial))
    targets = partial[np.asarray(target_ids, dtype=np.int64)]
    distance = cKDTree(partial).query(body, k=1, workers=-1)[0]
    removed_ids = np.argpartition(distance, budget - 1)[:budget]
    keep = np.ones(len(body), dtype=bool); keep[removed_ids] = False
    projected = np.concatenate((body[keep], targets), axis=0)
    return projected, {
        "valid": True,
        "reason": "uniform_observed_surface_mass_projection",
        "budget": int(budget),
        "budget_ratio": float(budget / len(body)),
        "complete_prior_points_preserved": int(keep.sum()),
        "complete_prior_fraction_preserved": float(keep.mean()),
        "exact_partial_points_inserted": int(len(targets)),
        "output_points": int(len(projected)),
        "removed_prior_distance_q95": float(np.quantile(distance[removed_ids], .95)),
    }


def select_observation_conditioned_postprocess(
    body, partial, projector, *, diagonal, seed=6145,
    mass_budgets=(.04, .08, .12),
    absorption_settings=((.04, .025), (.06, .040)),
    prior_mass_penalty=.008,
):
    """Route between identity, smooth absorption, and bounded mass projection."""
    body = np.asarray(body, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    before = visible_score(partial, body, projector, diagonal, pixel_radius=5.)
    candidates = [{
        "route": "identity", "body": body.copy(), "score": before,
        "selection_objective": float(before["objective"]),
        "info": {"complete_prior_fraction_preserved": 1.0},
    }]
    for influence_ratio, displacement_ratio in absorption_settings:
        moved, info = smooth_observation_absorption(
            body, partial, projector, bbox_diagonal=diagonal,
            pixel_radius=5., influence_ratio=float(influence_ratio),
            max_displacement_ratio=float(displacement_ratio), seed=int(seed))
        score = visible_score(partial, moved, projector, diagonal, pixel_radius=5.)
        candidates.append({
            "route": "smooth_absorption", "body": moved, "score": score,
            "selection_objective": float(score["objective"]),
            "info": {**info, "influence_ratio": float(influence_ratio),
                     "max_displacement_ratio": float(displacement_ratio),
                     "complete_prior_fraction_preserved": 1.0},
        })
    for budget_ratio in mass_budgets:
        moved, info = project_visible_surface_mass(
            body, partial, budget_ratio=float(budget_ratio), seed=int(seed))
        if not info.get("valid", False):
            continue
        score = visible_score(partial, moved, projector, diagonal, pixel_radius=5.)
        # A small explicit prior-mass cost chooses the minimum observation mass
        # needed to explain the scan instead of always saturating the budget.
        objective = float(score["objective"] + float(prior_mass_penalty) * budget_ratio)
        candidates.append({
            "route": "surface_mass_projection", "body": moved, "score": score,
            "selection_objective": objective, "info": info,
        })
    valid = [item for item in candidates if (
        np.isfinite(item["selection_objective"])
        and item["score"]["projection"]["coverage"]
            >= before["projection"]["coverage"] - .015
        and item["score"]["projection"]["iou"]
            >= before["projection"]["iou"] - .010)]
    selected = min(valid or candidates[:1], key=lambda item: item["selection_objective"])
    accepted = bool(
        selected["route"] != "identity"
        and selected["score"]["objective"] <= .99 * before["objective"])
    if not accepted:
        selected = candidates[0]
    return selected["body"], {
        "accepted": accepted,
        "selected_route": selected["route"],
        "before": before,
        "after": selected["score"],
        "selected_info": selected["info"],
        "candidate_summary": [{
            "route": item["route"],
            "selection_objective": item["selection_objective"],
            "visible_objective": item["score"]["objective"],
            "coverage": item["score"]["projection"]["coverage"],
            "iou": item["score"]["projection"]["iou"],
            "info": item["info"],
        } for item in candidates],
        "input_points": int(len(body)),
        "output_points": int(len(selected["body"])),
        "points_count_preserved": len(selected["body"]) == len(body),
    }
