"""True bidirectional visible-correspondence consensus for proper Sim(3)."""

from __future__ import annotations

import numpy as np

from src.bidirectional_cycle_registration import (
    cycle_errors,
    interpolate_sim3,
    invert_proper_sim3,
    robust_fit_similarity,
    visible_score,
)
from src.ray_consistent_registration import (
    apply_transform,
    bounded_delta_sim3,
    soft_ray_correspondences,
)


def _balanced_indices(length: int, count: int) -> np.ndarray:
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, count, dtype=np.int64)


def true_bidirectional_pairs(complete, partial, projector, *, diagonal,
                             pixel_radius, trim_quantile=.75,
                             max_distance_ratio=.14):
    """Build independent partial→prior and visible-prior→partial pairs."""
    forward = soft_ray_correspondences(
        partial, complete, projector, pixel_radius=pixel_radius,
        trim_quantile=trim_quantile, max_distance_ratio=max_distance_ratio,
        bbox_diagonal=diagonal)
    reverse = soft_ray_correspondences(
        complete, partial, projector, pixel_radius=pixel_radius,
        trim_quantile=trim_quantile, max_distance_ratio=max_distance_ratio,
        bbox_diagonal=diagonal)
    forward_pairs = np.c_[forward["partial_ids"], forward["generated_ids"]]
    # Normalize reverse pairs to the same (partial_id, complete_id) ordering.
    reverse_pairs = np.c_[reverse["generated_ids"], reverse["partial_ids"]]
    if len(forward_pairs) and len(reverse_pairs):
        forward_keys = {tuple(pair) for pair in forward_pairs.tolist()}
        mutual = np.asarray(
            [pair for pair in reverse_pairs.tolist() if tuple(pair) in forward_keys],
            dtype=np.int64).reshape(-1, 2)
    else:
        mutual = np.empty((0, 2), dtype=np.int64)
    return forward, reverse, forward_pairs, reverse_pairs, mutual


def _fit_pair_set(partial, complete, pairs):
    transform, kept = robust_fit_similarity(
        partial[pairs[:, 0]], complete[pairs[:, 1]])
    return transform, int(kept.sum())


def _pair_consensus_rms(forward_transform, partial, complete,
                        forward_pairs, reverse_pairs, diagonal):
    values = []
    for pairs in (forward_pairs, reverse_pairs):
        if len(pairs):
            residual = np.linalg.norm(
                apply_transform(partial[pairs[:, 0]], forward_transform)
                - complete[pairs[:, 1]], axis=1)
            values.append(float(np.sqrt(np.mean(residual * residual))))
    return float(np.mean(values) / max(float(diagonal), 1e-8)) if values else float("inf")


def bidirectional_consensus_step(
    complete, partial, projector, *, diagonal, pixel_radius=5.0,
    max_rotation_deg=3.0, scale_bounds=(0.96, 1.04),
    max_translation_ratio=0.03, min_pairs=96,
    fractions=(0.25, 0.5, 0.75, 1.0), max_cycle_ratio=0.03,
    return_best_candidate=False,
):
    """Select among forward, reverse, balanced, and reciprocal Sim(3) fits."""
    complete = np.asarray(complete, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    before = visible_score(partial, complete, projector, diagonal, pixel_radius)
    (forward_score, reverse_score, forward_pairs,
     reverse_pairs, mutual_pairs) = true_bidirectional_pairs(
        complete, partial, projector, diagonal=diagonal,
        pixel_radius=pixel_radius)
    if len(forward_pairs) < int(min_pairs) or len(reverse_pairs) < int(min_pairs):
        return complete, np.eye(4), {
            "accepted": False, "candidate_exposed_for_audit": False,
            "reason": "insufficient_true_bidirectional_pairs", "before": before,
            "forward_pair_count": int(len(forward_pairs)),
            "reverse_pair_count": int(len(reverse_pairs)),
            "mutual_pair_count": int(len(mutual_pairs)),
        }

    hypotheses = []
    forward_transform, forward_kept = _fit_pair_set(
        partial, complete, forward_pairs)
    hypotheses.append(("forward", forward_transform, forward_kept))
    reverse_transform, reverse_kept = robust_fit_similarity(
        complete[reverse_pairs[:, 1]], partial[reverse_pairs[:, 0]])
    hypotheses.append((
        "reverse", invert_proper_sim3(reverse_transform), int(reverse_kept.sum())))
    balanced_count = min(len(forward_pairs), len(reverse_pairs))
    balanced_pairs = np.concatenate((
        forward_pairs[_balanced_indices(len(forward_pairs), balanced_count)],
        reverse_pairs[_balanced_indices(len(reverse_pairs), balanced_count)]), axis=0)
    balanced_transform, balanced_kept = _fit_pair_set(
        partial, complete, balanced_pairs)
    hypotheses.append(("balanced", balanced_transform, balanced_kept))
    if len(mutual_pairs) >= int(min_pairs):
        mutual_transform, mutual_kept = _fit_pair_set(
            partial, complete, mutual_pairs)
        hypotheses.append(("mutual", mutual_transform, mutual_kept))

    candidates = []
    for hypothesis, raw_forward, kept_count in hypotheses:
        bounded_inverse = bounded_delta_sim3(
            invert_proper_sim3(raw_forward),
            max_rotation_deg=float(max_rotation_deg),
            scale_bounds=tuple(scale_bounds),
            max_translation=float(max_translation_ratio) * float(diagonal))
        for fraction in fractions:
            inverse_step = interpolate_sim3(bounded_inverse, fraction)
            forward_step = invert_proper_sim3(inverse_step)
            moved = apply_transform(complete, inverse_step)
            score = visible_score(
                partial, moved, projector, diagonal, pixel_radius)
            consensus_rms = _pair_consensus_rms(
                forward_step, partial, complete, forward_pairs,
                reverse_pairs, diagonal)
            # The consensus term is deliberately small: it breaks ties between
            # transformations that render similarly without overpowering the
            # saved-camera objective.
            joint_objective = float(score["objective"] + .08 * consensus_rms)
            candidates.append({
                "hypothesis": hypothesis, "kept_count": int(kept_count),
                "fraction": float(fraction), "inverse_step": inverse_step,
                "forward_step": forward_step, "moved": moved, "score": score,
                "pair_consensus_rms": consensus_rms,
                "joint_objective": joint_objective,
            })
    selected = min(candidates, key=lambda item: item["joint_objective"])
    # A genuinely independent reverse fit is now the witness.
    reverse_witness = bounded_delta_sim3(
        reverse_transform, max_rotation_deg=float(max_rotation_deg),
        scale_bounds=tuple(scale_bounds),
        max_translation=float(max_translation_ratio) * float(diagonal))
    cycle = cycle_errors(
        selected["forward_step"], selected["inverse_step"], reverse_witness,
        partial[forward_pairs[:, 0]], diagonal)
    projection_before = before["projection"]
    projection_after = selected["score"]["projection"]
    accepted = bool(
        np.isfinite(selected["joint_objective"])
        and selected["score"]["objective"] < before["objective"] * .9975
        and selected["score"]["geometric"]["objective"]
            <= before["geometric"]["objective"] * 1.002
        and projection_after["coverage"] >= projection_before["coverage"] - .02
        and projection_after["iou"] >= projection_before["iou"] - .01
        and cycle["independent_reverse_cycle_rms"] <= float(max_cycle_ratio)
        and cycle["exact_inverse_cycle_rms"] <= 1e-9)
    expose = bool(accepted or return_best_candidate)
    return (
        selected["moved"] if expose else complete,
        selected["inverse_step"] if expose else np.eye(4),
        {
            "accepted": accepted,
            "candidate_exposed_for_audit": bool(return_best_candidate and not accepted),
            "reason": "accepted" if accepted else "do_no_harm_gate",
            "before": before, "after": selected["score"], "cycle": cycle,
            "selected_hypothesis": selected["hypothesis"],
            "selected_fraction": selected["fraction"],
            "pair_consensus_rms": selected["pair_consensus_rms"],
            "forward_pair_count": int(len(forward_pairs)),
            "reverse_pair_count": int(len(reverse_pairs)),
            "mutual_pair_count": int(len(mutual_pairs)),
            "candidate_summary": [{
                "hypothesis": item["hypothesis"],
                "fraction": item["fraction"],
                "visible_objective": item["score"]["objective"],
                "pair_consensus_rms": item["pair_consensus_rms"],
                "joint_objective": item["joint_objective"],
            } for item in candidates],
            "forward_visible_score": forward_score,
            "reverse_visible_score": reverse_score,
        },
    )
