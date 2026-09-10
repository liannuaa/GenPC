"""Bidirectional, cycle-checked 2D+3D residual proper-Sim(3) registration.

The forward problem maps observed partial points into the currently visible
region of a frozen complete prior.  Only the exact inverse of that forward
proper Sim(3) is allowed to move the complete prior back toward the partial.
An independently fitted reverse map is used as a consistency witness, never
as the applied transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from src.ray_consistent_registration import (
    apply_transform,
    bounded_delta_sim3,
    fit_similarity_umeyama,
    soft_ray_correspondences,
    zbuffer_indices,
)


@dataclass(frozen=True)
class VisibleTargetCache:
    """Saved-camera quantities that are invariant across Sim(3) candidates."""

    correspondence_depth: np.ndarray
    correspondence_point_depth: np.ndarray
    correspondence_index: np.ndarray
    correspondence_y: np.ndarray
    correspondence_x: np.ndarray
    correspondence_tree: object | None
    projection_depth: np.ndarray
    projection_raw_mask: np.ndarray
    projection_mask: np.ndarray
    dilation: int


def prepare_visible_target(partial, projector, *, dilation: int = 1) -> VisibleTargetCache:
    """Render/build the fixed partial-side evidence once for a candidate sweep."""
    partial = np.asarray(partial, dtype=np.float64)
    partial_uv, partial_depth = projector.project(partial)
    correspondence_depth, correspondence_mask, correspondence_index = zbuffer_indices(
        partial_uv, partial_depth, projector.image_shape,
    )
    correspondence_y, correspondence_x = np.where(correspondence_mask)
    correspondence_tree = (
        cKDTree(np.c_[correspondence_x, correspondence_y])
        if len(correspondence_x) else None
    )
    projection_depth = _zbuffer(projector, partial)
    projection_raw_mask = np.isfinite(projection_depth)
    if int(dilation) > 0:
        structure = np.ones((2 * int(dilation) + 1,) * 2, dtype=bool)
        projection_mask = binary_dilation(projection_raw_mask, structure=structure)
    else:
        projection_mask = projection_raw_mask
    return VisibleTargetCache(
        correspondence_depth=correspondence_depth,
        correspondence_point_depth=partial_depth,
        correspondence_index=correspondence_index,
        correspondence_y=correspondence_y,
        correspondence_x=correspondence_x,
        correspondence_tree=correspondence_tree,
        projection_depth=projection_depth,
        projection_raw_mask=projection_raw_mask,
        projection_mask=projection_mask,
        dilation=int(dilation),
    )


def sim3_parts(transform):
    """Return isotropic scale, proper rotation, and translation or fail."""
    transform = np.asarray(transform, dtype=np.float64)
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("Transform is not a proper Sim(3)")
    scale = float(np.cbrt(determinant))
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Transform contains anisotropic scale or shear")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("Transform contains a reflection")
    return scale, rotation, transform[:3, 3].copy()


def invert_proper_sim3(transform):
    """Analytic inverse that preserves the proper isotropic Sim(3) contract."""
    scale, rotation, translation = sim3_parts(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T / scale
    inverse[:3, 3] = -(rotation.T @ translation) / scale
    return inverse


def interpolate_sim3(transform, fraction):
    """Interpolate from identity in Sim(3), including translation."""
    scale, rotation, translation = sim3_parts(transform)
    fraction = float(fraction)
    step = np.eye(4, dtype=np.float64)
    step[:3, :3] = (scale ** fraction) * Rotation.from_rotvec(
        fraction * Rotation.from_matrix(rotation).as_rotvec()
    ).as_matrix()
    step[:3, 3] = fraction * translation
    return step


def robust_fit_similarity(source, target, trim_quantile=0.75, iterations=3):
    """Shared trimmed Umeyama fit; no category or sample-dependent choices."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 6 or len(target) != len(source):
        raise ValueError("At least six paired points are required")
    kept = np.ones(len(source), dtype=bool)
    transform = fit_similarity_umeyama(source, target)
    for _ in range(int(iterations)):
        residual = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
        limit = float(np.quantile(residual, float(trim_quantile)))
        next_kept = residual <= max(limit, 1e-12)
        if next_kept.sum() < 6:
            break
        kept = next_kept
        transform = fit_similarity_umeyama(source[kept], target[kept])
    return transform, kept


def _zbuffer(projector, points):
    """Vectorized saved-camera z-buffer used for 2D silhouette/depth evidence."""
    uv, depth = projector.project(np.asarray(points, dtype=np.float64))
    height, width = map(int, projector.image_shape)
    xy = np.rint(uv).astype(np.int64)
    valid = (
        np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
        & (xy[:, 0] >= 0) & (xy[:, 0] < width)
        & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    )
    xy = xy[valid]
    depth = np.asarray(depth, dtype=np.float64)[valid]
    # A deliberately broad Sim(3) proposal may place the whole complete prior
    # outside Camera-1.  That candidate should receive no silhouette/depth
    # support, rather than making the candidate scorer fail before it can rank
    # the remaining proposals.
    if len(depth) == 0:
        return np.full((height, width), np.inf, dtype=np.float64)
    flat = xy[:, 1] * width + xy[:, 0]
    order = np.lexsort((depth, flat))
    flat_sorted = flat[order]
    first = np.r_[True, flat_sorted[1:] != flat_sorted[:-1]]
    selected_flat = flat_sorted[first]
    selected_depth = depth[order][first]
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    zbuffer[selected_flat] = selected_depth
    return zbuffer.reshape(height, width)


def projection_metrics(partial, complete, projector, diagonal, dilation=1, *,
                       target_cache: VisibleTargetCache | None = None):
    """Saved-camera silhouette overlap and visible depth agreement."""
    if target_cache is None:
        target_cache = prepare_visible_target(partial, projector, dilation=int(dilation))
    elif target_cache.dilation != int(dilation):
        raise ValueError("visible target cache dilation does not match the score")
    partial_depth = target_cache.projection_depth
    complete_depth = _zbuffer(projector, complete)
    partial_raw = target_cache.projection_raw_mask
    complete_raw = np.isfinite(complete_depth)
    partial_mask = target_cache.projection_mask
    if int(dilation) > 0:
        structure = np.ones((2 * int(dilation) + 1,) * 2, dtype=bool)
        complete_mask = binary_dilation(complete_raw, structure=structure)
    else:
        complete_mask = complete_raw
    intersection = int(np.count_nonzero(partial_mask & complete_mask))
    union = int(np.count_nonzero(partial_mask | complete_mask))
    partial_count = int(np.count_nonzero(partial_mask))
    complete_count = int(np.count_nonzero(complete_mask))
    exact_overlap = partial_raw & complete_raw
    if np.any(exact_overlap):
        depth_error = np.abs(partial_depth[exact_overlap] - complete_depth[exact_overlap])
        depth_mean = float(np.mean(np.minimum(depth_error, 0.20 * diagonal)))
    else:
        depth_mean = float("inf")
    return {
        "iou": float(intersection / max(union, 1)),
        "coverage": float(intersection / max(partial_count, 1)),
        "leakage": float((complete_count - intersection) / max(complete_count, 1)),
        "visible_depth_mean": depth_mean,
        "visible_depth_normalized": float(depth_mean / max(diagonal, 1e-8)),
    }


def visible_score(partial, complete, projector, diagonal, pixel_radius=5.0, *,
                  target_cache: VisibleTargetCache | None = None):
    """One dimensionless GT-free objective joining visible 2D and 3D cues."""
    if target_cache is None:
        target_cache = prepare_visible_target(partial, projector)
    pairs = soft_ray_correspondences(
        partial, complete, projector, pixel_radius=float(pixel_radius),
        trim_quantile=0.75, max_distance_ratio=0.14, partial_cache=target_cache,
        bbox_diagonal=float(diagonal),
    )
    projection = projection_metrics(
        partial, complete, projector, diagonal, target_cache=target_cache,
    )
    geometric = float(pairs["objective"] / max(diagonal, 1e-8))
    objective = (
        geometric
        + 0.20 * (1.0 - projection["iou"])
        + 0.10 * (1.0 - projection["coverage"])
        + 0.15 * projection["leakage"]
        + 0.25 * min(projection["visible_depth_normalized"], 0.20)
    )
    return {"objective": float(objective), "geometric": pairs,
            "projection": projection}


def cycle_errors(forward, applied_inverse, independent_reverse, points, diagonal):
    """Report exact algebraic cycle and independent reverse-fit disagreement."""
    points = np.asarray(points, dtype=np.float64)
    exact = apply_transform(apply_transform(points, forward), applied_inverse)
    witnessed = apply_transform(apply_transform(points, forward), independent_reverse)
    denominator = max(float(diagonal), 1e-8)
    return {
        "exact_inverse_cycle_rms": float(
            np.sqrt(np.mean(np.sum((exact - points) ** 2, axis=1))) / denominator),
        "independent_reverse_cycle_rms": float(
            np.sqrt(np.mean(np.sum((witnessed - points) ** 2, axis=1))) / denominator),
    }


def bidirectional_cycle_step(
    complete, partial, projector, *, diagonal, pixel_radius=5.0,
    max_rotation_deg=3.0, scale_bounds=(0.96, 1.04),
    max_translation_ratio=0.03, min_pairs=96,
    fractions=(0.25, 0.5, 0.75, 1.0), max_cycle_ratio=0.03,
    return_best_candidate=False,
):
    """Fit partial->visible-Pixal, then move Pixal only by its strict inverse."""
    complete = np.asarray(complete, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    target_cache = prepare_visible_target(partial, projector)
    before = visible_score(partial, complete, projector, diagonal, pixel_radius, target_cache=target_cache)
    pairs = before["geometric"]
    if len(pairs["partial_ids"]) < int(min_pairs):
        return complete, np.eye(4), {
            "accepted": False, "reason": "insufficient_visible_pairs", "before": before,
            "pair_count": int(len(pairs["partial_ids"])),
        }

    observed = partial[pairs["partial_ids"]]
    prior_visible = complete[pairs["generated_ids"]]
    raw_forward, forward_kept = robust_fit_similarity(observed, prior_visible)
    raw_inverse = invert_proper_sim3(raw_forward)
    bounded_inverse = bounded_delta_sim3(
        raw_inverse, max_rotation_deg=float(max_rotation_deg),
        scale_bounds=tuple(scale_bounds),
        max_translation=float(max_translation_ratio) * float(diagonal),
    )
    bounded_forward = invert_proper_sim3(bounded_inverse)
    independent_reverse, reverse_kept = robust_fit_similarity(prior_visible, observed)
    independent_reverse = bounded_delta_sim3(
        independent_reverse, max_rotation_deg=float(max_rotation_deg),
        scale_bounds=tuple(scale_bounds),
        max_translation=float(max_translation_ratio) * float(diagonal),
    )

    candidates = []
    for fraction in fractions:
        inverse_step = interpolate_sim3(bounded_inverse, fraction)
        forward_step = invert_proper_sim3(inverse_step)
        reverse_witness = interpolate_sim3(independent_reverse, fraction)
        moved = apply_transform(complete, inverse_step)
        score = visible_score(partial, moved, projector, diagonal, pixel_radius, target_cache=target_cache)
        cycle = cycle_errors(
            forward_step, inverse_step, reverse_witness, observed, diagonal)
        objective = score["objective"] + 0.20 * cycle["independent_reverse_cycle_rms"]
        candidates.append({
            "fraction": float(fraction), "inverse_step": inverse_step,
            "forward_step": forward_step, "moved": moved, "score": score,
            "cycle": cycle, "cycle_augmented_objective": float(objective),
        })
    selected = min(candidates, key=lambda item: item["cycle_augmented_objective"])
    projection_before = before["projection"]
    projection_after = selected["score"]["projection"]
    accepted = bool(
        np.isfinite(selected["cycle_augmented_objective"])
        and selected["cycle_augmented_objective"] < before["objective"] * 0.9975
        and selected["score"]["geometric"]["objective"]
            <= before["geometric"]["objective"] * 1.002
        and projection_after["coverage"] >= projection_before["coverage"] - 0.02
        and projection_after["iou"] >= projection_before["iou"] - 0.01
        and selected["cycle"]["independent_reverse_cycle_rms"]
            <= float(max_cycle_ratio)
        and selected["cycle"]["exact_inverse_cycle_rms"] <= 1e-9
    )
    expose_candidate = bool(accepted or return_best_candidate)
    return (
        selected["moved"] if expose_candidate else complete,
        selected["inverse_step"] if expose_candidate else np.eye(4),
        {
            "accepted": accepted,
            "candidate_exposed_for_audit": bool(return_best_candidate and not accepted),
            "reason": "accepted" if accepted else "do_no_harm_gate",
            "before": before,
            "after": selected["score"],
            "pair_count": int(len(observed)),
            "forward_trimmed_count": int(forward_kept.sum()),
            "reverse_trimmed_count": int(reverse_kept.sum()),
            "raw_forward_partial_to_pixal": raw_forward,
            "bounded_forward_partial_to_pixal": bounded_forward,
            "applied_strict_inverse_pixal_to_partial": selected["inverse_step"],
            "independent_reverse_witness": independent_reverse,
            "selected_fraction": selected["fraction"],
            "cycle": selected["cycle"],
            "candidate_summary": [{
                "fraction": item["fraction"],
                "visible_objective": item["score"]["objective"],
                "cycle_augmented_objective": item["cycle_augmented_objective"],
                "cycle": item["cycle"],
            } for item in candidates],
        },
    )


def partial_to_prior_inverse_step(
    complete, partial, projector, *, diagonal, pixel_radius=5.0,
    max_rotation_deg=3.0, scale_bounds=(.96, 1.04),
    max_translation_ratio=.03, min_pairs=96,
    fractions=(.25, .5, .75, 1.), return_best_candidate=False,
):
    """Fit partial→visible prior and move the prior only by its exact inverse."""
    complete = np.asarray(complete, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    target_cache = prepare_visible_target(partial, projector)
    before = visible_score(partial, complete, projector, diagonal, pixel_radius, target_cache=target_cache)
    pairs = before["geometric"]
    if len(pairs["partial_ids"]) < int(min_pairs):
        return complete, np.eye(4), {"accepted": False,
            "reason": "insufficient_visible_pairs", "before": before,
            "pair_count": int(len(pairs["partial_ids"]))}
    observed = partial[pairs["partial_ids"]]
    prior_visible = complete[pairs["generated_ids"]]
    raw_forward, kept = robust_fit_similarity(observed, prior_visible)
    bounded_inverse = bounded_delta_sim3(
        invert_proper_sim3(raw_forward), max_rotation_deg=float(max_rotation_deg),
        scale_bounds=tuple(scale_bounds),
        max_translation=float(max_translation_ratio) * float(diagonal))
    candidates = []
    for fraction in fractions:
        step = interpolate_sim3(bounded_inverse, fraction)
        moved = apply_transform(complete, step)
        candidates.append({"fraction": float(fraction), "inverse_step": step,
                           "moved": moved,
                           "score": visible_score(partial, moved, projector, diagonal, pixel_radius,
                                                  target_cache=target_cache)})
    selected = min(candidates, key=lambda item: item["score"]["objective"])
    projection_before = before["projection"]
    projection_after = selected["score"]["projection"]
    accepted = bool(
        np.isfinite(selected["score"]["objective"])
        and selected["score"]["objective"] < before["objective"] * .9975
        and selected["score"]["geometric"]["objective"] <= before["geometric"]["objective"] * 1.002
        and projection_after["coverage"] >= projection_before["coverage"] - .02
        and projection_after["iou"] >= projection_before["iou"] - .01)
    exposed = bool(accepted or return_best_candidate)
    forward = invert_proper_sim3(selected["inverse_step"])
    exact = apply_transform(apply_transform(observed, forward), selected["inverse_step"])
    cycle = float(np.sqrt(np.mean(np.sum((exact - observed) ** 2, axis=1))) /
                  max(float(diagonal), 1e-12))
    return (selected["moved"] if exposed else complete,
            selected["inverse_step"] if exposed else np.eye(4), {
        "accepted": accepted,
        "candidate_exposed_for_audit": bool(return_best_candidate and not accepted),
        "reason": "accepted" if accepted else "do_no_harm_gate",
        "before": before, "after": selected["score"],
        "pair_count": int(len(observed)), "forward_trimmed_count": int(kept.sum()),
        "raw_forward_partial_to_pixal": raw_forward,
        "applied_strict_inverse_pixal_to_partial": selected["inverse_step"],
        "selected_fraction": selected["fraction"],
        "cycle": {"exact_inverse_cycle_rms": cycle, "independent_reverse_cycle_rms": 0.},
        "candidate_summary": [{"fraction": item["fraction"],
                                "visible_objective": item["score"]["objective"]}
                               for item in candidates],
    })
