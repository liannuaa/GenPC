"""Direct visible-surface Sim(3) residuals from saved-view pixel indices.

The MoGe-to-partial bridge is effective because it does not stop at a 2-D
silhouette match: a pixel transfer identifies *visible 3-D point pairs*, and
a robust proper Sim(3) is fitted on those pairs.  This module applies the
same principle to a registered Pixal prior and the hard partial scan in their
shared saved Camera-1 view.

It is intentionally a residual method.  Camera-2 Pixal--MoGe evidence can be
recorded by the caller as a diagnostic, while the hard partial remains the
final Camera-1 alignment target.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from src.bidirectional_cycle_registration import interpolate_sim3, prepare_visible_target, visible_score
from src.indexed_pixel_sim3 import robust_indexed_sim3, unique_pixel_matches
from src.ray_consistent_registration import apply_transform, bounded_delta_sim3
from src.zbuffer import zbuffer_depth_with_indices


def visible_pixel_pairs(
    partial: np.ndarray,
    prior: np.ndarray,
    projector,
    *,
    max_pixel_distance: float = 2.0,
    max_pairs: int = 10_000,
) -> tuple[np.ndarray, dict]:
    """Return deterministic partial/prior pairs from Camera-1 visible pixels.

    The partial scan gets one exact z-buffer sample per raster cell.  The
    complete prior is z-buffered with a one-pixel splat to avoid artificial
    holes due to different point sampling densities.  Each partial cell then
    selects its nearest visible prior cell within a small image radius.  A
    mutual reduction keeps the closest partial claimant for every prior point,
    matching the indexed-pair contract used by the MoGe bridge.
    """
    partial, prior = (np.asarray(value, dtype=np.float64) for value in (partial, prior))
    partial_uv, partial_depth = projector.project(partial)
    prior_uv, prior_depth = projector.project(prior)
    _, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=0
    )
    _, prior_mask, prior_index = zbuffer_depth_with_indices(
        prior_uv, prior_depth, projector.image_shape, splat_radius=1
    )
    py, px = np.where(partial_mask)
    qy, qx = np.where(prior_mask)
    if len(px) < 6 or len(qx) < 6:
        raise ValueError("insufficient visible Camera-1 pixels for a direct residual")
    pixel_distance, nearest = cKDTree(np.c_[qx, qy]).query(np.c_[px, py], k=1)
    keep = pixel_distance <= float(max_pixel_distance)
    if int(keep.sum()) < 6:
        raise ValueError("insufficient nearby visible Camera-1 pixel pairs")
    partial_ids = partial_index[py[keep], px[keep]]
    prior_ids = prior_index[qy[nearest[keep]], qx[nearest[keep]]]
    pairs = np.column_stack((partial_ids, prior_ids, prior_ids, pixel_distance[keep]))
    pairs = unique_pixel_matches(pairs)
    if len(pairs) > int(max_pairs):
        pairs = pairs[np.linspace(0, len(pairs) - 1, int(max_pairs), dtype=np.int64)]
    if len(pairs) < 6:
        raise ValueError("mutual visible Camera-1 pairs are insufficient")
    return pairs.astype(np.float64), {
        "partial_visible_pixels": int(len(px)),
        "prior_visible_pixels": int(len(qx)),
        "nearby_pixel_pairs": int(keep.sum()),
        "mutual_pixel_pairs": int(len(pairs)),
        "max_pixel_distance": float(max_pixel_distance),
        "pair_subsample_limit": int(max_pairs),
    }


def pixel_pair_residual_candidates(
    partial: np.ndarray,
    registered_prior: np.ndarray,
    projector,
    *,
    diagonal: float,
    max_pixel_distance: float = 2.0,
    max_pairs: int = 10_000,
    trials: int = 64,
    seed: int = 6145,
    fractions: tuple[float, ...] = (0.125, 0.25, 0.50, 0.75, 1.0),
    max_rotation_deg: float = 4.0,
    scale_bounds: tuple[float, float] = (0.96, 1.04),
    max_translation_ratio: float = 0.04,
) -> tuple[list[tuple[str, np.ndarray]], dict]:
    """Fit MoGe-style robust 3-D residual candidates for a fixed registration.

    ``registered_prior`` is already in the partial frame.  The returned
    transforms therefore map this current Pixal body directly toward the hard
    partial.  Fractions of the robust fit retain a conservative trust region;
    the caller performs the multi-camera final gate.
    """
    partial, registered_prior = (np.asarray(value, dtype=np.float64)
                                 for value in (partial, registered_prior))
    pairs, pair_info = visible_pixel_pairs(
        partial, registered_prior, projector,
        max_pixel_distance=max_pixel_distance, max_pairs=max_pairs,
    )
    raw, fit = robust_indexed_sim3(
        registered_prior, partial, pairs, diagonal=float(diagonal),
        trials=int(trials), seed=int(seed),
    )
    prior_ids = pairs[:, 1].astype(np.int64)
    partial_ids = pairs[:, 0].astype(np.int64)
    before_residual = np.linalg.norm(registered_prior[prior_ids] - partial[partial_ids], axis=1)
    candidates = [("identity", np.eye(4, dtype=np.float64))]
    candidate_info = []
    for fraction in fractions:
        step = interpolate_sim3(raw, float(fraction))
        step = bounded_delta_sim3(
            step, max_rotation_deg=float(max_rotation_deg), scale_bounds=scale_bounds,
            max_translation=float(max_translation_ratio) * float(diagonal),
        )
        residual = np.linalg.norm(apply_transform(registered_prior[prior_ids], step) - partial[partial_ids], axis=1)
        limit = float(np.quantile(residual, .75))
        candidate_info.append({
            "fraction": float(fraction),
            "pair_median_before": float(np.median(before_residual)),
            "pair_median_after": float(np.median(residual)),
            "pair_trimmed_mean_before": float(before_residual[before_residual <= np.quantile(before_residual, .75)].mean()),
            "pair_trimmed_mean_after": float(residual[residual <= limit].mean()),
        })
        candidates.append((f"pixel_pair_fraction_{float(fraction):.3f}", step))
    return candidates, {
        "method": "moge_style_visible_pixel_indexed_3d_residual_sim3",
        "pairing": pair_info,
        "robust_fit": fit,
        "raw_residual": raw,
        "candidate_pair_residuals": candidate_info,
        "trust_region": {
            "max_rotation_deg": float(max_rotation_deg),
            "scale_bounds": [float(item) for item in scale_bounds],
            "max_translation_ratio": float(max_translation_ratio),
            "fractions": [float(item) for item in fractions],
        },
    }


def _centred_step(*, scale: float, rotation_deg: float, axis: int | None,
                  translation: np.ndarray | None, centre: np.ndarray) -> np.ndarray:
    """Return a proper residual Sim(3) about a stable partial-frame centre."""
    rotation = np.eye(3) if axis is None else Rotation.from_rotvec(
        np.eye(3)[axis] * np.deg2rad(float(rotation_deg))
    ).as_matrix()
    linear = float(scale) * rotation
    step = np.eye(4, dtype=np.float64)
    step[:3, :3] = linear
    step[:3, 3] = np.asarray(centre, dtype=np.float64) - linear @ np.asarray(centre, dtype=np.float64)
    if translation is not None:
        step[:3, 3] += np.asarray(translation, dtype=np.float64)
    return step


def _small_residual_proposals(*, scale_delta: float, rotation_deg: float,
                              translation: float, centre: np.ndarray) -> list[tuple[str, np.ndarray]]:
    proposals = [("identity", np.eye(4, dtype=np.float64))]
    for scale in (1. - float(scale_delta), 1. + float(scale_delta)):
        proposals.append((f"scale_{scale:.4f}", _centred_step(
            scale=scale, rotation_deg=0., axis=None, translation=None, centre=centre
        )))
    for axis, label in enumerate(("x", "y", "z")):
        for sign in (-1., 1.):
            proposals.append((f"rot_{label}_{sign:+.0f}", _centred_step(
                scale=1., rotation_deg=sign * float(rotation_deg), axis=axis,
                translation=None, centre=centre,
            )))
            shift = np.zeros(3, dtype=np.float64)
            shift[axis] = sign * float(translation)
            proposals.append((f"trans_{label}_{sign:+.0f}", _centred_step(
                scale=1., rotation_deg=0., axis=None, translation=shift, centre=centre,
            )))
    return proposals


def local_camera1_visible_refine(
    partial: np.ndarray,
    registered_prior: np.ndarray,
    projector,
    *,
    diagonal: float,
    search_points: int = 32_000,
    candidate_workers: int = 8,
    levels: tuple[tuple[float, float, float], ...] = (
        (.003, .15, .003), (.001, .06, .001), (.0005, .025, .0005),
    ),
) -> tuple[np.ndarray, dict]:
    """Make only a sub-percent Camera-1 Sim(3) correction after pair fitting.

    Pixel-indexed 3-D fitting supplies the meaningful basin.  This final
    deterministic coordinate descent optimizes the same no-GT Camera-1
    visible score on fixed subsamples, then the caller rechecks its selected
    step with the full point sets. Candidate scores are independent, so they
    may be evaluated concurrently; results are consumed in proposal order and
    the candidate set/selection rule remains unchanged. It does not add
    correspondences, points, or local deformation degrees of freedom.
    """
    partial, registered_prior = (np.asarray(value, dtype=np.float64)
                                 for value in (partial, registered_prior))
    if len(partial) < 6 or len(registered_prior) < 6:
        raise ValueError("nontrivial partial and prior point clouds are required")
    p_ids = (np.arange(len(partial), dtype=np.int64) if len(partial) <= int(search_points)
             else np.linspace(0, len(partial) - 1, int(search_points), dtype=np.int64))
    q_ids = (np.arange(len(registered_prior), dtype=np.int64) if len(registered_prior) <= int(search_points)
             else np.linspace(0, len(registered_prior) - 1, int(search_points), dtype=np.int64))
    target, current = partial[p_ids], registered_prior[q_ids].copy()
    target_cache = prepare_visible_target(target, projector)
    centre = np.median(partial, axis=0)
    total = np.eye(4, dtype=np.float64)
    before = visible_score(target, current, projector, float(diagonal), pixel_radius=5., target_cache=target_cache)
    trace = []
    for scale_delta, degrees, translation_ratio in levels:
        proposals = _small_residual_proposals(
            scale_delta=float(scale_delta), rotation_deg=float(degrees),
            translation=float(translation_ratio) * float(diagonal), centre=centre,
        )

        def score_proposal(proposal: tuple[str, np.ndarray]):
            action, step = proposal
            candidate = apply_transform(current, step)
            score = visible_score(target, candidate, projector, float(diagonal), pixel_radius=5., target_cache=target_cache)
            return score["objective"], action, step, candidate, score

        workers = min(max(int(candidate_workers), 1), len(proposals))
        if workers == 1:
            scored = [score_proposal(proposal) for proposal in proposals]
        else:
            # ``map`` preserves proposal order. This retains the sequential
            # ``min`` tie rule while releasing CPU-bound raster/KNN work.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                scored = list(executor.map(score_proposal, proposals))
        _, action, step, current, score = min(scored, key=lambda item: item[0])
        total = step @ total
        trace.append({
            "level": {"scale_delta": float(scale_delta), "rotation_deg": float(degrees),
                      "translation_ratio": float(translation_ratio)},
            "action": action,
            "score": {
                "objective": float(score["objective"]),
                "projection": {key: float(value) for key, value in score["projection"].items()},
            },
        })
    after = visible_score(target, current, projector, float(diagonal), pixel_radius=5., target_cache=target_cache)
    return total, {
        "method": "subpercent_camera1_visible_sim3_coordinate_refinement",
        "search_points": {"partial": int(len(target)), "prior": int(len(current))},
        "candidate_workers": int(max(candidate_workers, 1)),
        "levels": [{"scale_delta": float(item[0]), "rotation_deg": float(item[1]),
                    "translation_ratio": float(item[2])} for item in levels],
        "before": float(before["objective"]), "after": float(after["objective"]), "trace": trace,
    }
