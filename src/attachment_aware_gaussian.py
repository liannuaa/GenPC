"""Support-anchored axial deformation for a Gaussian-mean prior carrier.

This module is an isolated registration/adaptation proposal.  A complete
prior is represented by its sampled Gaussian means.  When a visible residual
forms a coherent screen component, the proposal identifies a graph attachment
to already aligned surface support and applies a *local axial stretch* about
that attachment.  The stretch is blended with a scalar deformation-graph
weight; matched partial/MoGe support and the exterior of the local band are
identity constraints.  Consequently it changes a supported length without
turning a limb into a translated, detached point set.

All selection quantities are computed from the partial, the registered prior,
the saved camera and (optionally) an already bridged auxiliary observation.
Complete shapes and offline metrics never enter the estimate.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

from src.residual_component_registration import (
    _geodesic_distance,
    _prior_surface_graph,
    _screen_components,
    _visible_residual_records,
)


def _solve_scalar_dirichlet(
    graph: sparse.csr_matrix,
    positive_ids: np.ndarray,
    zero_ids: np.ndarray,
    *,
    screening: float,
) -> tuple[np.ndarray, int]:
    """Return a bounded harmonic weight with positive and zero graph controls."""
    count = graph.shape[0]
    positive = np.unique(np.asarray(positive_ids, dtype=np.int64))
    zero = np.unique(np.asarray(zero_ids, dtype=np.int64))
    positive = np.setdiff1d(positive, zero, assume_unique=False)
    controls = np.r_[zero, positive]
    fixed = np.zeros(count, dtype=bool)
    fixed[controls] = True
    value = np.zeros(count, dtype=np.float64)
    value[positive] = 1.
    unknown = np.flatnonzero(~fixed)
    if len(unknown) == 0:
        return value, 0
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - graph
    system = laplacian[unknown][:, unknown].tocsr() + sparse.eye(
        len(unknown), format="csr", dtype=np.float64,
    ) * float(screening)
    rhs = -(laplacian[unknown][:, controls] @ value[controls])
    try:
        solution, status = sparse_linalg.cg(
            system, np.asarray(rhs).reshape(-1), rtol=1e-5, atol=0., maxiter=320,
        )
    except TypeError:  # SciPy < 1.12
        solution, status = sparse_linalg.cg(
            system, np.asarray(rhs).reshape(-1), tol=1e-5, maxiter=320,
        )
    value[unknown] = solution
    return np.clip(value, 0., 1.), int(status)


def _coherent_residual_selection(
    prior: np.ndarray,
    partial: np.ndarray,
    projector,
    camera_axes: np.ndarray,
    *,
    max_pixel_distance: float,
    minimum_residual_ratio: float,
    screen_component_radius: float,
    minimum_component_pairs: int,
    minimum_directional_coherence: float,
    minimum_camera_axis_dominance: float,
    component_direction_cosine: float,
    retain_all_compatible_components: bool = False,
) -> tuple[dict | None, dict]:
    """Select compatible high-residual components using partial-only evidence."""
    partial_ids, prior_ids, xy, residual, _ = _visible_residual_records(
        partial, prior, projector, max_pixel_distance=float(max_pixel_distance),
    )
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    residual_norm = np.linalg.norm(residual, axis=1)
    high = residual_norm >= float(minimum_residual_ratio) * diagonal
    base = {
        "visible_pairs": int(len(residual)),
        "large_residual_pairs": int(high.sum()),
        "diagonal": diagonal,
    }
    if int(high.sum()) < int(minimum_component_pairs):
        return None, {**base, "reason": "insufficient_large_visible_residuals"}
    high_indices = np.flatnonzero(high)
    count, labels = _screen_components(xy[high], radius=float(screen_component_radius))
    candidates: list[dict] = []
    for label in range(count):
        members = labels == label
        if int(members.sum()) < int(minimum_component_pairs):
            continue
        component_residual = residual[high][members]
        translation = np.median(component_residual, axis=0)
        magnitude = float(np.linalg.norm(translation))
        if magnitude <= 1e-12:
            continue
        coherence = float(np.mean(
            (component_residual @ translation)
            / np.maximum(np.linalg.norm(component_residual, axis=1) * magnitude, 1e-12)
        ))
        camera_translation = translation @ camera_axes.T
        camera_axis = int(np.argmax(np.abs(camera_translation)))
        dominance = float(abs(camera_translation[camera_axis]) / magnitude)
        if coherence < float(minimum_directional_coherence) or dominance < float(minimum_camera_axis_dominance):
            continue
        candidates.append({
            "label": int(label), "members": members, "translation": translation,
            "magnitude": magnitude, "coherence": coherence,
            "camera_axis": camera_axis, "camera_dominance": dominance,
        })
    if not candidates:
        return None, {**base, "screen_components": int(count),
                      "reason": "no_coherent_camera_axis_residual_component"}
    candidates.sort(key=lambda item: (-int(item["members"].sum()) * float(item["magnitude"]),
                                       int(item["label"])))
    reference = candidates[0]
    direction = reference["translation"] / float(reference["magnitude"])
    selected = candidates if retain_all_compatible_components else [item for item in candidates if (
        int(item["camera_axis"]) == int(reference["camera_axis"])
        and float(np.dot(item["translation"] / float(item["magnitude"]), direction))
        >= float(component_direction_cosine)
    )]
    mask = np.zeros(int(high.sum()), dtype=bool)
    for item in selected:
        mask |= np.asarray(item["members"], dtype=bool)
    record_indices = high_indices[mask]
    return {
        "partial_ids": partial_ids,
        "prior_ids": prior_ids,
        "residual": residual,
        "residual_norm": residual_norm,
        "record_indices": record_indices,
        "selected": selected,
        "diagonal": diagonal,
    }, {**base, "screen_components": int(count),
         "coherent_component_candidates": int(len(candidates)),
         "selected_components": int(len(selected)),
         "selected_component_pairs": int(len(record_indices)),
         "retain_all_compatible_components": bool(retain_all_compatible_components)}


def support_anchored_axial_gaussian_deformation(
    prior: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    camera_axes: np.ndarray,
    auxiliary_support: np.ndarray | None = None,
    auxiliary_support_max_distance: float | None = None,
    max_pixel_distance: float = 1.,
    minimum_residual_ratio: float = .15,
    screen_component_radius: float = 3.,
    minimum_component_pairs: int = 128,
    minimum_directional_coherence: float = .85,
    minimum_camera_axis_dominance: float = .75,
    component_direction_cosine: float = .95,
    stable_residual_ratio: float = .05,
    minimum_stable_pairs: int = 512,
    graph_neighbours: int = 8,
    graph_edge_ratio: float = 1.8,
    graph_screening: float = .0015,
    core_geodesic_radius_ratio: float = .04,
    influence_geodesic_radius_ratio: float = .10,
    maximum_influence_fraction: float = .30,
    maximum_log_scale: float = .4054651081081644,
    minimum_component_improvement: float = .25,
) -> tuple[np.ndarray, dict]:
    """Produce a bounded, attachment-anchored axial deformation candidate."""
    prior, partial = (np.asarray(value, dtype=np.float64) for value in (prior, partial))
    zero = np.zeros_like(prior)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    axes = np.asarray(camera_axes, dtype=np.float64)
    if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3), rtol=1e-4, atol=1e-5):
        raise ValueError("camera_axes must be an orthonormal (3, 3) row-axis matrix")
    if not (0. < stable_residual_ratio < 1. and 0. < core_geodesic_radius_ratio
            <= influence_geodesic_radius_ratio and 0. < maximum_influence_fraction <= 1.
            and maximum_log_scale > 0. and 0. < minimum_component_improvement < 1.):
        raise ValueError("invalid support-anchored deformation bounds")
    evidence, base = _coherent_residual_selection(
        prior, partial, projector, axes,
        max_pixel_distance=max_pixel_distance,
        minimum_residual_ratio=minimum_residual_ratio,
        screen_component_radius=screen_component_radius,
        minimum_component_pairs=minimum_component_pairs,
        minimum_directional_coherence=minimum_directional_coherence,
        minimum_camera_axis_dominance=minimum_camera_axis_dominance,
        component_direction_cosine=component_direction_cosine,
    )
    if evidence is None:
        return zero, {"active": False, **base}
    diagonal = float(evidence["diagonal"])
    records = np.asarray(evidence["record_indices"], dtype=np.int64)
    prior_ids, partial_ids = evidence["prior_ids"], evidence["partial_ids"]
    residual_norm = evidence["residual_norm"]
    partial_stable = np.unique(prior_ids[residual_norm <= float(stable_residual_ratio) * diagonal])
    seeds = np.setdiff1d(np.unique(prior_ids[records]), partial_stable, assume_unique=False)
    if len(seeds) < max(6, int(minimum_component_pairs) // 2) or len(partial_stable) < int(minimum_stable_pairs):
        return zero, {"active": False, **base, "reason": "insufficient_disjoint_fixed_visible_support"}
    graph = _prior_surface_graph(prior, neighbours=int(graph_neighbours), edge_ratio=float(graph_edge_ratio))
    geodesic = _geodesic_distance(graph, seeds)
    core = np.flatnonzero(geodesic <= float(core_geodesic_radius_ratio) * diagonal)
    influence = np.flatnonzero(geodesic <= float(influence_geodesic_radius_ratio) * diagonal)
    if (len(core) < max(6, int(minimum_component_pairs) // 2)
            or float(len(influence) / len(prior)) > float(maximum_influence_fraction)):
        return zero, {"active": False, **base, "reason": "unsupported_or_overbroad_local_structure",
                      "core_prior_points": int(len(core)), "influence_prior_points": int(len(influence))}
    # The graph boundary of the residual core is the automatically inferred
    # attachment interface. Prefer hard partial support; only then use nearby
    # exterior vertices so sparse scans can still make a proposal.
    coo = graph.tocoo()
    crossing = np.isin(coo.row, core) ^ np.isin(coo.col, core)
    exterior = np.where(np.isin(coo.row[crossing], core), coo.col[crossing], coo.row[crossing])
    attachment = np.intersect1d(np.unique(exterior), partial_stable, assume_unique=False)
    if len(attachment) < 6:
        attachment = np.unique(exterior)
    if len(attachment) < 6:
        return zero, {"active": False, **base, "reason": "missing_structural_attachment"}
    pivot = np.median(prior[attachment], axis=0)
    source = prior[prior_ids[records]] - pivot
    target = partial[partial_ids[records]] - pivot
    residual_direction = np.median(target - source, axis=0)
    component_center = np.median(prior[core], axis=0)
    _, _, right = np.linalg.svd(prior[core] - component_center, full_matrices=False)
    candidate_axes = np.vstack((
        component_center - pivot,
        residual_direction,
        right,
        axes,
    ))
    before = np.linalg.norm(target - source, axis=1)
    best: dict | None = None
    maximum_scale = float(np.exp(maximum_log_scale))
    minimum_scale = 1. / maximum_scale
    for direction in candidate_axes:
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        unit = direction / norm
        source_axis = source @ unit
        denominator = float(source_axis @ source_axis)
        if denominator <= 1e-12:
            continue
        scale = float(np.clip((source_axis @ (target @ unit)) / denominator,
                              minimum_scale, maximum_scale))
        prediction = source + (scale - 1.) * source_axis[:, None] * unit
        error = np.linalg.norm(target - prediction, axis=1)
        item = {"axis": unit, "scale": scale, "error": error}
        if best is None or float(np.median(error)) < float(np.median(best["error"])):
            best = item
    if best is None:
        return zero, {"active": False, **base, "reason": "unobservable_attachment_axis"}
    reduction = 1. - float(np.median(best["error"]) / max(np.median(before), 1e-12))
    if reduction < float(minimum_component_improvement):
        return zero, {"active": False, **base, "reason": "insufficient_axial_data_improvement",
                      "component_improvement": reduction}
    auxiliary_stable = np.empty(0, dtype=np.int64)
    if auxiliary_support is not None:
        support = np.asarray(auxiliary_support, dtype=np.float64)
        if support.ndim != 2 or support.shape[1] != 3 or len(support) < 6:
            raise ValueError("auxiliary_support must be a nontrivial (N, 3) point cloud")
        if auxiliary_support_max_distance is None or float(auxiliary_support_max_distance) <= 0.:
            raise ValueError("auxiliary support requires a positive max distance")
        distance, _ = cKDTree(support).query(prior, k=1, workers=-1)
        auxiliary_stable = np.setdiff1d(
            np.flatnonzero(distance <= float(auxiliary_support_max_distance)), core, assume_unique=False,
        )
    outside = np.setdiff1d(np.arange(len(prior), dtype=np.int64), influence, assume_unique=True)
    fixed = np.unique(np.r_[partial_stable, auxiliary_stable, outside])
    moving_core = np.setdiff1d(core, fixed, assume_unique=False)
    if len(moving_core) < max(6, int(minimum_component_pairs) // 2):
        return zero, {"active": False, **base, "reason": "core_overconstrained_by_observation"}
    weight, cg_status = _solve_scalar_dirichlet(
        graph, moving_core, fixed, screening=float(graph_screening),
    )
    axis = np.asarray(best["axis"], dtype=np.float64)
    local_coordinate = (prior - pivot) @ axis
    # A log-linear scale interpolation makes the transform identity on the
    # attachment/exterior controls and the fitted axial stretch in the core.
    scale_field = np.exp(np.log(float(best["scale"])) * weight)
    displacement = (scale_field - 1.)[:, None] * local_coordinate[:, None] * axis
    moved = np.linalg.norm(displacement, axis=1) > 1e-8
    return displacement, {
        "active": True,
        "method": "support_anchored_axial_gaussian_deformation",
        "field": "bounded_attachment_anchored_log_axial_scale",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        **base,
        "selected_component_pairs": int(len(records)),
        "core_prior_points": int(len(core)),
        "influence_prior_points": int(len(influence)),
        "partial_stable_prior_points": int(len(partial_stable)),
        "auxiliary_stable_prior_points": int(len(auxiliary_stable)),
        "outer_fixed_prior_points": int(len(outside)),
        "attachment_prior_points": int(len(attachment)),
        "pivot": pivot.tolist(),
        "axis": axis.tolist(),
        "core_scale": float(best["scale"]),
        "component_residual_median_before": float(np.median(before)),
        "component_residual_median_after": float(np.median(best["error"])),
        "component_improvement": reduction,
        "moved_prior_points": int(moved.sum()),
        "moved_fraction": float(moved.mean()),
        "mean_moved_displacement": float(np.linalg.norm(displacement[moved], axis=1).mean()) if np.any(moved) else 0.,
        "maximum_displacement": float(np.linalg.norm(displacement, axis=1).max(initial=0.)),
        "cg_status": cg_status,
        "carrier_slots_preserved": True,
    }
