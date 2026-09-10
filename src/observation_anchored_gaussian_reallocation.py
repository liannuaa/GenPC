"""Observation-anchored local reallocation for a complete Gaussian carrier.

One view cannot validate every part of a generative prior.  It can, however,
say something stronger than a small mean correction wherever the measured
surface is visible: a partial point is an immutable *surface boundary*
rather than a soft registration preference.  This module transfers those
boundaries to a fixed population of complete-prior Gaussian slots and solves
a compact harmonic continuation over the prior surface graph.

The carrier count is unchanged and no raw point-cloud union is formed.  Only
screen-connected partial residual regions may become hard controls; aligned
visible support and the exterior of each local geodesic band are fixed.  The
result is therefore a continuous local stretch from the prior to the observed
surface, rather than a detached component translation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from src.camera_conditioned_gaussian_adaptation import _edge_strain, _local_gaussian_graph, _solve_vector_system
from src.residual_component_registration import _screen_components
from src.zbuffer import zbuffer_depth_with_indices


def _saved_view_nearest_pairs(
    partial: np.ndarray,
    prior: np.ndarray,
    projector,
    *,
    maximum_screen_distance: float,
    mutual_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Associate visible partial pixels with nearest visible prior pixels.

    Unlike ordinary same-pixel matching, this deliberately permits a bounded
    screen gap.  Such a gap is the signature of a missing prior projection
    (for example a leg that is too short), and supplies an observation anchor
    rather than being silently discarded.
    """
    partial_uv, partial_depth = projector.project(partial)
    prior_uv, prior_depth = projector.project(prior)
    _, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=0,
    )
    _, prior_mask, prior_index = zbuffer_depth_with_indices(
        prior_uv, prior_depth, projector.image_shape, splat_radius=1,
    )
    py, px = np.where(partial_mask)
    qy, qx = np.where(prior_mask)
    if min(len(px), len(qx)) < 6:
        return (
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    screen_distance, nearest = cKDTree(np.c_[qx, qy]).query(np.c_[px, py], k=1)
    keep = screen_distance <= float(maximum_screen_distance)
    if mutual_only:
        _, reverse = cKDTree(np.c_[px, py]).query(np.c_[qx, qy], k=1)
        keep &= reverse[nearest] == np.arange(len(px), dtype=np.int64)
    partial_ids = partial_index[py[keep], px[keep]].astype(np.int64)
    prior_ids = prior_index[qy[nearest[keep]], qx[nearest[keep]]].astype(np.int64)
    valid = (partial_ids >= 0) & (prior_ids >= 0)
    partial_ids, prior_ids = partial_ids[valid], prior_ids[valid]
    xy = np.c_[px[keep][valid], py[keep][valid]].astype(np.float64)
    screen_distance = screen_distance[keep][valid]
    residual = np.linalg.norm(partial[partial_ids] - prior[prior_ids], axis=1)
    return partial_ids, prior_ids, xy, screen_distance, residual


def _component_observation_pairs(
    partial: np.ndarray,
    prior: np.ndarray,
    projector,
    *,
    maximum_screen_distance: float,
    stable_screen_distance: float,
    residual_ratio: float,
    minimum_component_pixels: int,
    component_radius: float,
    minimum_directional_coherence: float,
    mutual_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return hard candidate pairs and stable visible prior slots.

    Components are formed in the partial image, not on arbitrary 3-D nearest
    neighbours.  Thus a candidate is activated only where the actual scan
    exposes a coherent surface discrepancy.
    """
    partial_ids, prior_ids, xy, screen_distance, residual = _saved_view_nearest_pairs(
        partial, prior, projector, maximum_screen_distance=maximum_screen_distance,
        mutual_only=bool(mutual_only),
    )
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    if len(partial_ids) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.int64), {
            "active": False, "reason": "insufficient_visible_saved_view_pairs", "diagonal": diagonal,
            "mutual_only": bool(mutual_only),
        }
    high = residual >= float(residual_ratio) * diagonal
    # A pixel with a gap larger than the normal splat tolerance is an explicit
    # missing-prior surface observation even when its nearest 3-D residual is
    # hard to estimate from one view alone.
    missing = screen_distance > float(stable_screen_distance)
    candidate = high | missing
    stable = (screen_distance <= float(stable_screen_distance)) & ~high
    stable_prior = np.unique(prior_ids[stable])
    if int(candidate.sum()) < int(minimum_component_pixels):
        return np.empty((0, 4), dtype=np.float64), stable_prior, {
            "active": False, "reason": "insufficient_residual_or_missing_partial_pixels",
            "diagonal": diagonal, "visible_pairs": int(len(partial_ids)),
            "candidate_pixels": int(candidate.sum()), "stable_visible_prior": int(len(stable_prior)),
            "mutual_only": bool(mutual_only),
        }
    count, labels = _screen_components(xy[candidate], radius=float(component_radius))
    candidate_indices = np.flatnonzero(candidate)
    selected = np.zeros(len(partial_ids), dtype=bool)
    components: list[dict] = []
    vectors = partial[partial_ids] - prior[prior_ids]
    for label in range(count):
        member_local = labels == label
        members = candidate_indices[member_local]
        if len(members) < int(minimum_component_pixels):
            continue
        vector = vectors[members]
        median = np.median(vector, axis=0)
        magnitude = float(np.linalg.norm(median))
        if magnitude <= 1e-9:
            continue
        coherence = float(np.mean((vector @ median) / np.maximum(
            np.linalg.norm(vector, axis=1) * magnitude, 1e-12,
        )))
        # Components that are mostly a silhouette gap can have modest 3-D
        # directional coherence.  Retain them, but reject noisy mixtures that
        # would turn one image region into conflicting anchors.
        if coherence < float(minimum_directional_coherence):
            continue
        selected[members] = True
        components.append({
            "label": int(label), "pixels": int(len(members)),
            "median_residual": median.tolist(), "median_residual_norm": magnitude,
            "directional_coherence": coherence,
            "missing_pixels": int(missing[members].sum()),
        })
    pairs = np.c_[
        partial_ids[selected], prior_ids[selected], prior_ids[selected], screen_distance[selected],
    ].astype(np.float64)
    return pairs, stable_prior, {
        "active": bool(len(pairs)), "diagonal": diagonal,
        "visible_pairs": int(len(partial_ids)), "candidate_pixels": int(candidate.sum()),
        "selected_candidate_pixels": int(selected.sum()), "screen_components": int(count),
        "selected_components": components, "stable_visible_prior": int(len(stable_prior)),
        "mutual_only": bool(mutual_only),
    }


def _expand_screen_anchor_slots(
    pairs: np.ndarray,
    partial: np.ndarray,
    prior: np.ndarray,
    *,
    maximum_residual: float,
    allocation_neighbours: int,
    blocked_prior_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Give each partial observation a distinct local prior-carrier slot.

    Z-buffer visibility maps many adjacent partial pixels to one front-most
    prior point.  Treating that point as the only editable handle makes a
    dense observed leg collapse to a few anchors.  We instead borrow unused
    *local* Gaussian slots around that visible point.  The target is still the
    measured partial point and the population size never changes.
    """
    pairs = np.asarray(pairs, dtype=np.float64)
    blocked = np.zeros(len(prior), dtype=bool)
    blocked[np.asarray(blocked_prior_ids, dtype=np.int64)] = True
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64), {
            "pixel_pairs": 0, "collision_free_pairs": 0, "allocated_from_local_slots": 0,
        }
    partial_ids = pairs[:, 0].astype(np.int64)
    base_ids = pairs[:, 1].astype(np.int64)
    screen_distance = pairs[:, 3]
    query_k = min(int(allocation_neighbours), len(prior))
    if query_k < 1:
        return np.empty((0, 2), dtype=np.int64), {
            "pixel_pairs": int(len(pairs)), "collision_free_pairs": 0,
            "allocated_from_local_slots": 0,
        }
    distances, slot_pool = cKDTree(prior).query(prior[base_ids], k=query_k, workers=-1)
    if query_k == 1:
        distances, slot_pool = distances[:, None], slot_pool[:, None]
    # Allocate the least screen-supported observations first.  This makes the
    # scarce local slots resolve missing-surface pixels before ordinary nearby
    # residuals, while ties remain deterministic.
    source_residual = np.linalg.norm(partial[partial_ids] - prior[base_ids], axis=1)
    order = np.lexsort((base_ids, partial_ids, -source_residual, -screen_distance))
    used_slots = blocked.copy()
    used_partial: set[int] = set()
    selected: list[tuple[int, int]] = []
    local_allocations = 0
    for row in order:
        partial_id = int(partial_ids[row])
        if partial_id in used_partial:
            continue
        target = partial[partial_ids[row]]
        candidates = slot_pool[row]
        target_distance = np.linalg.norm(prior[candidates] - target, axis=1)
        valid = (~used_slots[candidates]) & (target_distance <= float(maximum_residual))
        if not valid.any():
            continue
        candidate = candidates[valid][np.argmin(target_distance[valid])]
        used_slots[candidate] = True
        used_partial.add(partial_id)
        selected.append((partial_id, int(candidate)))
        local_allocations += int(candidate != int(base_ids[row]))
    anchors = np.asarray(selected, dtype=np.int64).reshape(-1, 2)
    residual = np.linalg.norm(partial[anchors[:, 0]] - prior[anchors[:, 1]], axis=1) if len(anchors) else np.empty(0)
    return anchors, {
        "pixel_pairs": int(len(pairs)), "collision_free_pairs": int(len(anchors)),
        "allocated_from_local_slots": int(local_allocations), "allocation_neighbours": int(query_k),
        "blocked_stable_slots": int(blocked.sum()),
        "residual_median": float(np.median(residual)) if len(residual) else float("inf"),
        "residual_p90": float(np.quantile(residual, .90)) if len(residual) else float("inf"),
        "max_residual": float(maximum_residual),
    }


def _local_dirichlet_field(
    graph: sparse.csr_matrix,
    influence: np.ndarray,
    anchor_ids: np.ndarray,
    anchor_offsets: np.ndarray,
    stable_ids: np.ndarray,
    *,
    screening: float,
) -> tuple[np.ndarray, dict]:
    """Solve a compact exact-anchor field with a zero-motion outer boundary."""
    count = graph.shape[0]
    influence = np.asarray(influence, dtype=bool)
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError("anchor slots must be collision-free")
    anchor_lookup = np.full(count, -1, dtype=np.int64)
    anchor_lookup[anchors] = np.arange(len(anchors), dtype=np.int64)
    # The direct anchor values have already been collision-free reduced.
    values = np.zeros((count, 3), dtype=np.float64)
    values[anchors] = anchor_offsets
    # Explicitly fix the band boundary.  Holding only points outside the band
    # would leave cut edges unconstrained once the system is restricted.
    coo = graph.tocoo()
    crossing = influence[coo.row] ^ influence[coo.col]
    boundary = np.unique(np.where(influence[coo.row[crossing]], coo.row[crossing], coo.col[crossing]))
    fixed = ~influence
    fixed[np.asarray(stable_ids, dtype=np.int64)] = True
    fixed[boundary] = True
    # Exact observation anchors take precedence over a stale stable label.
    fixed[anchors] = True
    movable = influence & ~fixed
    unknown = np.flatnonzero(movable)
    fixed_ids = np.flatnonzero(fixed)
    if len(unknown) == 0:
        return values, {"cg_status": [0, 0, 0], "boundary_gaussians": int(len(boundary)),
                        "movable_gaussians": 0}
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - graph
    system = laplacian[unknown][:, unknown].tocsr() + sparse.eye(
        len(unknown), format="csr", dtype=np.float64,
    ) * float(screening)
    rhs = -(laplacian[unknown][:, fixed_ids] @ values[fixed_ids])
    solved, status = _solve_vector_system(system, np.asarray(rhs))
    values[unknown] = solved
    return values, {
        "cg_status": status, "boundary_gaussians": int(len(boundary)),
        "movable_gaussians": int(len(unknown)),
    }


def observation_anchored_gaussian_reallocation(
    prior: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    maximum_screen_distance: float = 20.,
    stable_screen_distance: float = 1.,
    residual_ratio: float = .045,
    minimum_component_pixels: int = 48,
    component_radius: float = 3.,
    minimum_directional_coherence: float = .35,
    maximum_anchor_residual_ratio: float = .32,
    allocation_neighbours: int = 16,
    influence_radius_ratios: tuple[float, ...] = (.10, .14, .18, .24, .30),
    maximum_influence_fraction: float = .55,
    neighbours: int = 10,
    edge_ratio: float = 1.8,
    screening: float = .003,
    maximum_edge_stretch: float = 2.6,
    minimum_edge_compression: float = .38,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Reallocate prior Gaussian means to coherent partial surface evidence.

    The only variable selected from data is the local screen-connected residual
    support.  Candidate band radii are deterministic continuation levels; the
    first one that preserves local graph continuity is used.  Neither GT nor
    category-specific geometry participates in activation or selection.
    """
    prior, partial = np.asarray(prior, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if (maximum_screen_distance <= stable_screen_distance or residual_ratio <= 0.
            or minimum_component_pixels < 6 or component_radius <= 0.
            or not 0. <= minimum_directional_coherence <= 1.
            or maximum_anchor_residual_ratio <= 0. or not influence_radius_ratios
            or not 0. < maximum_influence_fraction <= 1. or neighbours < 2
            or allocation_neighbours < 1
            or edge_ratio <= 0. or screening <= 0. or maximum_edge_stretch <= 1.
            or not 0. < minimum_edge_compression < 1.):
        raise ValueError("invalid observation-anchored reallocation parameters")
    empty = {
        "anchors": np.zeros(len(prior), dtype=bool), "stable": np.zeros(len(prior), dtype=bool),
        "influence": np.zeros(len(prior), dtype=bool), "boundary": np.zeros(len(prior), dtype=bool),
        "moved": np.zeros(len(prior), dtype=bool),
    }
    pairs, stable_ids, evidence = _component_observation_pairs(
        partial, prior, projector, maximum_screen_distance=float(maximum_screen_distance),
        stable_screen_distance=float(stable_screen_distance), residual_ratio=float(residual_ratio),
        minimum_component_pixels=int(minimum_component_pixels), component_radius=float(component_radius),
        minimum_directional_coherence=float(minimum_directional_coherence),
    )
    if not len(pairs):
        return prior.copy(), {**evidence, "active": False, "method": "observation_anchored_local_gaussian_reallocation"}, empty
    diagonal = float(evidence["diagonal"])
    anchors, anchor_info = _expand_screen_anchor_slots(
        pairs, partial, prior, maximum_residual=float(maximum_anchor_residual_ratio) * diagonal,
        allocation_neighbours=int(allocation_neighbours), blocked_prior_ids=stable_ids,
    )
    if len(anchors) < int(minimum_component_pixels):
        return prior.copy(), {
            **evidence, **anchor_info, "active": False, "method": "observation_anchored_local_gaussian_reallocation",
            "reason": "insufficient_collision_free_observation_anchors",
        }, empty
    partial_ids, anchor_ids = anchors[:, 0], anchors[:, 1]
    offsets = partial[partial_ids] - prior[anchor_ids]
    graph, metric, edge_count = _local_gaussian_graph(prior, neighbours=int(neighbours), edge_ratio=float(edge_ratio))
    if edge_count == 0:
        return prior.copy(), {
            **evidence, **anchor_info, "active": False, "method": "observation_anchored_local_gaussian_reallocation",
            "reason": "empty_local_gaussian_surface_graph",
        }, empty
    geodesic = csgraph.dijkstra(metric, directed=False, indices=anchor_ids, min_only=True)
    stable_ids = np.setdiff1d(stable_ids, anchor_ids, assume_unique=False)
    best_failure: dict | None = None
    for radius_ratio in influence_radius_ratios:
        radius = float(radius_ratio) * diagonal
        influence = np.isfinite(geodesic) & (geodesic <= radius)
        if not influence[anchor_ids].all() or float(influence.mean()) > float(maximum_influence_fraction):
            continue
        displacement, solved = _local_dirichlet_field(
            graph, influence, anchor_ids, offsets, stable_ids, screening=float(screening),
        )
        # Reconstruct the boundary state for output without duplicating the
        # solve's graph logic in the caller.
        coo = graph.tocoo()
        crossing = influence[coo.row] ^ influence[coo.col]
        boundary_ids = np.unique(np.where(influence[coo.row[crossing]], coo.row[crossing], coo.col[crossing]))
        magnitude = np.linalg.norm(displacement, axis=1)
        edited = prior + displacement
        strain = _edge_strain(prior, edited, graph)
        trial = {
            "radius_ratio": float(radius_ratio), "influence_gaussians": int(influence.sum()),
            "influence_fraction": float(influence.mean()), **solved, **strain,
        }
        if (strain["edge_stretch_p01"] >= float(minimum_edge_compression)
                and strain["edge_stretch_p999"] <= float(maximum_edge_stretch)):
            masks = {
                "anchors": np.zeros(len(prior), dtype=bool),
                "stable": np.zeros(len(prior), dtype=bool), "influence": influence,
                "boundary": np.zeros(len(prior), dtype=bool), "moved": magnitude > 1e-8,
            }
            masks["anchors"][anchor_ids] = True
            masks["stable"][stable_ids] = True
            masks["boundary"][boundary_ids] = True
            return edited, {
                "active": True, "method": "observation_anchored_local_gaussian_reallocation",
                "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
                "carrier_slots_preserved": bool(len(edited) == len(prior)), **evidence, **anchor_info,
                "anchors": int(len(anchor_ids)), "stable_visible_gaussians": int(len(stable_ids)),
                "surface_graph_edges": int(edge_count), "selected_band": trial,
                "moved_gaussians": int((magnitude > 1e-8).sum()),
                "mean_displacement": float(magnitude.mean()),
                "maximum_displacement": float(magnitude.max(initial=0.)),
                "anchor_residual_before_median": float(np.median(np.linalg.norm(offsets, axis=1))),
                "anchor_residual_after_median": float(np.median(np.linalg.norm(
                    edited[anchor_ids] - partial[partial_ids], axis=1,
                ))),
                "field": "partial_depth_dirichlet_anchors_with_geodesic_gaussian_continuation",
            }, masks
        best_failure = trial
    return prior.copy(), {
        **evidence, **anchor_info, "active": False, "method": "observation_anchored_local_gaussian_reallocation",
        "reason": "no_continuous_local_geodesic_band", "last_trial": best_failure,
    }, empty
