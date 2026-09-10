"""Structure-aware partial-to-prior ARAP adaptation for Gaussian means.

The saved camera is reliable for global registration, but a single depth
layer may hide a supporting leg or appendage.  This module extracts additional
positive constraints directly from the registered 3-D partial: a partial
point may anchor a prior Gaussian only when its local normal is compatible and
its residual belongs to a spatially connected, directionally coherent
component.  The selected components are then transferred through a bounded
ARAP field, while low-residual structural correspondences are fixed.

It is a fixed-cardinality 3DGS-mean edit.  No category semantics, mesh
topology, GT, or point-cloud concatenation is used.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from src.arap_gaussian_reallocation import _arap_surface_field
from src.camera_conditioned_gaussian_adaptation import _edge_strain, _local_gaussian_graph


def _estimate_normals(points: np.ndarray, *, neighbours: int) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=int(neighbours)))
    return np.asarray(cloud.normals, dtype=np.float64)


def moge_unsupported_partial_mask(
    partial: np.ndarray,
    moge: np.ndarray,
    *,
    normal_neighbours: int = 24,
    maximum_distance_ratio: float = .045,
    minimum_normal_agreement: float = .35,
) -> tuple[np.ndarray, dict]:
    """Identify observed partial regions that MoGe cannot safely bridge.

    The returned mask is deliberately conservative: a partial sample is
    *unsupported* only when the bridge cloud has neither a nearby compatible
    surface nor a locally consistent normal.  It is a diagnostic of the
    image-derived bridge, not an assertion that the partial point is an empty
    space constraint.  Consequently downstream code can give such points a
    direct partial-to-prior path without forcing the global bridge to explain
    a genuinely different local shape.
    """
    partial, moge = np.asarray(partial, dtype=np.float64), np.asarray(moge, dtype=np.float64)
    if partial.ndim != 2 or moge.ndim != 2 or partial.shape[1:] != (3,) or moge.shape[1:] != (3,):
        raise ValueError("partial and moge must be (N, 3)")
    if len(partial) < normal_neighbours or len(moge) < normal_neighbours:
        raise ValueError("partial and moge must contain at least normal_neighbours points")
    if normal_neighbours < 6 or maximum_distance_ratio <= 0. or not 0. <= minimum_normal_agreement <= 1.:
        raise ValueError("invalid MoGe-support parameters")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    partial_normals = _estimate_normals(partial, neighbours=int(normal_neighbours))
    moge_normals = _estimate_normals(moge, neighbours=int(normal_neighbours))
    distance, nearest = cKDTree(moge).query(partial, k=1, workers=-1)
    agreement = np.abs((partial_normals * moge_normals[nearest]).sum(axis=1))
    supported = ((distance <= float(maximum_distance_ratio) * diagonal)
                 & (agreement >= float(minimum_normal_agreement)))
    unsupported = ~supported
    return unsupported, {
        "method": "partial_moge_local_surface_support",
        "partial_diagonal": diagonal,
        "maximum_distance_ratio": float(maximum_distance_ratio),
        "minimum_normal_agreement": float(minimum_normal_agreement),
        "supported_partial": int(supported.sum()),
        "unsupported_partial": int(unsupported.sum()),
        "unsupported_fraction": float(unsupported.mean()),
        "nearest_distance_median": float(np.median(distance)),
        "nearest_distance_p90": float(np.quantile(distance, .90)),
    }


def _metric_components(points: np.ndarray, *, radius: float) -> tuple[int, np.ndarray]:
    """Connected components of a small observed residual point set."""
    if len(points) == 0:
        return 0, np.empty(0, dtype=np.int64)
    edges = cKDTree(points).query_pairs(r=float(radius), output_type="ndarray")
    if len(edges) == 0:
        return len(points), np.arange(len(points), dtype=np.int64)
    graph = sparse.coo_matrix(
        (np.ones(2 * len(edges), dtype=np.float64),
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(points), len(points)), dtype=np.float64,
    ).tocsr()
    return csgraph.connected_components(graph, directed=False)


def _aggregate_anchor_targets(
    partial: np.ndarray,
    prior: np.ndarray,
    partial_ids: np.ndarray,
    prior_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make one robust target per prior Gaussian without duplicating slots."""
    unique, inverse = np.unique(prior_ids, return_inverse=True)
    targets = np.empty((len(unique), 3), dtype=np.float64)
    for index in range(len(unique)):
        targets[index] = np.median(partial[partial_ids[inverse == index]], axis=0)
    return unique, targets


def structure_aware_arap_gaussian_adaptation(
    prior: np.ndarray,
    partial: np.ndarray,
    *,
    locked_prior_mask: np.ndarray | None = None,
    bridge_unknown_partial_mask: np.ndarray | None = None,
    normal_neighbours: int = 24,
    normal_agreement: float = .65,
    bridge_unknown_normal_agreement: float = .35,
    maximum_correspondence_ratio: float = .20,
    residual_ratio: float = .045,
    stable_ratio: float = .020,
    component_radius_ratio: float = .018,
    minimum_component_points: int = 64,
    minimum_directional_coherence: float = .80,
    graph_neighbours: int = 10,
    edge_ratio: float = 1.8,
    influence_radius_ratios: tuple[float, ...] = (.12, .16, .21, .27, .33),
    maximum_influence_fraction: float = .65,
    arap_iterations: int = 5,
    screening: float = 1e-4,
    maximum_edge_stretch: float = 2.4,
    minimum_edge_compression: float = .52,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Use coherent structure-compatible partial residuals as ARAP anchors.

    ``locked_prior_mask`` supports a coarse-to-fine continuation without
    re-editing already explained geometry.  It is intentionally a per-carrier
    boolean mask rather than a category- or part-specific instruction: a
    previous iteration may lock its exact anchors and stable visible support,
    while a later iteration can only use independently supported residuals on
    the remaining carrier.
    """
    prior, partial = np.asarray(prior, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if (normal_neighbours < 6 or not 0. <= normal_agreement <= 1. or maximum_correspondence_ratio <= 0.
            or not 0. <= bridge_unknown_normal_agreement <= normal_agreement
            or not 0. < stable_ratio < residual_ratio < maximum_correspondence_ratio
            or component_radius_ratio <= 0. or minimum_component_points < 6
            or not 0. <= minimum_directional_coherence <= 1. or graph_neighbours < 2 or edge_ratio <= 0.
            or not influence_radius_ratios or not 0. < maximum_influence_fraction <= 1.
            or arap_iterations < 1 or screening <= 0. or maximum_edge_stretch <= 1.
            or not 0. < minimum_edge_compression < 1.):
        raise ValueError("invalid structure-aware ARAP parameters")
    if locked_prior_mask is None:
        inherited_locked = np.zeros(len(prior), dtype=bool)
    else:
        inherited_locked = np.asarray(locked_prior_mask, dtype=bool).reshape(-1)
        if inherited_locked.shape != (len(prior),):
            raise ValueError("locked_prior_mask must be a boolean vector with one entry per prior point")
    if bridge_unknown_partial_mask is None:
        bridge_unknown = np.zeros(len(partial), dtype=bool)
    else:
        bridge_unknown = np.asarray(bridge_unknown_partial_mask, dtype=bool).reshape(-1)
        if bridge_unknown.shape != (len(partial),):
            raise ValueError("bridge_unknown_partial_mask must have one entry per partial point")
    empty = {
        "anchors": np.zeros(len(prior), dtype=bool), "stable": inherited_locked.copy(),
        "influence": np.zeros(len(prior), dtype=bool), "boundary": np.zeros(len(prior), dtype=bool),
        "moved": np.zeros(len(prior), dtype=bool),
    }
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    prior_normals = _estimate_normals(prior, neighbours=int(normal_neighbours))
    partial_normals = _estimate_normals(partial, neighbours=int(normal_neighbours))
    distance, nearest = cKDTree(prior).query(partial, k=1, workers=-1)
    agreement = np.abs((partial_normals * prior_normals[nearest]).sum(axis=1))
    standard_compatible = ((agreement >= float(normal_agreement))
                           & (distance <= float(maximum_correspondence_ratio) * diagonal))
    unknown_compatible = (bridge_unknown
                          & (agreement >= float(bridge_unknown_normal_agreement))
                          & (distance <= float(maximum_correspondence_ratio) * diagonal))
    compatible = standard_compatible | unknown_compatible
    # Do not let a relaxed bridge-unknown match freeze the carrier.  Stable
    # support must remain backed by the stricter, direct local-surface test.
    stable_ids = np.unique(nearest[standard_compatible & (distance <= float(stable_ratio) * diagonal)])
    inherited_ids = np.flatnonzero(inherited_locked)
    stable_ids = np.unique(np.r_[stable_ids, inherited_ids])
    residual = compatible & (distance >= float(residual_ratio) * diagonal)
    partial_ids = np.flatnonzero(residual)
    if len(partial_ids) < int(minimum_component_points):
        return prior.copy(), {
            "active": False, "method": "structure_aware_partial_arap_gaussian_adaptation",
            "reason": "insufficient_structure_compatible_residuals", "diagonal": diagonal,
            "structure_compatible_partial": int(compatible.sum()), "residual_partial": int(len(partial_ids)),
            "bridge_unknown_partial": int(bridge_unknown.sum()),
            "bridge_unknown_residual_partial": int(bridge_unknown[partial_ids].sum()),
        }, empty
    component_count, labels = _metric_components(
        partial[partial_ids], radius=float(component_radius_ratio) * diagonal,
    )
    selected = np.zeros(len(partial_ids), dtype=bool)
    components: list[dict] = []
    vectors = partial[partial_ids] - prior[nearest[partial_ids]]
    for label in range(component_count):
        member = labels == label
        component_size = int(member.sum())
        if component_size < int(minimum_component_points):
            continue
        vector = vectors[member]
        median = np.median(vector, axis=0)
        magnitude = float(np.linalg.norm(median))
        if magnitude <= 1e-9:
            continue
        coherence = float(np.mean((vector @ median) / np.maximum(
            np.linalg.norm(vector, axis=1) * magnitude, 1e-12,
        )))
        if coherence < float(minimum_directional_coherence):
            continue
        # An inherited lock is a hard physical constraint from an earlier,
        # independently verified stage.  Do not turn it into a new anchor,
        # even when its nearest partial sample still has a large residual.
        unlocked_member = member.copy()
        unlocked_member[member] = ~inherited_locked[nearest[partial_ids[member]]]
        unlocked_size = int(unlocked_member.sum())
        if unlocked_size < int(minimum_component_points):
            continue
        selected[unlocked_member] = True
        components.append({
            "component": int(label), "partial_points": component_size,
            "unlocked_partial_points": unlocked_size,
            "bridge_unknown_partial_points": int(bridge_unknown[partial_ids[member]].sum()),
            "bridge_unknown_fraction": float(bridge_unknown[partial_ids[member]].mean()),
            "median_residual": median.tolist(), "median_residual_norm": magnitude,
            "directional_coherence": coherence,
        })
    chosen_partial = partial_ids[selected]
    if len(chosen_partial) < int(minimum_component_points):
        return prior.copy(), {
            "active": False, "method": "structure_aware_partial_arap_gaussian_adaptation",
            "reason": "no_coherent_structural_residual_component", "diagonal": diagonal,
            "structure_compatible_partial": int(compatible.sum()), "residual_partial": int(len(partial_ids)),
            "bridge_unknown_partial": int(bridge_unknown.sum()),
            "bridge_unknown_residual_partial": int(bridge_unknown[partial_ids].sum()),
            "residual_components": int(component_count),
        }, empty
    anchor_ids, anchor_targets = _aggregate_anchor_targets(
        partial, prior, chosen_partial, nearest[chosen_partial],
    )
    # Inherited locks always win over new targets.  Current low-residual slots
    # remain fixed unless they are selected as a fresh exact anchor.
    stable_ids = np.setdiff1d(stable_ids, anchor_ids, assume_unique=False)
    graph, metric, edge_count = _local_gaussian_graph(
        prior, neighbours=int(graph_neighbours), edge_ratio=float(edge_ratio),
    )
    if edge_count == 0:
        return prior.copy(), {"active": False, "method": "structure_aware_partial_arap_gaussian_adaptation",
                              "reason": "empty_prior_surface_graph"}, empty
    geodesic = csgraph.dijkstra(metric, directed=False, indices=anchor_ids, min_only=True)
    last_trial: dict | None = None
    for radius_ratio in influence_radius_ratios:
        influence = np.isfinite(geodesic) & (geodesic <= float(radius_ratio) * diagonal)
        if not influence[anchor_ids].all() or float(influence.mean()) > float(maximum_influence_fraction):
            continue
        edited, solved = _arap_surface_field(
            prior, graph, influence, anchor_ids, anchor_targets, stable_ids,
            iterations=int(arap_iterations), screening=float(screening),
        )
        strain = _edge_strain(prior, edited, graph)
        magnitude = np.linalg.norm(edited - prior, axis=1)
        trial = {"radius_ratio": float(radius_ratio), "influence_gaussians": int(influence.sum()),
                 "influence_fraction": float(influence.mean()), **solved, **strain}
        if (strain["edge_stretch_p01"] >= float(minimum_edge_compression)
                and strain["edge_stretch_p999"] <= float(maximum_edge_stretch)):
            boundary = np.zeros(len(prior), dtype=bool)
            coo = graph.tocoo(); crossing = influence[coo.row] ^ influence[coo.col]
            boundary[np.unique(np.where(influence[coo.row[crossing]], coo.row[crossing], coo.col[crossing]))] = True
            masks = {
                "anchors": np.zeros(len(prior), dtype=bool), "stable": np.zeros(len(prior), dtype=bool),
                "influence": influence, "boundary": boundary, "moved": magnitude > 1e-8,
            }
            masks["anchors"][anchor_ids] = True; masks["stable"][stable_ids] = True
            return edited, {
                "active": True, "method": "structure_aware_partial_arap_gaussian_adaptation",
                "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
                "carrier_slots_preserved": bool(len(edited) == len(prior)), "diagonal": diagonal,
                "structure_compatible_partial": int(compatible.sum()), "residual_partial": int(len(partial_ids)),
                "bridge_unknown_partial": int(bridge_unknown.sum()),
                "bridge_unknown_residual_partial": int(bridge_unknown[partial_ids].sum()),
                "residual_components": int(component_count), "selected_components": components,
                "anchors": int(len(anchor_ids)), "stable_visible_gaussians": int(len(stable_ids)),
                "inherited_locked_gaussians": int(inherited_locked.sum()),
                "surface_graph_edges": int(edge_count), "selected_band": trial,
                "moved_gaussians": int((magnitude > 1e-8).sum()),
                "mean_displacement": float(magnitude.mean()), "maximum_displacement": float(magnitude.max(initial=0.)),
                "anchor_residual_before_median": float(np.median(np.linalg.norm(
                    prior[anchor_ids] - anchor_targets, axis=1))),
                "anchor_residual_after_median": float(np.median(np.linalg.norm(
                    edited[anchor_ids] - anchor_targets, axis=1))),
                "field": "structure_compatible_partial_anchors_with_geodesic_arap_gaussian_continuation",
            }, masks
        last_trial = trial
    return prior.copy(), {
        "active": False, "method": "structure_aware_partial_arap_gaussian_adaptation", "diagonal": diagonal,
        "structure_compatible_partial": int(compatible.sum()), "residual_partial": int(len(partial_ids)),
        "bridge_unknown_partial": int(bridge_unknown.sum()),
        "bridge_unknown_residual_partial": int(bridge_unknown[partial_ids].sum()),
        "residual_components": int(component_count), "selected_components": components,
        "anchors": int(len(anchor_ids)), "stable_visible_gaussians": int(len(stable_ids)),
        "reason": "no_structure_aware_arap_band_preserves_local_geometry", "last_trial": last_trial,
    }, empty
