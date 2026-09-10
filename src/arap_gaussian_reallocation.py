"""Partial-depth anchored ARAP editing of a fixed Gaussian-prior carrier.

This is the local-geometry counterpart to global Sim(3).  A registered prior
can agree globally yet leave a visible appendage short or misplaced.  We use
the partial scan as hard 3-D surface evidence, but do not translate its
nearest prior points independently.  Instead, an as-rigid-as-possible (ARAP)
field on the prior's Gaussian graph distributes the displacement through a
geodesically bounded band between the observed residual and stable support.

The method edits only Gaussian means; its 100k carrier slots, unobserved
support, input partial, and all decisions remain fixed/deterministic.  It
does not concatenate point clouds or use ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from src.camera_conditioned_gaussian_adaptation import _edge_strain, _local_gaussian_graph, _solve_vector_system
from src.observation_anchored_gaussian_reallocation import (
    _component_observation_pairs,
    _expand_screen_anchor_slots,
)


class _PCAOrthographicProjector:
    """Deterministic partial/prior self-view with a positive depth buffer."""

    def __init__(self, centre: np.ndarray, basis: np.ndarray, *, depth_axis: int, sign: int,
                 lower: np.ndarray, extent: np.ndarray, image_shape: tuple[int, int]):
        self.centre = np.asarray(centre, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)
        self.depth_axis = int(depth_axis)
        self.sign = int(sign)
        self.planar_axes = [axis for axis in range(3) if axis != self.depth_axis]
        self.lower = np.asarray(lower, dtype=np.float64)
        self.extent = np.asarray(extent, dtype=np.float64)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.depth_origin = 0.

    def configure_depth_origin(self, points: np.ndarray) -> None:
        coordinates = (np.asarray(points, dtype=np.float64) - self.centre) @ self.basis
        self.depth_origin = float(np.min(float(self.sign) * coordinates[:, self.depth_axis])) - 1.

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = (np.asarray(points, dtype=np.float64) - self.centre) @ self.basis
        plane = coordinates[:, self.planar_axes]
        height, width = self.image_shape
        uv = (plane - self.lower) / self.extent * np.array([width - 1, height - 1], dtype=np.float64)
        depth = float(self.sign) * coordinates[:, self.depth_axis] - self.depth_origin
        return uv, depth


def _pca_self_view_projectors(
    partial: np.ndarray,
    prior: np.ndarray,
    *,
    views: int,
    resolution: int,
) -> list[tuple[str, _PCAOrthographicProjector]]:
    """Build signed-PCA self-cameras shared by registered partial and prior."""
    if int(views) <= 0:
        return []
    centre = prior.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(prior - centre, full_matrices=False)
    basis = right.T
    combined = np.concatenate(((partial - centre) @ basis, (prior - centre) @ basis), axis=0)
    result: list[tuple[str, _PCAOrthographicProjector]] = []
    for axis, sign in [(axis, sign) for axis in range(3) for sign in (-1, 1)][:int(views)]:
        planar_axes = [candidate for candidate in range(3) if candidate != axis]
        plane = combined[:, planar_axes]
        lower = plane.min(axis=0)
        extent = np.maximum(plane.max(axis=0) - lower, 1e-9)
        projector = _PCAOrthographicProjector(
            centre, basis, depth_axis=axis, sign=sign, lower=lower, extent=extent,
            image_shape=(int(resolution), int(resolution)),
        )
        projector.configure_depth_origin(np.concatenate((partial, prior), axis=0))
        result.append((f"pca_axis_{axis}_{'positive' if sign > 0 else 'negative'}", projector))
    return result


def _collect_observation_evidence(
    partial: np.ndarray,
    prior: np.ndarray,
    saved_projector,
    *,
    virtual_views: int,
    virtual_resolution: int,
    maximum_screen_distance: float,
    stable_screen_distance: float,
    residual_ratio: float,
    minimum_component_pixels: int,
    component_radius: float,
    minimum_directional_coherence: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collect positive residual observations from Camera-1 and PCA self-views."""
    sources: list[tuple[str, object]] = [("saved_camera_1", saved_projector)]
    sources.extend(_pca_self_view_projectors(
        partial, prior, views=int(virtual_views), resolution=int(virtual_resolution),
    ))
    all_pairs: list[np.ndarray] = []
    stable: list[np.ndarray] = []
    per_view: list[dict] = []
    for source_id, (name, projector) in enumerate(sources):
        # Only the physical saved camera can interpret a screen-space gap as a
        # silhouette/depth discrepancy.  A virtual re-render of an incomplete
        # scan has unknown occlusion, so it contributes strictly positive
        # mutual overlap evidence (screen gap disabled) and stable support.
        local_maximum_screen_distance = (
            float(maximum_screen_distance) if source_id == 0 else float(stable_screen_distance)
        )
        pairs, stable_ids, info = _component_observation_pairs(
            partial, prior, projector, maximum_screen_distance=local_maximum_screen_distance,
            stable_screen_distance=float(stable_screen_distance), residual_ratio=float(residual_ratio),
            minimum_component_pixels=int(minimum_component_pixels), component_radius=float(component_radius),
            minimum_directional_coherence=float(minimum_directional_coherence),
            mutual_only=bool(source_id > 0),
        )
        if len(pairs):
            all_pairs.append(pairs)
        stable.append(stable_ids)
        per_view.append({"view": name, **info})
    combined = np.concatenate(all_pairs, axis=0) if all_pairs else np.empty((0, 4), dtype=np.float64)
    stable_ids = np.unique(np.concatenate(stable)) if stable else np.empty(0, dtype=np.int64)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    return combined, stable_ids, {
        "active": bool(len(combined)), "diagonal": diagonal,
        "observation_views": len(sources), "virtual_views": int(virtual_views),
        "selected_candidate_pixels": int(len(combined)),
        "stable_visible_prior": int(len(stable_ids)), "per_view": per_view,
        "evidence_policy": "positive_partial_surface_residuals_only; unobserved_pixels_are_not_constraints",
    }


def _band_boundary(graph: sparse.csr_matrix, influence: np.ndarray) -> np.ndarray:
    """Return local-band vertices connected to a fixed exterior vertex."""
    coo = graph.tocoo()
    crossing = influence[coo.row] ^ influence[coo.col]
    return np.unique(np.where(influence[coo.row[crossing]], coo.row[crossing], coo.col[crossing]))


def _arap_surface_field(
    prior: np.ndarray,
    graph: sparse.csr_matrix,
    influence: np.ndarray,
    anchor_ids: np.ndarray,
    anchor_targets: np.ndarray,
    stable_ids: np.ndarray,
    *,
    iterations: int,
    screening: float,
) -> tuple[np.ndarray, dict]:
    """Solve a local ARAP field with exact partial and zero-motion boundaries."""
    count = len(prior)
    boundary = _band_boundary(graph, influence)
    fixed = ~influence
    fixed[np.asarray(stable_ids, dtype=np.int64)] = True
    fixed[boundary] = True
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError("anchor slots must be collision-free")
    fixed[anchors] = True
    target = prior.copy()
    target[anchors] = anchor_targets
    unknown = np.flatnonzero(~fixed)
    fixed_ids = np.flatnonzero(fixed)
    if len(unknown) == 0:
        return target, {"cg_status": [0, 0, 0], "boundary_gaussians": int(len(boundary)),
                        "movable_gaussians": 0, "iterations": 0}
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - graph
    system = laplacian[unknown][:, unknown].tocsr() + sparse.eye(
        len(unknown), format="csr", dtype=np.float64,
    ) * float(screening)
    # One undirected representation carries the ARAP edge term.
    row, col = sparse.triu(graph, k=1).nonzero()
    weight = np.asarray(graph[row, col]).reshape(-1)
    active_edge = influence[row] | influence[col]
    row, col, weight = row[active_edge], col[active_edge], weight[active_edge]
    rest_edge = prior[row] - prior[col]
    deformed = target.copy()
    rotations = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    final_status = [0, 0, 0]
    for _ in range(int(iterations)):
        deformed_edge = deformed[row] - deformed[col]
        covariance = np.zeros((count, 3, 3), dtype=np.float64)
        contribution = weight[:, None, None] * np.einsum("ni,nj->nij", deformed_edge, rest_edge)
        np.add.at(covariance, row, contribution)
        np.add.at(covariance, col, contribution)
        active_vertices = np.flatnonzero(influence)
        if len(active_vertices):
            left, _, right = np.linalg.svd(covariance[active_vertices], full_matrices=False)
            local_rotation = left @ right
            determinant = np.linalg.det(local_rotation)
            if np.any(determinant < 0.):
                left[determinant < 0., :, -1] *= -1.
                local_rotation = left @ right
            rotations[active_vertices] = local_rotation
        edge_rotation = .5 * (rotations[row] + rotations[col])
        desired_edge = np.einsum("nij,nj->ni", edge_rotation, rest_edge)
        rhs = np.zeros((count, 3), dtype=np.float64)
        np.add.at(rhs, row, weight[:, None] * desired_edge)
        np.add.at(rhs, col, -weight[:, None] * desired_edge)
        # q_u is solved with the actual fixed (partial and boundary) means.
        rhs_unknown = rhs[unknown] - laplacian[unknown][:, fixed_ids] @ target[fixed_ids]
        solution, final_status = _solve_vector_system(system, np.asarray(rhs_unknown))
        deformed[unknown] = solution
        deformed[fixed_ids] = target[fixed_ids]
    return deformed, {
        "cg_status": final_status, "boundary_gaussians": int(len(boundary)),
        "movable_gaussians": int(len(unknown)), "iterations": int(iterations),
    }


def partial_depth_arap_gaussian_adaptation(
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
    virtual_views: int = 0,
    virtual_resolution: int = 384,
    influence_radius_ratios: tuple[float, ...] = (.12, .16, .21, .27, .33),
    maximum_influence_fraction: float = .65,
    neighbours: int = 10,
    edge_ratio: float = 1.8,
    arap_iterations: int = 5,
    screening: float = 1e-4,
    maximum_edge_stretch: float = 2.2,
    minimum_edge_compression: float = .52,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Adapt visible prior geometry with a partial-depth anchored ARAP field."""
    prior, partial = np.asarray(prior, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if (maximum_screen_distance <= stable_screen_distance or residual_ratio <= 0.
            or minimum_component_pixels < 6 or component_radius <= 0. or allocation_neighbours < 1
            or not 0. <= minimum_directional_coherence <= 1.
            or maximum_anchor_residual_ratio <= 0. or not influence_radius_ratios
            or not 0. < maximum_influence_fraction <= 1. or neighbours < 2 or edge_ratio <= 0.
            or arap_iterations < 1 or screening <= 0. or maximum_edge_stretch <= 1.
            or not 0. < minimum_edge_compression < 1. or not 0 <= virtual_views <= 6
            or virtual_resolution < 32):
        raise ValueError("invalid ARAP Gaussian adaptation parameters")
    empty = {
        "anchors": np.zeros(len(prior), dtype=bool), "stable": np.zeros(len(prior), dtype=bool),
        "influence": np.zeros(len(prior), dtype=bool), "boundary": np.zeros(len(prior), dtype=bool),
        "moved": np.zeros(len(prior), dtype=bool),
    }
    pairs, stable_ids, evidence = _collect_observation_evidence(
        partial, prior, projector, virtual_views=int(virtual_views), virtual_resolution=int(virtual_resolution),
        maximum_screen_distance=float(maximum_screen_distance), stable_screen_distance=float(stable_screen_distance),
        residual_ratio=float(residual_ratio), minimum_component_pixels=int(minimum_component_pixels),
        component_radius=float(component_radius), minimum_directional_coherence=float(minimum_directional_coherence),
    )
    if not len(pairs):
        return prior.copy(), {**evidence, "active": False, "method": "partial_depth_arap_gaussian_adaptation"}, empty
    diagonal = float(evidence["diagonal"])
    anchors, anchor_info = _expand_screen_anchor_slots(
        pairs, partial, prior, maximum_residual=float(maximum_anchor_residual_ratio) * diagonal,
        allocation_neighbours=int(allocation_neighbours), blocked_prior_ids=stable_ids,
    )
    if len(anchors) < int(minimum_component_pixels):
        return prior.copy(), {
            **evidence, **anchor_info, "active": False, "method": "partial_depth_arap_gaussian_adaptation",
            "reason": "insufficient_collision_free_observation_anchors",
        }, empty
    partial_ids, anchor_ids = anchors[:, 0], anchors[:, 1]
    anchor_targets = partial[partial_ids]
    graph, metric, edge_count = _local_gaussian_graph(prior, neighbours=int(neighbours), edge_ratio=float(edge_ratio))
    if edge_count == 0:
        return prior.copy(), {
            **evidence, **anchor_info, "active": False, "method": "partial_depth_arap_gaussian_adaptation",
            "reason": "empty_local_gaussian_surface_graph",
        }, empty
    geodesic = csgraph.dijkstra(metric, directed=False, indices=anchor_ids, min_only=True)
    stable_ids = np.setdiff1d(stable_ids, anchor_ids, assume_unique=False)
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
        trial = {
            "radius_ratio": float(radius_ratio), "influence_gaussians": int(influence.sum()),
            "influence_fraction": float(influence.mean()), **solved, **strain,
        }
        if (strain["edge_stretch_p01"] >= float(minimum_edge_compression)
                and strain["edge_stretch_p999"] <= float(maximum_edge_stretch)):
            boundary = _band_boundary(graph, influence)
            masks = {
                "anchors": np.zeros(len(prior), dtype=bool), "stable": np.zeros(len(prior), dtype=bool),
                "influence": influence, "boundary": np.zeros(len(prior), dtype=bool),
                "moved": magnitude > 1e-8,
            }
            masks["anchors"][anchor_ids] = True
            masks["stable"][stable_ids] = True
            masks["boundary"][boundary] = True
            return edited, {
                "active": True, "method": "partial_depth_arap_gaussian_adaptation",
                "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
                "carrier_slots_preserved": bool(len(edited) == len(prior)), **evidence, **anchor_info,
                "anchors": int(len(anchor_ids)), "stable_visible_gaussians": int(len(stable_ids)),
                "surface_graph_edges": int(edge_count), "selected_band": trial,
                "moved_gaussians": int((magnitude > 1e-8).sum()),
                "mean_displacement": float(magnitude.mean()),
                "maximum_displacement": float(magnitude.max(initial=0.)),
                "anchor_residual_before_median": float(np.median(np.linalg.norm(
                    prior[anchor_ids] - anchor_targets, axis=1,
                ))),
                "anchor_residual_after_median": float(np.median(np.linalg.norm(
                    edited[anchor_ids] - anchor_targets, axis=1,
                ))),
                "field": "partial_depth_anchors_with_geodesic_arap_gaussian_continuation",
            }, masks
        last_trial = trial
    return prior.copy(), {
        **evidence, **anchor_info, "active": False, "method": "partial_depth_arap_gaussian_adaptation",
        "reason": "no_arap_continuation_band_preserves_local_geometry", "last_trial": last_trial,
    }, empty
