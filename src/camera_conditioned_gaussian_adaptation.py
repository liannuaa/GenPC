"""Camera-conditioned 3DGS-mean adaptation from a partial observation.

The complete generative prior is represented by a fixed set of Gaussian means.
This module decides which means may move using *only* visible Camera-1
evidence: pixel-indexed partial/prior residuals are separated into coherent
high-residual components (editable), low-residual observations (locked), and
the remaining complete prior (protected).  A screened vector field then
propagates the full camera-frame ``(du, dv, dz)`` correction on the local
Gaussian surface graph.  No category-specific part labels or ground truth are
used.

The output preserves all prior slots.  It is therefore directly usable as a
geometry-only 3D Gaussian field and can also be decoded as a complete point
cloud for evaluation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

from src.attachment_aware_gaussian import _coherent_residual_selection


def _solve_vector_system(matrix: sparse.csr_matrix, rhs: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Solve three screened Gaussian-mean displacement coordinates."""
    result = np.zeros_like(rhs, dtype=np.float64)
    status: list[int] = []
    for axis in range(3):
        try:
            value, code = sparse_linalg.cg(matrix, rhs[:, axis], rtol=1e-5, atol=0., maxiter=360)
        except TypeError:  # SciPy < 1.12
            value, code = sparse_linalg.cg(matrix, rhs[:, axis], tol=1e-5, maxiter=360)
        result[:, axis] = value
        status.append(int(code))
    return result, status


def _local_gaussian_graph(prior: np.ndarray, *, neighbours: int,
                          edge_ratio: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix, int]:
    """Return affinity and metric kNN graphs without bridging surface gaps."""
    count = len(prior)
    query_k = min(int(neighbours) + 1, count)
    if query_k < 2:
        empty = sparse.csr_matrix((count, count), dtype=np.float64)
        return empty, empty, 0
    distances, indices = cKDTree(prior).query(prior, k=query_k, workers=-1)
    local = np.median(distances[:, 1:], axis=1)
    local = np.maximum(local, max(float(np.quantile(local, .01)) * .25, 1e-9))
    source = np.repeat(np.arange(count, dtype=np.int64), query_k - 1)
    target = indices[:, 1:].reshape(-1).astype(np.int64)
    length = distances[:, 1:].reshape(-1)
    keep = source < target
    source, target, length = source[keep], target[keep], length[keep]
    scale = np.sqrt(local[source] * local[target])
    keep = length <= float(edge_ratio) * scale
    source, target, length, scale = source[keep], target[keep], length[keep], scale[keep]
    affinity = np.exp(-np.square(length / np.maximum(scale, 1e-12)))
    graph = sparse.coo_matrix(
        (np.r_[affinity, affinity], (np.r_[source, target], np.r_[target, source])),
        shape=(count, count), dtype=np.float64,
    ).tocsr()
    metric = sparse.coo_matrix(
        (np.r_[length, length], (np.r_[source, target], np.r_[target, source])),
        shape=(count, count), dtype=np.float64,
    ).tocsr()
    graph.sum_duplicates()
    metric.sum_duplicates()
    return graph, metric, int(len(source))


def _edge_strain(prior: np.ndarray, edited: np.ndarray, graph: sparse.csr_matrix) -> dict:
    """Report local Gaussian-carrier strain; there is no mesh topology to flip."""
    rows, cols = sparse.triu(graph, k=1).nonzero()
    if len(rows) == 0:
        return {"edge_stretch_p01": 1., "edge_stretch_p50": 1., "edge_stretch_p99": 1., "edge_stretch_p999": 1.}
    original = np.linalg.norm(prior[rows] - prior[cols], axis=1)
    candidate = np.linalg.norm(edited[rows] - edited[cols], axis=1)
    ratio = candidate / np.maximum(original, 1e-12)
    return {
        "edge_stretch_p01": float(np.quantile(ratio, .01)),
        "edge_stretch_p50": float(np.quantile(ratio, .50)),
        "edge_stretch_p99": float(np.quantile(ratio, .99)),
        "edge_stretch_p999": float(np.quantile(ratio, .999)),
    }


def camera_conditioned_gaussian_adaptation(
    prior: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    camera_axes: np.ndarray,
    max_pixel_distance: float = 1.,
    residual_ratio: float = .12,
    stable_ratio: float = .045,
    screen_component_radius: float = 3.,
    minimum_component_pairs: int = 128,
    minimum_directional_coherence: float = .85,
    minimum_camera_axis_dominance: float = .65,
    component_direction_cosine: float = .92,
    retain_all_compatible_components: bool = True,
    neighbours: int = 10,
    edge_ratio: float = 1.8,
    influence_geodesic_radius_ratio: float = .18,
    maximum_influence_fraction: float = .55,
    # A soft camera data term lets the field spread across the same connected
    # surface before it meets a locked observation. This avoids a sharp local
    # jump at a limb/body attachment while retaining the measured direction.
    data_weight: float = .25,
    screening: float = .015,
    maximum_displacement_ratio: float = .30,
    minimum_improvement: float = .08,
    minimum_depth_improvement: float = .02,
    locked_prior_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Adapt Gaussian means with coherent Camera-1 projection-and-depth evidence.

    ``camera_axes`` stores the Camera-1 row axes. Residual targets are first
    represented in this frame, so the saved image-plane and its physical depth
    are jointly constrained before returning to the common world frame.
    """
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    axes = np.asarray(camera_axes, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3), rtol=1e-4, atol=1e-5):
        raise ValueError("camera_axes must be a (3, 3) orthonormal row-axis matrix")
    if not (0. < stable_ratio < residual_ratio and 0. < influence_geodesic_radius_ratio
            and 0. < maximum_influence_fraction <= 1. and data_weight > 0. and screening > 0.
            and maximum_displacement_ratio > 0. and 0. < minimum_improvement < 1.
            and 0. <= minimum_depth_improvement < 1.):
        raise ValueError("invalid camera-conditioned Gaussian adaptation bounds")
    zero = np.zeros_like(prior)
    empty_masks = {
        "editable": np.zeros(len(prior), dtype=bool),
        "locked": np.zeros(len(prior), dtype=bool),
        "protected": np.ones(len(prior), dtype=bool),
        "moved": np.zeros(len(prior), dtype=bool),
    }
    evidence, base = _coherent_residual_selection(
        prior, partial, projector, axes,
        max_pixel_distance=float(max_pixel_distance),
        minimum_residual_ratio=float(residual_ratio),
        screen_component_radius=float(screen_component_radius),
        minimum_component_pairs=int(minimum_component_pairs),
        minimum_directional_coherence=float(minimum_directional_coherence),
        minimum_camera_axis_dominance=float(minimum_camera_axis_dominance),
        component_direction_cosine=float(component_direction_cosine),
        retain_all_compatible_components=bool(retain_all_compatible_components),
    )
    if evidence is None:
        return prior.copy(), {"active": False, "method": "camera_conditioned_gaussian_mean_adaptation", **base}, empty_masks
    diagonal = float(evidence["diagonal"])
    prior_ids = np.asarray(evidence["prior_ids"], dtype=np.int64)
    partial_ids = np.asarray(evidence["partial_ids"], dtype=np.int64)
    residual_norm = np.asarray(evidence["residual_norm"], dtype=np.float64)
    records = np.asarray(evidence["record_indices"], dtype=np.int64)
    selected_prior = prior_ids[records]
    selected_partial = partial_ids[records]
    controls, inverse = np.unique(selected_prior, return_inverse=True)
    raw_camera_residual = (partial[selected_partial] - prior[selected_prior]) @ axes.T
    target_camera = np.zeros((len(controls), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(controls)).astype(np.float64)
    np.add.at(target_camera, inverse, raw_camera_residual)
    target_camera /= counts[:, None]
    target_world = target_camera @ axes

    locked = np.zeros(len(prior), dtype=bool)
    locked[np.unique(prior_ids[residual_norm <= float(stable_ratio) * diagonal])] = True
    inherited_locked = np.zeros(len(prior), dtype=bool)
    if locked_prior_mask is not None:
        inherited_locked = np.asarray(locked_prior_mask, dtype=bool)
        if inherited_locked.shape != (len(prior),):
            raise ValueError("locked_prior_mask must be a boolean vector matching prior slots")
        locked |= inherited_locked
    editable = np.zeros(len(prior), dtype=bool)
    editable[controls] = True
    editable &= ~locked
    controls = np.flatnonzero(editable)
    if len(controls) < max(6, int(minimum_component_pairs) // 2):
        return prior.copy(), {"active": False, "method": "camera_conditioned_gaussian_mean_adaptation", **base,
                              "reason": "coherent_controls_conflict_with_locked_visible_support"}, empty_masks
    # Re-aggregate targets after controls that conflict with locked support are removed.
    keep = editable[selected_prior]
    selected_prior, raw_camera_residual = selected_prior[keep], raw_camera_residual[keep]
    control_lookup = np.full(len(prior), -1, dtype=np.int64)
    control_lookup[controls] = np.arange(len(controls), dtype=np.int64)
    target_camera = np.zeros((len(controls), 3), dtype=np.float64)
    inverse = control_lookup[selected_prior]
    counts = np.bincount(inverse, minlength=len(controls)).astype(np.float64)
    np.add.at(target_camera, inverse, raw_camera_residual)
    target_camera /= np.maximum(counts[:, None], 1.)
    target_world = target_camera @ axes

    graph, metric, edge_count = _local_gaussian_graph(prior, neighbours=int(neighbours), edge_ratio=float(edge_ratio))
    if edge_count == 0:
        return prior.copy(), {"active": False, "method": "camera_conditioned_gaussian_mean_adaptation", **base,
                              "reason": "empty_gaussian_surface_graph"}, empty_masks
    geodesic = csgraph.dijkstra(metric, directed=False, indices=controls, min_only=True)
    influence = np.isfinite(geodesic) & (geodesic <= float(influence_geodesic_radius_ratio) * diagonal)
    if float(influence.mean()) > float(maximum_influence_fraction):
        return prior.copy(), {"active": False, "method": "camera_conditioned_gaussian_mean_adaptation", **base,
                              "reason": "overbroad_camera_supported_edit", "influence_gaussians": int(influence.sum())}, empty_masks
    protected = ~influence
    fixed = locked | protected
    movable = influence & ~locked
    control_mask = movable[controls]
    controls, target_camera, target_world = controls[control_mask], target_camera[control_mask], target_world[control_mask]
    if len(controls) < max(6, int(minimum_component_pairs) // 2):
        return prior.copy(), {"active": False, "method": "camera_conditioned_gaussian_mean_adaptation", **base,
                              "reason": "camera_supported_edit_overconstrained"}, empty_masks
    movable_ids = np.flatnonzero(movable)
    index = np.full(len(prior), -1, dtype=np.int64)
    index[movable_ids] = np.arange(len(movable_ids), dtype=np.int64)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - graph
    system = laplacian[movable_ids][:, movable_ids].tocsr() + sparse.eye(
        len(movable_ids), format="csr", dtype=np.float64,
    ) * float(screening)
    local_controls = index[controls]
    system = system + sparse.coo_matrix(
        (np.full(len(local_controls), float(data_weight)), (local_controls, local_controls)),
        shape=system.shape, dtype=np.float64,
    ).tocsr()
    rhs = np.zeros((len(movable_ids), 3), dtype=np.float64)
    rhs[local_controls] = float(data_weight) * target_world
    field, cg_status = _solve_vector_system(system, rhs)
    raw_displacement = np.zeros_like(prior)
    raw_displacement[movable_ids] = field
    maximum_displacement = float(maximum_displacement_ratio) * diagonal
    magnitude = np.linalg.norm(raw_displacement, axis=1, keepdims=True)
    raw_displacement *= np.minimum(1., maximum_displacement / np.maximum(magnitude, 1e-12))

    before = np.linalg.norm(target_camera, axis=1)
    before_depth = np.abs(target_camera[:, 2])
    accepted = None
    best_improvement = -np.inf
    best_depth_improvement = -np.inf
    last_strain: dict | None = None
    for alpha in (1., .85, .7, .55, .4, .3, .2, .1):
        displacement = float(alpha) * raw_displacement
        achieved = displacement[controls] @ axes.T
        after = np.linalg.norm(target_camera - achieved, axis=1)
        after_depth = np.abs(target_camera[:, 2] - achieved[:, 2])
        improvement = 1. - float(np.median(after) / max(np.median(before), 1e-12))
        depth_improvement = 1. - float(np.median(after_depth) / max(np.median(before_depth), 1e-12))
        best_improvement, best_depth_improvement = max(best_improvement, improvement), max(best_depth_improvement, depth_improvement)
        strain = _edge_strain(prior, prior + displacement, graph)
        last_strain = strain
        if (improvement >= float(minimum_improvement)
                and depth_improvement >= float(minimum_depth_improvement)
                and strain["edge_stretch_p01"] >= .55
                and strain["edge_stretch_p999"] <= 1.9):
            accepted = alpha, displacement, after, after_depth, improvement, depth_improvement, strain
            break
    common = {
        "method": "camera_conditioned_gaussian_mean_adaptation", "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False, **base, "gaussian_slots_preserved": True,
        "selected_controls": int(len(controls)), "locked_visible_gaussians": int(locked.sum()),
        "inherited_locked_gaussians": int(inherited_locked.sum()),
        "editable_gaussians": int(editable.sum()), "influence_gaussians": int(influence.sum()),
        "protected_gaussians": int(protected.sum()), "surface_graph_edges": int(edge_count),
    }
    if accepted is None:
        return prior.copy(), {"active": False, **common, "reason": "insufficient_or_noncoherent_camera_conditioned_update",
                              "best_projection_improvement": float(best_improvement),
                              "best_depth_improvement": float(best_depth_improvement), **(last_strain or {})}, empty_masks
    alpha, displacement, after, after_depth, improvement, depth_improvement, strain = accepted
    moved = np.linalg.norm(displacement, axis=1) > 1e-8
    masks = {"editable": editable, "locked": locked, "protected": protected, "moved": moved}
    return prior + displacement, {
        "active": True, **common, "field": "camera1_projection_depth_screened_gaussian_mean_field",
        "camera_frame": "saved_camera_1", "topology_backtracking_alpha": float(alpha),
        "projection_residual_median_before": float(np.median(before)),
        "projection_residual_median_after": float(np.median(after)), "projection_residual_improvement": float(improvement),
        "depth_residual_median_before": float(np.median(before_depth)),
        "depth_residual_median_after": float(np.median(after_depth)), "depth_residual_improvement": float(depth_improvement),
        "moved_gaussians": int(moved.sum()), "moved_fraction": float(moved.mean()),
        "mean_displacement": float(np.linalg.norm(displacement, axis=1).mean()),
        "maximum_displacement": float(np.linalg.norm(displacement, axis=1).max(initial=0.)),
        "cg_status": cg_status, **strain,
    }, masks
