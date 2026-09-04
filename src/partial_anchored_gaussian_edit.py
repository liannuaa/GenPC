"""Single-view boundary-conditioned edits for a partial-anchored Gaussian field.

The edit is deliberately a geometry-only 3DGS mean update: observed partial
points are immutable evidence, while the complete Pixal population remains the
only complete-body carrier.  In particular, this module never creates a raw
point-cloud union and never inspects GT metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

from src.partial_anchored_gaussian_decode import collision_free_anchor_pairs


def _clip_vectors(vectors: np.ndarray, maximum: float) -> np.ndarray:
    """Bound vector length without changing its direction."""
    magnitude = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * np.minimum(1., float(maximum) / np.maximum(magnitude, 1e-12))


def _solve_cg(matrix: sparse.csr_matrix, rhs: np.ndarray, *, tolerance: float,
              max_iterations: int) -> tuple[np.ndarray, list[int]]:
    """Solve the three displacement coordinates with a compatible SciPy CG API."""
    solution = np.zeros((matrix.shape[0], rhs.shape[1]), dtype=np.float64)
    status: list[int] = []
    for axis in range(rhs.shape[1]):
        try:
            vector, code = sparse_linalg.cg(
                matrix, rhs[:, axis], rtol=float(tolerance), atol=0.,
                maxiter=int(max_iterations),
            )
        except TypeError:  # SciPy < 1.12
            vector, code = sparse_linalg.cg(
                matrix, rhs[:, axis], tol=float(tolerance), maxiter=int(max_iterations),
            )
        solution[:, axis] = vector
        status.append(int(code))
    return solution, status


def _orthographic_prior_protection(
    prior: np.ndarray,
    *,
    views: int,
    bins: int = 192,
) -> np.ndarray:
    """Return prior means exposed in fixed self-rendered protection views.

    These are not invented partial observations.  They are deterministic
    orthographic z-buffers of the initial complete prior along signed PCA
    directions, used only to regularize unsupported deformation.  A point is
    retained if it wins at least one front-most pixel in a protection view.
    """
    if views <= 0:
        return np.zeros(len(prior), dtype=bool)
    centre = prior.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(prior - centre, full_matrices=False)
    coordinates = (prior - centre) @ right.T
    selected = np.zeros(len(prior), dtype=bool)
    directions = [(axis, sign) for axis in range(3) for sign in (-1, 1)][:int(views)]
    for depth_axis, sign in directions:
        planar_axes = [axis for axis in range(3) if axis != depth_axis]
        plane = coordinates[:, planar_axes]
        lower, upper = plane.min(axis=0), plane.max(axis=0)
        extent = np.maximum(upper - lower, 1e-9)
        uv = np.floor((plane - lower) / extent * (int(bins) - 1)).astype(np.int64)
        key = uv[:, 0] * int(bins) + uv[:, 1]
        # Sign chooses either the front or rear side in this deterministic
        # self-render; stable sorting keeps ties reproducible.
        depth = sign * coordinates[:, depth_axis]
        order = np.lexsort((np.arange(len(prior)), depth, key))
        ordered_key = key[order]
        keep = np.concatenate(([True], ordered_key[1:] != ordered_key[:-1]))
        selected[order[keep]] = True
    return selected


def boundary_conditioned_graph_displacement(
    prior: np.ndarray,
    partial: np.ndarray,
    pixel_pairs: np.ndarray,
    *,
    max_anchor_residual: float,
    max_displacement: float,
    neighbours: int = 8,
    edge_ratio: float = 1.8,
    screening: float = .003,
    prior_protection_views: int = 6,
    prior_protection_weight: float = .02,
    protection_exclusion_ratio: float = .08,
    remote_gain: float = 1.,
    remote_gain_radius_ratio: float = .08,
    remote_displacement_cap_multiplier: float = 1.,
    cg_tolerance: float = 1e-5,
    cg_max_iterations: int = 240,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Propagate hard saved-view evidence across the prior's surface graph.

    Pixel-indexed partial/Pixal matches are *Dirichlet controls*: their prior
    Gaussian centres are fixed to the measured partial offsets.  All other
    Pixal Gaussians receive a displacement only through a local 3-D kNN graph.
    The graph is pruned by local sampling scale, which prevents propagation
    across large gaps (e.g. chair legs or the two sides of an armrest).  A
    small zero-displacement screen keeps components with no observation still
    and limits extrapolation into genuinely unobserved structure.

    This is the single-view counterpart of boundary-value Gaussian editing:
    observed image neighbourhoods are fixed, while remote geometry follows
    only if it belongs to the same local surface structure.
    """
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if len(prior) < 3:
        raise ValueError("at least three prior points are required")
    if min(max_anchor_residual, max_displacement, edge_ratio, screening, cg_tolerance,
           protection_exclusion_ratio, remote_gain, remote_gain_radius_ratio,
           remote_displacement_cap_multiplier) <= 0. or prior_protection_weight < 0.:
        raise ValueError("metric bounds, graph bounds, screening, and tolerance must be positive")
    if neighbours < 2 or cg_max_iterations <= 0:
        raise ValueError("neighbours must be at least two and cg_max_iterations positive")

    anchors, anchor_info = collision_free_anchor_pairs(
        pixel_pairs, partial, prior, max_residual=float(max_anchor_residual),
    )
    if len(anchors) < 6:
        return prior.copy(), anchors, {
            **anchor_info, "active": False,
            "reason": "insufficient_collision_free_anchors",
            "field": "boundary_conditioned_surface_graph",
        }

    partial_ids, control_ids = anchors[:, 0], anchors[:, 1]
    control_offsets = _clip_vectors(
        partial[partial_ids] - prior[control_ids], float(max_displacement),
    )
    is_control = np.zeros(len(prior), dtype=bool)
    is_control[control_ids] = True

    # The graph is built in the registered complete-prior frame.  Its local
    # radius follows sampling density, rather than a global Euclidean ball,
    # so neighbouring but distinct surfaces do not get unintentionally tied.
    query_k = min(int(neighbours) + 1, len(prior))
    distances, indices = cKDTree(prior).query(prior, k=query_k, workers=-1)
    if query_k == 1:  # covered above, kept for static safety
        return prior.copy(), anchors, {**anchor_info, "active": False, "reason": "too_few_prior_points"}
    local_scale = np.median(distances[:, 1:], axis=1)
    local_scale = np.maximum(local_scale, max(float(np.quantile(local_scale, .01)) * .25, 1e-9))
    source = np.repeat(np.arange(len(prior), dtype=np.int64), query_k - 1)
    target = indices[:, 1:].reshape(-1).astype(np.int64)
    edge_distance = distances[:, 1:].reshape(-1)
    # Retaining an undirected edge once avoids directional density bias.
    keep = source < target
    source, target, edge_distance = source[keep], target[keep], edge_distance[keep]
    scale = np.sqrt(local_scale[source] * local_scale[target])
    keep = edge_distance <= float(edge_ratio) * scale
    source, target, edge_distance, scale = (
        source[keep], target[keep], edge_distance[keep], scale[keep]
    )
    if len(source) == 0:
        return prior.copy(), anchors, {**anchor_info, "active": False, "reason": "empty_surface_graph"}
    weights = np.exp(-np.square(edge_distance / np.maximum(scale, 1e-12)))
    adjacency = sparse.coo_matrix(
        (np.concatenate((weights, weights)),
         (np.concatenate((source, target)), np.concatenate((target, source)))),
        shape=(len(prior), len(prior)), dtype=np.float64,
    ).tocsr()
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - adjacency
    unknown_ids = np.flatnonzero(~is_control)
    displacement = np.zeros_like(prior)
    displacement[control_ids] = control_offsets
    # The only target-view evidence is the saved camera.  Fixed PCA self-views
    # therefore contribute *zero-displacement preservation* rather than fake
    # negative evidence from an incomplete rotated partial scan.  Excluding a
    # metric neighbourhood of actual controls permits coherent local follow.
    control_tree = cKDTree(prior[control_ids])
    nearest_control, _ = control_tree.query(prior, k=1, workers=-1)
    protection_radius = float(protection_exclusion_ratio) * max(
        float(np.linalg.norm(np.ptp(prior, axis=0))), 1e-9,
    )
    protection_visible = _orthographic_prior_protection(
        prior, views=int(prior_protection_views),
    )
    protection_mask = protection_visible & ~is_control & (nearest_control > protection_radius)
    if len(unknown_ids):
        # Dirichlet harmonic extension with a weak screened term:
        #  min sum_ij w_ij ||d_i-d_j||^2 + screening * sum_i ||d_i||^2,
        #  subject to d_control = measured partial offset.
        unknown_laplacian = laplacian[unknown_ids][:, unknown_ids].tocsr()
        diagonal = float(screening) + float(prior_protection_weight) * protection_mask[unknown_ids]
        system = unknown_laplacian + sparse.diags(diagonal, format="csr")
        coupling = laplacian[unknown_ids][:, control_ids]
        rhs = -(coupling @ control_offsets)
        solution, cg_status = _solve_cg(
            system, np.asarray(rhs), tolerance=float(cg_tolerance),
            max_iterations=int(cg_max_iterations),
        )
        # Controls remain exact.  Only non-controls may receive a continuous
        # distance-monotone gain, allowing remote structure to follow a local
        # edit more strongly without loosening any observed correspondence.
        gain_radius = float(remote_gain_radius_ratio) * max(
            float(np.linalg.norm(np.ptp(prior, axis=0))), 1e-9,
        )
        gain = 1. + (float(remote_gain) - 1.) * (1. - np.exp(-nearest_control[unknown_ids] / gain_radius))
        remote_solution = solution * gain[:, None]
        displacement[unknown_ids] = _clip_vectors(
            remote_solution,
            float(max_displacement) * float(remote_displacement_cap_multiplier),
        )
    else:
        cg_status = [0, 0, 0]

    components, labels = sparse.csgraph.connected_components(adjacency, directed=False)
    component_has_control = np.zeros(components, dtype=bool)
    component_has_control[labels[control_ids]] = True
    active_nodes = component_has_control[labels]
    magnitude = np.linalg.norm(displacement, axis=1)
    edited = prior + displacement
    return edited, anchors, {
        **anchor_info,
        "active": True,
        "controls": int(len(control_ids)),
        "max_displacement": float(max_displacement),
        "neighbours": int(query_k - 1),
        "edge_ratio": float(edge_ratio),
        "screening": float(screening),
        "prior_protection_views": int(prior_protection_views),
        "prior_protection_weight": float(prior_protection_weight),
        "protection_exclusion_radius": float(protection_radius),
        "prior_protection_gaussians": int(protection_mask.sum()),
        "remote_gain": float(remote_gain),
        "remote_gain_radius": float(gain_radius) if len(unknown_ids) else 0.,
        "remote_displacement_cap": float(max_displacement) * float(remote_displacement_cap_multiplier),
        "graph_edges": int(len(source)),
        "graph_components": int(components),
        "control_components": int(component_has_control.sum()),
        "graph_reachable_prior_gaussians": int(active_nodes.sum()),
        "edited_prior_gaussians": int((magnitude > 1e-12).sum()),
        "untouched_prior_gaussians": int((magnitude <= 1e-12).sum()),
        "mean_displacement": float(magnitude.mean()),
        "p95_displacement": float(np.quantile(magnitude, .95)),
        "max_observed_displacement": float(magnitude.max(initial=0.)),
        "mean_noncontrol_displacement": float(magnitude[~is_control].mean()) if np.any(~is_control) else 0.,
        "cg_status": cg_status,
        "field": "boundary_conditioned_surface_graph_with_multiview_prior_protection",
    }


def compact_anchor_displacement(
    prior: np.ndarray,
    partial: np.ndarray,
    pixel_pairs: np.ndarray,
    *,
    max_anchor_residual: float,
    support_radius: float,
    max_displacement: float,
    neighbours: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Condition editable Gaussian centres on hard anchors with compact support.

    The input pairs are mutually visible saved-view matches.  Their observed
    point-to-prior offsets are handles of a compact Wendland field over the
    *registered prior* coordinates.  Hence only the observed neighbourhood is
    edited, no raw partial union is formed, and all remote complete-prior
    Gaussians receive exactly zero displacement.
    """
    prior, partial = np.asarray(prior, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if min(max_anchor_residual, support_radius, max_displacement) <= 0. or neighbours <= 0:
        raise ValueError("all metric bounds and neighbours must be positive")
    anchors, anchor_info = collision_free_anchor_pairs(
        pixel_pairs, partial, prior, max_residual=float(max_anchor_residual),
    )
    if len(anchors) < 6:
        return prior.copy(), anchors, {**anchor_info, "active": False,
                                       "reason": "insufficient_collision_free_anchors"}
    partial_ids, prior_ids = anchors[:, 0], anchors[:, 1]
    controls, offsets = prior[prior_ids], partial[partial_ids] - prior[prior_ids]
    tree = cKDTree(controls)
    distance, control_ids = tree.query(prior, k=min(int(neighbours), len(controls)), workers=-1)
    if distance.ndim == 1:
        distance, control_ids = distance[:, None], control_ids[:, None]
    normalized = distance / float(support_radius)
    # C2 Wendland kernel: smooth at the support boundary and exactly zero
    # outside, which protects unobserved complete-prior geometry.
    inside = normalized < 1.
    weights = np.where(inside, (1. - normalized) ** 4 * (4. * normalized + 1.), 0.)
    weight_sum = weights.sum(axis=1, keepdims=True)
    displacement = (weights[..., None] * offsets[control_ids]).sum(axis=1)
    displacement /= np.maximum(weight_sum, 1e-12)
    magnitude = np.linalg.norm(displacement, axis=1, keepdims=True)
    displacement *= np.minimum(1., float(max_displacement) / np.maximum(magnitude, 1e-12))
    edited = prior + displacement
    return edited, anchors, {
        **anchor_info, "active": True, "controls": int(len(controls)),
        "support_radius": float(support_radius), "max_displacement": float(max_displacement),
        "neighbours": int(min(int(neighbours), len(controls))),
        "edited_prior_gaussians": int((weight_sum[:, 0] > 0.).sum()),
        "untouched_prior_gaussians": int((weight_sum[:, 0] == 0.).sum()),
        "mean_displacement": float(np.linalg.norm(displacement, axis=1).mean()),
        "p95_displacement": float(np.quantile(np.linalg.norm(displacement, axis=1), .95)),
        "max_observed_displacement": float(np.linalg.norm(displacement, axis=1).max(initial=0.)),
        "field": "compact_wendland_source_space_gaussian_mean_displacement",
    }
