"""Partial-supported coherent residual registration for complete priors.

The ordinary mainline estimates one proper global Sim(3).  A single partial
view can nevertheless expose a structurally coherent residual at an appendage
while the rest of the registered body already agrees.  This module treats that
case as a *registration* proposal, rather than as a dense Gaussian anchor
edit: it fits one rigid translation from mutually visible residual components
and transfers it to the corresponding complete-prior surface neighbourhood.

The proposal is intentionally conservative and self-contained.  It activates
only for large screen-connected components whose 3-D residuals are highly
coherent and predominantly lie on one saved-camera axis.  Existing low-
residual visible support is held fixed, so a rear discrepancy cannot pull an
already aligned front structure. No complete shape, category, or offline
metric participates in the decision.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph, linalg as sparse_linalg
from scipy.spatial import cKDTree

from src.zbuffer import zbuffer_depth_with_indices


def _visible_residual_records(
    partial: np.ndarray,
    prior: np.ndarray,
    projector,
    *,
    max_pixel_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Camera-1 visible partial/prior records with 2-D coordinates."""
    partial_uv, partial_depth = projector.project(partial)
    prior_uv, prior_depth = projector.project(prior)
    _, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=0,
    )
    _, prior_mask, prior_index = zbuffer_depth_with_indices(
        prior_uv, prior_depth, projector.image_shape, splat_radius=1,
    )
    partial_y, partial_x = np.where(partial_mask)
    prior_y, prior_x = np.where(prior_mask)
    if min(len(partial_x), len(prior_x)) < 6:
        return (
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64), np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    pixel_distance, nearest = cKDTree(np.c_[prior_x, prior_y]).query(
        np.c_[partial_x, partial_y], k=1,
    )
    keep = pixel_distance <= float(max_pixel_distance)
    partial_ids = partial_index[partial_y[keep], partial_x[keep]].astype(np.int64)
    prior_ids = prior_index[prior_y[nearest[keep]], prior_x[nearest[keep]]].astype(np.int64)
    valid = (partial_ids >= 0) & (prior_ids >= 0)
    partial_ids, prior_ids = partial_ids[valid], prior_ids[valid]
    xy = np.c_[partial_x[keep][valid], partial_y[keep][valid]].astype(np.float64)
    residual = partial[partial_ids] - prior[prior_ids]
    return partial_ids, prior_ids, xy, residual, pixel_distance[keep][valid]


def _screen_components(xy: np.ndarray, *, radius: float) -> tuple[int, np.ndarray]:
    """Connected components over a sparse set of residual image pixels."""
    if len(xy) == 0:
        return 0, np.empty(0, dtype=np.int64)
    pairs = cKDTree(xy).query_pairs(r=float(radius), output_type="ndarray")
    if len(pairs) == 0:
        return len(xy), np.arange(len(xy), dtype=np.int64)
    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * len(pairs), dtype=np.float64),
            (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]),
        ),
        shape=(len(xy), len(xy)),
    ).tocsr()
    return csgraph.connected_components(adjacency, directed=False)


def _prior_surface_graph(prior: np.ndarray, *, neighbours: int,
                         edge_ratio: float) -> sparse.csr_matrix:
    """Build the local-density-pruned surface graph used for coherent transfer."""
    query_k = min(int(neighbours) + 1, len(prior))
    if query_k < 2:
        return sparse.csr_matrix((len(prior), len(prior)), dtype=np.float64)
    distances, indices = cKDTree(prior).query(prior, k=query_k, workers=-1)
    local_scale = np.median(distances[:, 1:], axis=1)
    floor = max(float(np.quantile(local_scale, .01)) * .25, 1e-9)
    local_scale = np.maximum(local_scale, floor)
    source = np.repeat(np.arange(len(prior), dtype=np.int64), query_k - 1)
    target = indices[:, 1:].reshape(-1).astype(np.int64)
    edge_distance = distances[:, 1:].reshape(-1)
    keep = source < target
    source, target, edge_distance = source[keep], target[keep], edge_distance[keep]
    scale = np.sqrt(local_scale[source] * local_scale[target])
    keep = edge_distance <= float(edge_ratio) * scale
    source, target, edge_distance = source[keep], target[keep], edge_distance[keep]
    return sparse.coo_matrix(
        (
            np.r_[edge_distance, edge_distance],
            (np.r_[source, target], np.r_[target, source]),
        ),
        shape=(len(prior), len(prior)),
    ).tocsr()


def _dirichlet_component_field(
    graph: sparse.csr_matrix,
    control_ids: np.ndarray,
    control_displacements: np.ndarray,
    *,
    screening: float,
) -> tuple[np.ndarray, list[int]]:
    """Interpolate structural component motion while fixing supported surfaces."""
    count = graph.shape[0]
    controls = np.asarray(control_ids, dtype=np.int64)
    values = np.asarray(control_displacements, dtype=np.float64)
    fixed = np.zeros(count, dtype=bool)
    fixed[controls] = True
    unknown = np.flatnonzero(~fixed)
    displacement = np.zeros((count, 3), dtype=np.float64)
    displacement[controls] = values
    if len(unknown) == 0:
        return displacement, [0, 0, 0]
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - graph
    system = laplacian[unknown][:, unknown].tocsr() + sparse.eye(
        len(unknown), format="csr", dtype=np.float64,
    ) * float(screening)
    rhs = -(laplacian[unknown][:, controls] @ values)
    status: list[int] = []
    for axis in range(3):
        try:
            solution, code = sparse_linalg.cg(
                system, np.asarray(rhs)[:, axis], rtol=1e-5, atol=0., maxiter=320,
            )
        except TypeError:  # SciPy < 1.12
            solution, code = sparse_linalg.cg(
                system, np.asarray(rhs)[:, axis], tol=1e-5, maxiter=320,
            )
        displacement[unknown, axis] = solution
        status.append(int(code))
    return displacement, status


def _geodesic_distance(graph: sparse.csr_matrix, seeds: np.ndarray) -> np.ndarray:
    """Compute deterministic multi-source surface distance from residual seeds."""
    count = graph.shape[0]
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    valid = seeds[(seeds >= 0) & (seeds < count)]
    if len(valid) == 0:
        return np.full(count, np.inf, dtype=np.float64)
    graph_coo = graph.tocoo()
    virtual = count
    augmented = sparse.coo_matrix(
        (
            np.r_[graph_coo.data, np.full(2 * len(valid), 1e-12)],
            (
                np.r_[graph_coo.row, np.full(len(valid), virtual), valid],
                np.r_[graph_coo.col, valid, np.full(len(valid), virtual)],
            ),
        ),
        shape=(count + 1, count + 1), dtype=np.float64,
    ).tocsr()
    return np.asarray(csgraph.dijkstra(augmented, indices=virtual)[:count], dtype=np.float64)


def coherent_residual_component_translation(
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
    maximum_translation_ratio: float = .30,
    component_direction_cosine: float = .95,
    graph_neighbours: int = 8,
    graph_edge_ratio: float = 1.8,
    stable_residual_ratio: float = .05,
    minimum_stable_pairs: int = 512,
    graph_screening: float = .0015,
    component_geodesic_radius_ratio: float = .040,
    maximum_component_fraction: float = .20,
    influence_geodesic_radius_ratio: float = .10,
    maximum_influence_fraction: float = .22,
) -> tuple[np.ndarray, dict]:
    """Propose one coherent component translation from visible residual evidence.

    The residual component's local surface neighbourhood receives one common
    translation. All partial-supported prior nodes with a small residual
    become zero-displacement Dirichlet controls. Optionally, an already
    bridged soft observation (for example MoGe) protects only nodes outside
    the hard partial residual component; a hard scan disagreement always
    takes precedence. The harmonic continuation is itself geodesically
    bounded: every point outside its local influence band is another
    zero-displacement boundary condition. This prevents a small residual from
    leaking through an unsupported graph component and pulling a distant,
    already plausible part of the prior.
    """
    prior, partial = (np.asarray(value, dtype=np.float64) for value in (prior, partial))
    zero = np.zeros_like(prior)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if min(max_pixel_distance, minimum_residual_ratio, screen_component_radius,
           minimum_directional_coherence, minimum_camera_axis_dominance,
           maximum_translation_ratio, component_direction_cosine,
           graph_edge_ratio, stable_residual_ratio, graph_screening,
           component_geodesic_radius_ratio, maximum_component_fraction,
           influence_geodesic_radius_ratio, maximum_influence_fraction) <= 0.:
        raise ValueError("all residual-component bounds must be positive")
    if max(minimum_directional_coherence, minimum_camera_axis_dominance,
           component_direction_cosine, stable_residual_ratio, maximum_component_fraction,
           maximum_influence_fraction) > 1.:
        raise ValueError("coherence, dominance, cosine, stable residual ratio, and component fractions must not exceed one")
    if float(influence_geodesic_radius_ratio) < float(component_geodesic_radius_ratio):
        raise ValueError("influence geodesic radius must include the residual-component core")
    if int(minimum_component_pairs) < 6 or int(minimum_stable_pairs) < 6 or int(graph_neighbours) < 2:
        raise ValueError("minimum component support and graph neighbours are too small")
    axes = np.asarray(camera_axes, dtype=np.float64)
    if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3), rtol=1e-4, atol=1e-5):
        raise ValueError("camera_axes must be an orthonormal (3, 3) row-axis matrix")
    partial_ids, prior_ids, xy, residual, pixel_distance = _visible_residual_records(
        partial, prior, projector, max_pixel_distance=float(max_pixel_distance),
    )
    if auxiliary_support is not None:
        auxiliary_support = np.asarray(auxiliary_support, dtype=np.float64)
        if auxiliary_support.ndim != 2 or auxiliary_support.shape[1] != 3 or len(auxiliary_support) < 6:
            raise ValueError("auxiliary_support must be a nontrivial (N, 3) point cloud")
        if auxiliary_support_max_distance is None or float(auxiliary_support_max_distance) <= 0.:
            raise ValueError("auxiliary support requires a positive max distance")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    residual_norm = np.linalg.norm(residual, axis=1)
    high = residual_norm >= float(minimum_residual_ratio) * diagonal
    if int(high.sum()) < int(minimum_component_pairs):
        return zero, {
            "active": False, "reason": "insufficient_large_visible_residuals",
            "visible_pairs": int(len(residual)), "large_residual_pairs": int(high.sum()),
        }
    high_prior, high_xy, high_residual = prior_ids[high], xy[high], residual[high]
    count, labels = _screen_components(high_xy, radius=float(screen_component_radius))
    candidates: list[dict] = []
    for label in range(count):
        members = labels == label
        if int(members.sum()) < int(minimum_component_pairs):
            continue
        component_residual = high_residual[members]
        translation = np.median(component_residual, axis=0)
        magnitude = float(np.linalg.norm(translation))
        if magnitude <= 1e-12:
            continue
        directional = float(np.mean(
            (component_residual @ translation)
            / np.maximum(np.linalg.norm(component_residual, axis=1) * magnitude, 1e-12)
        ))
        camera_translation = translation @ axes.T
        axis_index = int(np.argmax(np.abs(camera_translation)))
        dominance = float(abs(camera_translation[axis_index]) / magnitude)
        if directional < float(minimum_directional_coherence) or dominance < float(minimum_camera_axis_dominance):
            continue
        candidates.append({
            "label": int(label), "members": members, "translation": translation,
            "magnitude": magnitude, "directional_coherence": directional,
            "camera_translation": camera_translation, "camera_axis": axis_index,
            "camera_axis_dominance": dominance,
        })
    if not candidates:
        return zero, {
            "active": False, "reason": "no_coherent_camera_axis_residual_component",
            "visible_pairs": int(len(residual)), "large_residual_pairs": int(high.sum()),
            "screen_components": int(count),
        }
    candidates.sort(key=lambda item: (-int(item["members"].sum()) * float(item["magnitude"]), int(item["label"])))
    reference = candidates[0]
    reference_direction = reference["translation"] / float(reference["magnitude"])
    selected = [candidate for candidate in candidates if (
        int(candidate["camera_axis"]) == int(reference["camera_axis"])
        and float(np.dot(candidate["translation"] / float(candidate["magnitude"]), reference_direction))
        >= float(component_direction_cosine)
    )]
    member_mask = np.zeros(len(high_prior), dtype=bool)
    for candidate in selected:
        member_mask |= np.asarray(candidate["members"], dtype=bool)
    selected_residual = high_residual[member_mask]
    translation = np.median(selected_residual, axis=0)
    translation = translation * min(
        1., float(maximum_translation_ratio) * diagonal / max(float(np.linalg.norm(translation)), 1e-12),
    )
    stable = residual_norm <= float(stable_residual_ratio) * diagonal
    partial_stable_ids = np.unique(prior_ids[stable])
    seed_ids = np.setdiff1d(np.unique(high_prior[member_mask]), partial_stable_ids, assume_unique=False)
    if (len(seed_ids) < max(6, int(minimum_component_pairs) // 2)
            or len(partial_stable_ids) < int(minimum_stable_pairs)):
        return zero, {
            "active": False, "reason": "insufficient_disjoint_fixed_visible_support",
            "residual_component_prior_points": int(np.unique(high_prior[member_mask]).size),
            "moving_seed_prior_points": int(len(seed_ids)),
            "stable_prior_points": int(len(partial_stable_ids)),
        }
    graph = _prior_surface_graph(prior, neighbours=int(graph_neighbours), edge_ratio=float(graph_edge_ratio))
    if graph.nnz == 0:
        return zero, {"active": False, "reason": "empty_prior_surface_graph"}
    geodesic = _geodesic_distance(graph, seed_ids)
    component_ids = np.flatnonzero(
        geodesic <= float(component_geodesic_radius_ratio) * diagonal,
    )
    moving_ids = np.setdiff1d(component_ids, partial_stable_ids, assume_unique=False)
    if (len(moving_ids) < max(6, int(minimum_component_pairs) // 2)
            or float(len(component_ids) / len(prior)) > float(maximum_component_fraction)):
        return zero, {
            "active": False, "reason": "unsupported_or_overbroad_residual_surface_component",
            "seed_prior_points": int(len(seed_ids)), "component_prior_points": int(len(component_ids)),
            "moving_component_prior_points": int(len(moving_ids)),
        }
    influence_ids = np.flatnonzero(
        geodesic <= float(influence_geodesic_radius_ratio) * diagonal,
    )
    if float(len(influence_ids) / len(prior)) > float(maximum_influence_fraction):
        return zero, {
            "active": False, "reason": "overbroad_structural_continuation_band",
            "seed_prior_points": int(len(seed_ids)), "component_prior_points": int(len(component_ids)),
            "influence_prior_points": int(len(influence_ids)),
        }
    auxiliary_stable_ids = np.empty(0, dtype=np.int64)
    if auxiliary_support is not None:
        auxiliary_distance, _ = cKDTree(auxiliary_support).query(prior, k=1, workers=-1)
        # A soft observation does not override a coherent hard-scan residual.
        auxiliary_stable_ids = np.setdiff1d(
            np.flatnonzero(auxiliary_distance <= float(auxiliary_support_max_distance)),
            component_ids, assume_unique=False,
        )
    # The complement of the local band is an explicit zero-motion boundary:
    # unsupported graph paths must never drag a distant, already aligned limb
    # or body part. Hard partial support and nonconflicting soft MoGe support
    # remain fixed inside the band as well.
    outer_fixed_ids = np.setdiff1d(
        np.arange(len(prior), dtype=np.int64), influence_ids, assume_unique=True,
    )
    stable_ids = np.unique(np.r_[partial_stable_ids, auxiliary_stable_ids, outer_fixed_ids])
    control_ids = np.r_[stable_ids, moving_ids]
    control_displacements = np.r_[
        np.zeros((len(stable_ids), 3), dtype=np.float64),
        np.broadcast_to(translation, (len(moving_ids), 3)).copy(),
    ]
    displacement, cg_status = _dirichlet_component_field(
        graph, control_ids, control_displacements, screening=float(graph_screening),
    )
    magnitude = np.linalg.norm(displacement, axis=1)
    moved = magnitude > 1e-8
    component_records = [{
        "screen_component": int(candidate["label"]), "pairs": int(candidate["members"].sum()),
        "median_translation": np.asarray(candidate["translation"]).tolist(),
        "median_translation_camera": np.asarray(candidate["camera_translation"]).tolist(),
        "magnitude": float(candidate["magnitude"]),
        "directional_coherence": float(candidate["directional_coherence"]),
        "camera_axis": int(candidate["camera_axis"]),
        "camera_axis_dominance": float(candidate["camera_axis_dominance"]),
    } for candidate in selected]
    return displacement, {
        "active": True,
        "method": "camera1_coherent_residual_component_translation",
        "visible_pairs": int(len(residual)), "large_residual_pairs": int(high.sum()),
        "screen_components": int(count), "selected_components": component_records,
        "selected_component_pairs": int(member_mask.sum()),
        "seed_prior_points": int(len(seed_ids)), "component_prior_points": int(len(component_ids)),
        "influence_prior_points": int(len(influence_ids)),
        "moving_component_prior_points": int(len(moving_ids)), "stable_prior_points": int(len(stable_ids)),
        "partial_stable_prior_points": int(len(partial_stable_ids)),
        "auxiliary_stable_prior_points": int(len(auxiliary_stable_ids)),
        "outer_fixed_prior_points": int(len(outer_fixed_ids)),
        "stable_visible_pairs": int(stable.sum()),
        "moved_prior_points": int(moved.sum()), "moved_fraction": float(moved.mean()),
        "mean_moved_displacement": float(magnitude[moved].mean()) if np.any(moved) else 0.,
        "maximum_displacement": float(magnitude.max(initial=0.)), "cg_status": cg_status,
        "translation": translation.tolist(), "translation_camera": (translation @ axes.T).tolist(),
        "parameters": {
            "max_pixel_distance": float(max_pixel_distance),
            "minimum_residual_ratio": float(minimum_residual_ratio),
            "screen_component_radius": float(screen_component_radius),
            "minimum_component_pairs": int(minimum_component_pairs),
            "minimum_directional_coherence": float(minimum_directional_coherence),
            "minimum_camera_axis_dominance": float(minimum_camera_axis_dominance),
            "maximum_translation_ratio": float(maximum_translation_ratio),
            "component_direction_cosine": float(component_direction_cosine),
            "graph_neighbours": int(graph_neighbours), "graph_edge_ratio": float(graph_edge_ratio),
            "stable_residual_ratio": float(stable_residual_ratio),
            "minimum_stable_pairs": int(minimum_stable_pairs),
            "graph_screening": float(graph_screening),
            "component_geodesic_radius_ratio": float(component_geodesic_radius_ratio),
            "maximum_component_fraction": float(maximum_component_fraction),
            "influence_geodesic_radius_ratio": float(influence_geodesic_radius_ratio),
            "maximum_influence_fraction": float(maximum_influence_fraction),
            "auxiliary_support_max_distance": (
                float(auxiliary_support_max_distance) if auxiliary_support is not None else None
            ),
        },
        "carrier_slots_preserved": True,
        "ground_truth_cd_emd_used": False,
        "field": "coherent_registration_component_with_fixed_supported_surface",
    }
