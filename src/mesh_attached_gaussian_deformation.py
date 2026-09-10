"""Topology-preserving local deformation of a registered textured prior mesh.

The object pipeline normally carries a uniformly sampled 100k prior.  That
carrier is convenient for registration but has no surface connectivity: a
point-graph displacement may separate an appendage from its body.  This module
recovers the exact global proper Sim(3) from the slot-preserving carrier,
transfers it to the original Pixal mesh, and performs a support-anchored axial
deformation on the *mesh* graph.  Gaussians/points sampled from the mesh remain
attached to its triangles, so the same operation is directly compatible with a
mesh-attached 3DGS representation.

The proposal is category-free.  It requires a coherent visible residual,
fixed observed support, a bounded geodesic influence region, an inferred mesh
attachment, and a visible-data improvement before it may modify the mesh.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree
from scipy import ndimage

from src.attachment_aware_gaussian import (
    _coherent_residual_selection,
    _solve_scalar_dirichlet,
)
from src.robust_similarity import validate_proper_sim3, weighted_umeyama
from src.zbuffer import zbuffer_depth_with_indices


def fit_carrier_sim3(source_carrier: np.ndarray, registered_carrier: np.ndarray) -> tuple[np.ndarray, dict]:
    """Recover the slot-preserving global Sim(3) applied to a Pixal carrier."""
    source = np.asarray(source_carrier, dtype=np.float64)
    target = np.asarray(registered_carrier, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 6:
        raise ValueError("source and registered carriers must be aligned (N, 3) arrays")
    transform = weighted_umeyama(source, target, np.ones(len(source), dtype=np.float64))
    scale = validate_proper_sim3(transform)
    residual = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
    return transform, {
        "scale": float(scale), "carrier_sim3_mean_residual": float(residual.mean()),
        "carrier_sim3_p99_residual": float(np.quantile(residual, .99)),
        "carrier_sim3_max_residual": float(residual.max()),
    }


def fit_carrier_affine(source_carrier: np.ndarray, adapted_carrier: np.ndarray) -> tuple[np.ndarray, dict]:
    """Recover a slot-preserving, orientation-preserving carrier adaptation.

    This is used only after the globally registered prior has received a
    partial-supported axis calibration.  It transfers that calibration to the
    original textured mesh without changing faces or UV indexing.
    """
    source = np.asarray(source_carrier, dtype=np.float64)
    target = np.asarray(adapted_carrier, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 6:
        raise ValueError("source and adapted carriers must be aligned (N, 3) arrays")
    design = np.c_[source, np.ones(len(source), dtype=np.float64)]
    coefficient, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = coefficient[:3].T
    transform[:3, 3] = coefficient[3]
    prediction = source @ transform[:3, :3].T + transform[:3, 3]
    residual = np.linalg.norm(prediction - target, axis=1)
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    determinant = float(np.linalg.det(transform[:3, :3]))
    if determinant <= 0. or not np.isfinite(singular).all() or singular[-1] <= 1e-10:
        raise ValueError("carrier adaptation is not an orientation-preserving affine map")
    return transform, {
        "carrier_affine_mean_residual": float(residual.mean()),
        "carrier_affine_p99_residual": float(np.quantile(residual, .99)),
        "carrier_affine_max_residual": float(residual.max()),
        "carrier_affine_determinant": determinant,
        "carrier_affine_singular_values": singular.tolist(),
    }


def mesh_surface_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    weld_relative_tolerance: float = 1e-7,
) -> sparse.csr_matrix:
    """Build a UV-seam-welded triangle-edge graph with geometric edge lengths.

    Image-to-3D meshes often duplicate a geometric vertex for different UV
    charts.  Keeping those duplicates disconnected makes a visually connected
    surface look like thousands of separate parts to a deformation solver.
    We add only tolerance-quantized coincident-vertex edges to the *solver*
    graph; the textured mesh and its indexed faces remain untouched.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("vertices must be (N, 3) and faces must be triangular")
    if weld_relative_tolerance <= 0.:
        raise ValueError("weld_relative_tolerance must be positive")
    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0)
    diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    tolerance = float(weld_relative_tolerance) * diagonal
    quantized = np.rint((vertices - vertices.min(axis=0)) / tolerance).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    ordered_ids = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_ids)) + 1]
    stops = np.r_[starts[1:], len(order)]
    seam_edges = []
    for start, stop in zip(starts, stops):
        group = order[start:stop]
        if len(group) > 1:
            # A short chain provides exact chart stitching without making the
            # sparse graph dense for a vertex shared by many UV charts.
            seam_edges.append(np.column_stack((group[:-1], group[1:])))
    if seam_edges:
        edges = np.concatenate((edges, *seam_edges), axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    # Dijkstra permits zero-cost links, but an epsilon prevents backend-
    # dependent handling of exact duplicates while preserving the surface
    # distance scale.
    length = np.maximum(length, tolerance * 1e-3)
    return sparse.coo_matrix(
        (np.r_[length, length], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(vertices), len(vertices)), dtype=np.float64,
    ).tocsr()


def _edge_boundary(graph: sparse.csr_matrix, region: np.ndarray) -> np.ndarray:
    """Return vertices immediately outside a mesh region across triangle edges."""
    region = np.asarray(region, dtype=np.int64)
    membership = np.zeros(graph.shape[0], dtype=bool)
    membership[region] = True
    coo = graph.tocoo()
    crossing = membership[coo.row] ^ membership[coo.col]
    return np.unique(np.where(membership[coo.row[crossing]], coo.col[crossing], coo.row[crossing]))


def _surface_distance(graph: sparse.csr_matrix, sources: np.ndarray) -> np.ndarray:
    """Return exact multi-source geodesic distance on a mesh edge graph."""
    sources = np.unique(np.asarray(sources, dtype=np.int64))
    if len(sources) == 0:
        return np.full(graph.shape[0], np.inf, dtype=np.float64)
    coo = graph.tocoo()
    virtual = graph.shape[0]
    augmented = sparse.coo_matrix(
        (
            np.r_[coo.data, np.full(2 * len(sources), 1e-12)],
            (np.r_[coo.row, np.full(len(sources), virtual), sources],
             np.r_[coo.col, sources, np.full(len(sources), virtual)]),
        ), shape=(virtual + 1, virtual + 1), dtype=np.float64,
    ).tocsr()
    return np.asarray(csgraph.dijkstra(augmented, indices=virtual)[:virtual], dtype=np.float64)


def _surface_orientation_quality(
    vertices: np.ndarray,
    deformed: np.ndarray,
    faces: np.ndarray,
) -> dict:
    """Check orientation only on geometrically resolved mesh triangles.

    Generative meshes can contain a tiny number of zero-area or nearly
    zero-area triangles.  Their normals are numerically undefined and should
    not veto a deformation of the resolved surface.  The threshold is derived
    from the mesh itself, never from a category or an image.
    """
    reference_cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                               vertices[faces[:, 2]] - vertices[faces[:, 0]])
    deformed_cross = np.cross(deformed[faces[:, 1]] - deformed[faces[:, 0]],
                              deformed[faces[:, 2]] - deformed[faces[:, 0]])
    reference_area = .5 * np.linalg.norm(reference_cross, axis=1)
    resolved_area = max(1e-12, float(np.median(reference_area)) * 1e-4)
    resolved = reference_area >= resolved_area
    normal_dot = np.sum(reference_cross * deformed_cross, axis=1)
    flipped = resolved & (normal_dot <= 0.)
    original_edges = np.stack((
        np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1),
        np.linalg.norm(vertices[faces[:, 2]] - vertices[faces[:, 1]], axis=1),
        np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=1),
    ), axis=1)
    deformed_edges = np.stack((
        np.linalg.norm(deformed[faces[:, 1]] - deformed[faces[:, 0]], axis=1),
        np.linalg.norm(deformed[faces[:, 2]] - deformed[faces[:, 1]], axis=1),
        np.linalg.norm(deformed[faces[:, 0]] - deformed[faces[:, 2]], axis=1),
    ), axis=1)
    resolved_edges = original_edges[resolved]
    ratios = (deformed_edges[resolved] / np.maximum(resolved_edges, 1e-12)).ravel()
    return {
        "resolved_triangle_area_threshold": resolved_area,
        "resolved_triangle_fraction": float(resolved.mean()),
        "flipped_resolved_triangle_fraction": float(flipped[resolved].mean()) if resolved.any() else 1.,
        "unresolved_triangle_fraction": float((~resolved).mean()),
        "edge_stretch_p01": float(np.quantile(ratios, .01)),
        "edge_stretch_p50": float(np.quantile(ratios, .50)),
        "edge_stretch_p99": float(np.quantile(ratios, .99)),
        "edge_stretch_p999": float(np.quantile(ratios, .999)),
    }


def _choose_axial_fit(
    source: np.ndarray,
    target: np.ndarray,
    pivot: np.ndarray,
    component_vertices: np.ndarray,
    camera_axes: np.ndarray,
    *,
    maximum_log_scale: float,
) -> dict | None:
    """Select a bounded local stretch axis using only matched visible points."""
    centred_source, centred_target = source - pivot, target - pivot
    residual_direction = np.median(centred_target - centred_source, axis=0)
    centre = np.median(component_vertices, axis=0)
    _, _, right = np.linalg.svd(component_vertices - centre, full_matrices=False)
    directions = np.vstack((centre - pivot, residual_direction, right, camera_axes))
    before = np.linalg.norm(centred_target - centred_source, axis=1)
    max_scale = float(np.exp(maximum_log_scale))
    min_scale = 1. / max_scale
    best: dict | None = None
    for direction in directions:
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        axis = direction / length
        coordinate = centred_source @ axis
        denominator = float(coordinate @ coordinate)
        if denominator <= 1e-12:
            continue
        scale = float(np.clip(coordinate @ (centred_target @ axis) / denominator, min_scale, max_scale))
        prediction = centred_source + (scale - 1.) * coordinate[:, None] * axis
        error = np.linalg.norm(centred_target - prediction, axis=1)
        item = {"axis": axis, "scale": scale, "before": before, "after": error}
        if best is None or float(np.median(error)) < float(np.median(best["after"])):
            best = item
    return best


def _partial_control_targets(
    vertices: np.ndarray,
    prior: np.ndarray,
    partial: np.ndarray,
    prior_ids: np.ndarray,
    partial_ids: np.ndarray,
    records: np.ndarray,
    mesh_tree: cKDTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate coherent partial correspondences into mesh-vertex targets."""
    vertex_ids = mesh_tree.query(prior[prior_ids[records]], k=1, workers=-1)[1]
    displacements = partial[partial_ids[records]] - prior[prior_ids[records]]
    unique, inverse = np.unique(vertex_ids, return_inverse=True)
    totals = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(totals, inverse, displacements)
    return unique, vertices[unique] + totals / counts[:, None]


def _arap_surface_deformation(
    vertices: np.ndarray,
    graph: sparse.csr_matrix,
    movable: np.ndarray,
    fixed: np.ndarray,
    control_vertices: np.ndarray,
    control_targets: np.ndarray,
    *,
    data_weight: float = 32.,
    iterations: int = 5,
) -> tuple[np.ndarray, dict]:
    """Solve a local as-rigid-as-possible surface deformation.

    The surface graph is UV-seam welded, while the exported mesh retains its
    original vertex/UV indexing.  Only a residual-supported region is free;
    partial-consistent and outer vertices remain fixed exactly.
    """
    movable = np.unique(np.asarray(movable, dtype=np.int64))
    fixed = np.unique(np.asarray(fixed, dtype=np.int64))
    if len(movable) == 0 or len(control_vertices) == 0:
        raise ValueError("ARAP requires movable vertices and data controls")
    index = np.full(len(vertices), -1, dtype=np.int64)
    index[movable] = np.arange(len(movable), dtype=np.int64)
    coo = graph.tocoo()
    edge_mask = coo.row < coo.col
    edge_i, edge_j = coo.row[edge_mask], coo.col[edge_mask]
    # Uniform graph weights prevent UV seam epsilon lengths from dominating
    # the metric.  The mesh graph encodes topology, not a physical stiffness.
    weights = np.ones(len(edge_i), dtype=np.float64)
    movable_i, movable_j = index[edge_i], index[edge_j]
    touches = (movable_i >= 0) | (movable_j >= 0)
    edge_i, edge_j, weights, movable_i, movable_j = (
        array[touches] for array in (edge_i, edge_j, weights, movable_i, movable_j)
    )
    row_parts, col_parts, data_parts = [], [], []
    diagonal = np.zeros(len(movable), dtype=np.float64)
    both = (movable_i >= 0) & (movable_j >= 0)
    if both.any():
        ui, uj, ww = movable_i[both], movable_j[both], weights[both]
        row_parts.extend((ui, uj))
        col_parts.extend((uj, ui))
        data_parts.extend((-ww, -ww))
    for local, valid in ((movable_i, movable_i >= 0), (movable_j, movable_j >= 0)):
        np.add.at(diagonal, local[valid], weights[valid])
    controls = np.asarray(control_vertices, dtype=np.int64)
    targets = np.asarray(control_targets, dtype=np.float64)
    local_control = index[controls]
    keep = local_control >= 0
    if not keep.any():
        raise ValueError("all residual controls are fixed; no admissible ARAP update")
    local_control, targets = local_control[keep], targets[keep]
    accum_targets = np.zeros((len(movable), 3), dtype=np.float64)
    np.add.at(accum_targets, local_control, targets * float(data_weight))
    np.add.at(diagonal, local_control, float(data_weight))
    row_parts.append(np.arange(len(movable), dtype=np.int64))
    col_parts.append(np.arange(len(movable), dtype=np.int64))
    data_parts.append(diagonal)
    system = sparse.coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(len(movable), len(movable)), dtype=np.float64,
    ).tocsc()
    solve = sparse_linalg.factorized(system)
    rest_edges = vertices[edge_i] - vertices[edge_j]
    deformed = vertices.copy()
    rotations = np.broadcast_to(np.eye(3), (len(vertices), 3, 3)).copy()
    for _ in range(int(iterations)):
        deformed_edges = deformed[edge_i] - deformed[edge_j]
        covariance = np.zeros((len(movable), 3, 3), dtype=np.float64)
        outer = np.einsum("ni,nj->nij", deformed_edges, rest_edges)
        for local, valid in ((movable_i, movable_i >= 0), (movable_j, movable_j >= 0)):
            np.add.at(covariance, local[valid], outer[valid])
        u, _, vh = np.linalg.svd(covariance, full_matrices=False)
        local_rotations = u @ vh
        determinant = np.linalg.det(local_rotations)
        if np.any(determinant < 0.):
            u[determinant < 0., :, -1] *= -1.
            local_rotations = u @ vh
        rotations[movable] = local_rotations
        edge_rotation = .5 * (rotations[edge_i] + rotations[edge_j])
        arap_edge = np.einsum("nij,nj->ni", edge_rotation, rest_edges)
        rhs = accum_targets.copy()
        # Fixed neighbours contribute their original positions to the Laplacian
        # system; ARAP edge terms contribute with opposite signs at its ends.
        for vertex, other, local, sign in (
            (edge_i, edge_j, movable_i, 1.),
            (edge_j, edge_i, movable_j, -1.),
        ):
            valid = local >= 0
            fixed_neighbour = valid & (index[other] < 0)
            if fixed_neighbour.any():
                np.add.at(rhs, local[fixed_neighbour], vertices[other[fixed_neighbour]])
            if valid.any():
                np.add.at(rhs, local[valid], sign * arap_edge[valid])
        for dimension in range(3):
            deformed[movable, dimension] = solve(rhs[:, dimension])
    displacement = deformed - vertices
    residual_before = np.linalg.norm(vertices[controls[keep]] - targets, axis=1)
    residual_after = np.linalg.norm(deformed[controls[keep]] - targets, axis=1)
    return displacement, {
        "arap_iterations": int(iterations), "arap_data_weight": float(data_weight),
        "arap_control_vertices": int(len(controls[keep])),
        "arap_movable_vertices": int(len(movable)),
        "arap_control_residual_median_before": float(np.median(residual_before)),
        "arap_control_residual_median_after": float(np.median(residual_after)),
        "arap_control_improvement": float(1. - np.median(residual_after) / max(np.median(residual_before), 1e-12)),
    }


def _camera_depth_vertex_controls(
    vertices: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    minimum_depth_residual: float,
    stable_depth_residual: float,
    minimum_component_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read partial depth residuals at the mesh's visible Camera-1 pixels.

    This is a 2.5-D observation, not a nearest-neighbour correspondence: the
    same saved camera maps a visible mesh vertex and a partial sample to the
    same raster cell.  Each control is therefore constrained only along the
    camera depth axis, keeping its observed image position intact.
    """
    mesh_uv, mesh_depth = projector.project(vertices)
    partial_uv, partial_depth = projector.project(partial)
    partial_raster, partial_mask, _ = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=1,
    )
    mesh_raster, mesh_mask, mesh_index = zbuffer_depth_with_indices(
        mesh_uv, mesh_depth, projector.image_shape, splat_radius=1,
    )
    overlap = partial_mask & mesh_mask & (mesh_index >= 0)
    residual_image = np.zeros(projector.image_shape, dtype=np.float64)
    residual_image[overlap] = partial_raster[overlap] - mesh_raster[overlap]
    strong = overlap & (np.abs(residual_image) >= float(minimum_depth_residual))
    selected = np.zeros_like(strong)
    components = 0
    for sign in (-1., 1.):
        labels, count = ndimage.label(strong & (sign * residual_image > 0.))
        components += int(count)
        if count:
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            selected |= (labels > 0) & (sizes[labels] >= int(minimum_component_pixels))
    selected_ids = mesh_index[selected]
    selected_delta = residual_image[selected]
    if len(selected_ids) == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64), {
                    "overlap_pixels": int(overlap.sum()), "strong_depth_pixels": int(strong.sum()),
                    "selected_depth_pixels": 0, "depth_components": components,
                })
    ids, inverse = np.unique(selected_ids, return_inverse=True)
    total = np.zeros(len(ids), dtype=np.float64)
    count = np.bincount(inverse, minlength=len(ids)).astype(np.float64)
    np.add.at(total, inverse, selected_delta)
    stable = mesh_index[overlap & (np.abs(residual_image) < float(stable_depth_residual))]
    return ids, total / count, np.unique(stable[stable >= 0]), {
        "overlap_pixels": int(overlap.sum()), "strong_depth_pixels": int(strong.sum()),
        "selected_depth_pixels": int(selected.sum()), "depth_components": components,
        "depth_residual_median_abs": float(np.median(np.abs(selected_delta))),
        "depth_residual_p90_abs": float(np.quantile(np.abs(selected_delta), .90)),
    }


def _screened_depth_field(
    graph: sparse.csr_matrix,
    movable: np.ndarray,
    controls: np.ndarray,
    values: np.ndarray,
    *,
    data_weight: float,
) -> np.ndarray:
    """Solve a smooth scalar Camera-depth field with soft pixel controls."""
    movable = np.unique(np.asarray(movable, dtype=np.int64))
    index = np.full(graph.shape[0], -1, dtype=np.int64)
    index[movable] = np.arange(len(movable), dtype=np.int64)
    control_local = index[np.asarray(controls, dtype=np.int64)]
    keep = control_local >= 0
    if not keep.any():
        raise ValueError("no visible depth controls lie in the admissible mesh region")
    control_local, values = control_local[keep], np.asarray(values, dtype=np.float64)[keep]
    coo = graph.tocoo()
    edge = coo.row < coo.col
    left, right = coo.row[edge], coo.col[edge]
    local_left, local_right = index[left], index[right]
    active = (local_left >= 0) | (local_right >= 0)
    left, right, local_left, local_right = (
        item[active] for item in (left, right, local_left, local_right)
    )
    diagonal = np.zeros(len(movable), dtype=np.float64)
    rows, cols, data = [], [], []
    both = (local_left >= 0) & (local_right >= 0)
    if both.any():
        rows.extend((local_left[both], local_right[both]))
        cols.extend((local_right[both], local_left[both]))
        data.extend((-np.ones(int(both.sum())), -np.ones(int(both.sum()))))
    for local, valid in ((local_left, local_left >= 0), (local_right, local_right >= 0)):
        np.add.at(diagonal, local[valid], 1.)
    target_sum = np.zeros(len(movable), dtype=np.float64)
    target_count = np.bincount(control_local, minlength=len(movable)).astype(np.float64)
    np.add.at(target_sum, control_local, values)
    diagonal += float(data_weight) * target_count
    rhs = float(data_weight) * target_sum
    rows.append(np.arange(len(movable), dtype=np.int64))
    cols.append(np.arange(len(movable), dtype=np.int64))
    data.append(diagonal)
    system = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(movable), len(movable)), dtype=np.float64,
    ).tocsc()
    field = np.zeros(graph.shape[0], dtype=np.float64)
    field[movable] = sparse_linalg.spsolve(system, rhs)
    return field


def _camera_projection_vertex_controls(
    vertices: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    max_pixel_distance: float,
    minimum_residual: float,
    stable_residual: float,
    minimum_component_pixels: int,
    camera_axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build camera-plane + depth controls from projected partial pixels.

    A partial point is paired with the closest *visible mesh pixel* in the
    saved camera within a small raster tolerance.  Its complete 3-D offset is
    kept in camera coordinates, thereby retaining both image-plane evidence
    and measured partial depth rather than using an arbitrary world KNN.
    """
    mesh_uv, mesh_depth = projector.project(vertices)
    partial_uv, partial_depth = projector.project(partial)
    _, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=1,
    )
    _, mesh_mask, mesh_index = zbuffer_depth_with_indices(
        mesh_uv, mesh_depth, projector.image_shape, splat_radius=1,
    )
    partial_yx = np.argwhere(partial_mask)
    mesh_yx = np.argwhere(mesh_mask)
    if min(len(partial_yx), len(mesh_yx)) == 0:
        return (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64), {"partial_visible_pixels": int(len(partial_yx)),
                                                "mesh_visible_pixels": int(len(mesh_yx))})
    distance, nearest = cKDTree(mesh_yx.astype(np.float64)).query(
        partial_yx.astype(np.float64), k=1, distance_upper_bound=float(max_pixel_distance), workers=-1,
    )
    valid = np.isfinite(distance) & (nearest < len(mesh_yx))
    pyx, myx = partial_yx[valid], mesh_yx[nearest[valid]]
    partial_ids = partial_index[pyx[:, 0], pyx[:, 1]]
    mesh_ids = mesh_index[myx[:, 0], myx[:, 1]]
    valid_ids = (partial_ids >= 0) & (mesh_ids >= 0)
    pyx, partial_ids, mesh_ids = pyx[valid_ids], partial_ids[valid_ids], mesh_ids[valid_ids]
    residual_world = partial[partial_ids] - vertices[mesh_ids]
    residual_camera = residual_world @ np.asarray(camera_axes, dtype=np.float64).T
    residual_norm = np.linalg.norm(residual_camera, axis=1)
    strong = residual_norm >= float(minimum_residual)
    selected = np.zeros(len(mesh_ids), dtype=bool)
    components = 0
    label_image = np.zeros(projector.image_shape, dtype=np.int32)
    # A pixel appears at most once in the partial z-buffer.  Connected regions
    # with coherent 3-D camera residuals are admissible controls.
    label_image[pyx[strong, 0], pyx[strong, 1]] = 1
    labels, count = ndimage.label(label_image)
    components = int(count)
    for label in range(1, count + 1):
        member = labels[pyx[:, 0], pyx[:, 1]] == label
        if int(member.sum()) < int(minimum_component_pixels):
            continue
        vectors = residual_camera[member]
        direction = np.median(vectors, axis=0)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-9:
            continue
        cosine = (vectors @ direction) / np.maximum(np.linalg.norm(vectors, axis=1) * direction_norm, 1e-12)
        if float(np.median(cosine)) >= .70:
            selected |= member
    ids = mesh_ids[selected]
    values = residual_camera[selected]
    stable = mesh_ids[residual_norm <= float(stable_residual)]
    if len(ids) == 0:
        return (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64), np.unique(stable), {
            "partial_visible_pixels": int(len(partial_yx)), "mesh_visible_pixels": int(len(mesh_yx)),
            "camera_pixel_pairs": int(len(mesh_ids)), "strong_projection_pixels": int(strong.sum()),
            "selected_projection_pixels": 0, "projection_components": components,
        })
    unique, inverse = np.unique(ids, return_inverse=True)
    total = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(total, inverse, values)
    return unique, total / counts[:, None], np.unique(stable), {
        "partial_visible_pixels": int(len(partial_yx)), "mesh_visible_pixels": int(len(mesh_yx)),
        "camera_pixel_pairs": int(len(mesh_ids)), "strong_projection_pixels": int(strong.sum()),
        "selected_projection_pixels": int(selected.sum()), "projection_components": components,
        "camera_residual_median": float(np.median(np.linalg.norm(values, axis=1))),
        "camera_residual_p90": float(np.quantile(np.linalg.norm(values, axis=1), .90)),
        "camera_depth_residual_median": float(np.median(np.abs(values[:, 2]))),
    }


def _screened_vector_field(
    graph: sparse.csr_matrix,
    movable: np.ndarray,
    controls: np.ndarray,
    values: np.ndarray,
    *,
    data_weight: float,
) -> np.ndarray:
    """Vector extension of the screened mesh field with shared topology."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("camera-projection controls must be (N, 3)")
    movable = np.unique(np.asarray(movable, dtype=np.int64))
    index = np.full(graph.shape[0], -1, dtype=np.int64)
    index[movable] = np.arange(len(movable), dtype=np.int64)
    local_controls = index[np.asarray(controls, dtype=np.int64)]
    keep = local_controls >= 0
    if not keep.any():
        raise ValueError("no projection controls lie in the admissible mesh region")
    local_controls, values = local_controls[keep], values[keep]
    coo = graph.tocoo()
    edge = coo.row < coo.col
    left, right = coo.row[edge], coo.col[edge]
    local_left, local_right = index[left], index[right]
    active = (local_left >= 0) | (local_right >= 0)
    left, right, local_left, local_right = (item[active] for item in (left, right, local_left, local_right))
    diagonal = np.zeros(len(movable), dtype=np.float64)
    rows, cols, data = [], [], []
    both = (local_left >= 0) & (local_right >= 0)
    if both.any():
        rows.extend((local_left[both], local_right[both])); cols.extend((local_right[both], local_left[both]))
        data.extend((-np.ones(int(both.sum())), -np.ones(int(both.sum()))))
    for local, valid in ((local_left, local_left >= 0), (local_right, local_right >= 0)):
        np.add.at(diagonal, local[valid], 1.)
    target_sum = np.zeros((len(movable), 3), dtype=np.float64)
    target_count = np.bincount(local_controls, minlength=len(movable)).astype(np.float64)
    np.add.at(target_sum, local_controls, values)
    diagonal += float(data_weight) * target_count
    rows.append(np.arange(len(movable), dtype=np.int64)); cols.append(np.arange(len(movable), dtype=np.int64)); data.append(diagonal)
    system = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(movable), len(movable)), dtype=np.float64,
    ).tocsc()
    field = np.zeros((graph.shape[0], 3), dtype=np.float64)
    field[movable] = np.asarray(sparse_linalg.spsolve(system, float(data_weight) * target_sum), dtype=np.float64)
    return field


def deform_mesh_from_camera_projection(
    vertices: np.ndarray,
    faces: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    camera_axes: np.ndarray,
    max_pixel_distance: float = 4.,
    projection_residual_ratio: float = .035,
    stable_residual_ratio: float = .0175,
    minimum_component_pixels: int = 32,
    core_geodesic_radius_ratio: float = .035,
    influence_geodesic_radius_ratio: float = .12,
    maximum_influence_fraction: float = .30,
    projection_data_weight: float = .02,
    minimum_projection_improvement: float = .08,
) -> tuple[np.ndarray, dict]:
    """Adapt a mesh using partial camera-plane and depth evidence jointly."""
    vertices, faces, partial = (np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64),
                                np.asarray(partial, dtype=np.float64))
    zero = np.zeros_like(vertices)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    if not (max_pixel_distance > 0. and 0. < stable_residual_ratio < projection_residual_ratio
            and 0. < core_geodesic_radius_ratio <= influence_geodesic_radius_ratio and 0. < maximum_influence_fraction <= 1.
            and projection_data_weight > 0. and 0. < minimum_projection_improvement < 1.):
        raise ValueError("invalid camera-projection mesh deformation bounds")
    controls, residuals, stable, observation = _camera_projection_vertex_controls(
        vertices, partial, projector, max_pixel_distance=float(max_pixel_distance),
        minimum_residual=float(projection_residual_ratio) * diagonal,
        stable_residual=float(stable_residual_ratio) * diagonal,
        minimum_component_pixels=int(minimum_component_pixels), camera_axes=np.asarray(camera_axes, dtype=np.float64),
    )
    base = {"method": "camera1_partial_projection_depth_mesh_deformation", "strict_zero_shot": True,
            "ground_truth_cd_emd_used": False, "partial_diagonal": diagonal, **observation}
    if len(controls) < max(6, int(minimum_component_pixels)):
        return zero, {"active": False, **base, "reason": "insufficient_coherent_camera_projection_controls"}
    graph = mesh_surface_graph(vertices, faces)
    geodesic = _surface_distance(graph, controls)
    core = np.flatnonzero(geodesic <= float(core_geodesic_radius_ratio) * diagonal)
    influence = np.flatnonzero(geodesic <= float(influence_geodesic_radius_ratio) * diagonal)
    if len(core) < max(6, int(minimum_component_pixels)) or len(influence) / len(vertices) > float(maximum_influence_fraction):
        return zero, {"active": False, **base, "reason": "unsupported_or_overbroad_camera_projection_region",
                      "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence))}
    outside = np.setdiff1d(np.arange(len(vertices), dtype=np.int64), influence, assume_unique=True)
    fixed = np.unique(np.r_[stable, outside])
    movable = np.setdiff1d(influence, fixed, assume_unique=False)
    control_mask = np.isin(controls, movable)
    if len(movable) < max(6, int(minimum_component_pixels)) or int(control_mask.sum()) < max(6, int(minimum_component_pixels)):
        return zero, {"active": False, **base, "reason": "camera_projection_region_overconstrained"}
    raw_camera_field = _screened_vector_field(
        graph, movable, controls, residuals, data_weight=float(projection_data_weight),
    )
    world_field = raw_camera_field @ np.asarray(camera_axes, dtype=np.float64)
    observed_controls, observed_residuals = controls[control_mask], residuals[control_mask]
    before = np.linalg.norm(observed_residuals, axis=1)
    accepted = None
    best_improvement = -np.inf
    last_quality: dict | None = None
    for alpha in (1., .75, .5, .35, .25, .15, .1, .075, .05, .03):
        displacement = float(alpha) * world_field
        achieved_camera = displacement[observed_controls] @ np.asarray(camera_axes, dtype=np.float64).T
        after = np.linalg.norm(observed_residuals - achieved_camera, axis=1)
        improvement = 1. - float(np.median(after) / max(np.median(before), 1e-12))
        best_improvement = max(best_improvement, improvement)
        if improvement < float(minimum_projection_improvement):
            continue
        quality = _surface_orientation_quality(vertices, vertices + displacement, faces)
        last_quality = quality
        if quality["flipped_resolved_triangle_fraction"] <= 1e-5:
            accepted = alpha, displacement, improvement, quality, after
            break
    if accepted is None:
        reason = ("insufficient_camera_projection_improvement" if best_improvement < float(minimum_projection_improvement)
                  else "mesh_triangle_orientation_violation")
        return zero, {"active": False, **base, "reason": reason,
                      "best_observable_projection_improvement": float(best_improvement), **(last_quality or {})}
    alpha, displacement, improvement, quality, after = accepted
    moved = np.linalg.norm(displacement, axis=1) > 1e-8
    return displacement, {
        "active": True, **base, "field": "camera1_projection_depth_screened_mesh_field",
        "camera_projection_controls": int(len(observed_controls)), "camera_projection_stable_vertices": int(len(stable)),
        "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence)),
        "outer_fixed_mesh_vertices": int(len(outside)), "movable_mesh_vertices": int(len(movable)),
        "topology_backtracking_alpha": float(alpha), "projection_data_weight": float(projection_data_weight),
        "projection_residual_median_before": float(np.median(before)),
        "projection_residual_median_after": float(np.median(after)), "projection_residual_improvement": float(improvement),
        "camera_depth_residual_median_after": float(np.median(np.abs(observed_residuals[:, 2] - (displacement[observed_controls] @ np.asarray(camera_axes, dtype=np.float64).T)[:, 2]))),
        "moved_mesh_vertices": int(moved.sum()), "moved_mesh_fraction": float(moved.mean()),
        "maximum_displacement": float(np.linalg.norm(displacement, axis=1).max(initial=0.)), **quality,
    }


def deform_mesh_from_camera_carrier_controls(
    vertices: np.ndarray,
    faces: np.ndarray,
    registered_carrier: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    camera_axes: np.ndarray,
    minimum_residual_ratio: float = .15,
    stable_residual_ratio: float = .05,
    minimum_component_pairs: int = 128,
    core_geodesic_radius_ratio: float = .035,
    influence_geodesic_radius_ratio: float = .12,
    maximum_influence_fraction: float = .30,
    projection_data_weight: float = .02,
    minimum_projection_improvement: float = .08,
) -> tuple[np.ndarray, dict]:
    """Transfer pixel-indexed partial 2-D+depth evidence onto a mesh field.

    The 100k registered carrier is a uniform surface sample and is therefore a
    more stable saved-view raster than raw GLB vertices.  Its coherent
    Camera-1 partial correspondences are transferred to the textured mesh
    only after residual screening; both camera-plane and depth components are
    retained in the mesh deformation target.
    """
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    prior, partial = np.asarray(registered_carrier, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    zero = np.zeros_like(vertices)
    if not (0. < stable_residual_ratio < minimum_residual_ratio and 0. < core_geodesic_radius_ratio <= influence_geodesic_radius_ratio
            and 0. < maximum_influence_fraction <= 1. and projection_data_weight > 0. and 0. < minimum_projection_improvement < 1.):
        raise ValueError("invalid carrier-projection mesh deformation bounds")
    evidence, base = _coherent_residual_selection(
        prior, partial, projector, np.asarray(camera_axes, dtype=np.float64),
        max_pixel_distance=1.,
        minimum_residual_ratio=float(minimum_residual_ratio),
        screen_component_radius=3.,
        minimum_component_pairs=int(minimum_component_pairs),
        minimum_directional_coherence=.85,
        minimum_camera_axis_dominance=.75,
        component_direction_cosine=.95,
    )
    if evidence is None:
        return zero, {"active": False, "method": "camera1_carrier_projection_depth_mesh_deformation", **base}
    diagonal = float(evidence["diagonal"])
    prior_ids = np.asarray(evidence["prior_ids"], dtype=np.int64)
    partial_ids = np.asarray(evidence["partial_ids"], dtype=np.int64)
    records = np.asarray(evidence["record_indices"], dtype=np.int64)
    residual_norm = np.asarray(evidence["residual_norm"], dtype=np.float64)
    graph = mesh_surface_graph(vertices, faces)
    tree = cKDTree(vertices)
    control_vertices = tree.query(prior[prior_ids[records]], k=1, workers=-1)[1]
    residual_camera = (partial[partial_ids[records]] - prior[prior_ids[records]]) @ np.asarray(camera_axes, dtype=np.float64).T
    controls, inverse = np.unique(control_vertices, return_inverse=True)
    totals = np.zeros((len(controls), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(controls)).astype(np.float64)
    np.add.at(totals, inverse, residual_camera)
    controls_camera_residual = totals / counts[:, None]
    stable = np.unique(tree.query(
        prior[prior_ids[residual_norm <= float(stable_residual_ratio) * diagonal]], k=1, workers=-1,
    )[1])
    seeds = np.setdiff1d(controls, stable, assume_unique=False)
    if len(seeds) < max(6, int(minimum_component_pairs) // 2):
        return zero, {"active": False, "method": "camera1_carrier_projection_depth_mesh_deformation", **base,
                      "reason": "insufficient_unanchored_camera_carrier_controls"}
    geodesic = _surface_distance(graph, seeds)
    core = np.flatnonzero(geodesic <= float(core_geodesic_radius_ratio) * diagonal)
    influence = np.flatnonzero(geodesic <= float(influence_geodesic_radius_ratio) * diagonal)
    if len(core) < max(6, int(minimum_component_pairs) // 2) or len(influence) / len(vertices) > float(maximum_influence_fraction):
        return zero, {"active": False, "method": "camera1_carrier_projection_depth_mesh_deformation", **base,
                      "reason": "unsupported_or_overbroad_carrier_projection_region",
                      "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence))}
    outside = np.setdiff1d(np.arange(len(vertices), dtype=np.int64), influence, assume_unique=True)
    fixed = np.unique(np.r_[stable, outside])
    movable = np.setdiff1d(influence, fixed, assume_unique=False)
    control_mask = np.isin(controls, movable)
    if len(movable) < max(6, int(minimum_component_pairs) // 2) or int(control_mask.sum()) < max(6, int(minimum_component_pairs) // 2):
        return zero, {"active": False, "method": "camera1_carrier_projection_depth_mesh_deformation", **base,
                      "reason": "carrier_projection_region_overconstrained"}
    raw_camera_field = _screened_vector_field(
        graph, movable, controls, controls_camera_residual, data_weight=float(projection_data_weight),
    )
    axes = np.asarray(camera_axes, dtype=np.float64)
    world_field = raw_camera_field @ axes
    observed_controls, observed_residuals = controls[control_mask], controls_camera_residual[control_mask]
    before = np.linalg.norm(observed_residuals, axis=1)
    accepted = None
    best_improvement = -np.inf
    last_quality: dict | None = None
    for alpha in (1., .75, .5, .35, .25, .15, .1, .075, .05, .03):
        displacement = float(alpha) * world_field
        achieved = displacement[observed_controls] @ axes.T
        after = np.linalg.norm(observed_residuals - achieved, axis=1)
        improvement = 1. - float(np.median(after) / max(np.median(before), 1e-12))
        best_improvement = max(best_improvement, improvement)
        if improvement < float(minimum_projection_improvement):
            continue
        quality = _surface_orientation_quality(vertices, vertices + displacement, faces)
        last_quality = quality
        if quality["flipped_resolved_triangle_fraction"] <= 1e-5:
            accepted = alpha, displacement, improvement, quality, after
            break
    common = {"method": "camera1_carrier_projection_depth_mesh_deformation", "strict_zero_shot": True,
              "ground_truth_cd_emd_used": False, **base,
              "selected_carrier_controls": int(len(controls)), "stable_mesh_vertices": int(len(stable)),
              "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence)),
              "outer_fixed_mesh_vertices": int(len(outside)), "movable_mesh_vertices": int(len(movable))}
    if accepted is None:
        reason = ("insufficient_carrier_projection_improvement" if best_improvement < float(minimum_projection_improvement)
                  else "mesh_triangle_orientation_violation")
        return zero, {"active": False, **common, "reason": reason,
                      "best_observable_projection_improvement": float(best_improvement), **(last_quality or {})}
    alpha, displacement, improvement, quality, after = accepted
    moved = np.linalg.norm(displacement, axis=1) > 1e-8
    return displacement, {
        "active": True, **common, "field": "carrier_pixel_indexed_camera_projection_depth_mesh_field",
        "topology_backtracking_alpha": float(alpha), "projection_data_weight": float(projection_data_weight),
        "projection_residual_median_before": float(np.median(before)),
        "projection_residual_median_after": float(np.median(after)), "projection_residual_improvement": float(improvement),
        "camera_depth_residual_median_before": float(np.median(np.abs(observed_residuals[:, 2]))),
        "camera_depth_residual_median_after": float(np.median(np.abs(observed_residuals[:, 2] - (displacement[observed_controls] @ axes.T)[:, 2]))),
        "moved_mesh_vertices": int(moved.sum()), "moved_mesh_fraction": float(moved.mean()),
        "maximum_displacement": float(np.linalg.norm(displacement, axis=1).max(initial=0.)), **quality,
    }


def deform_mesh_from_camera_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    camera_axes: np.ndarray,
    depth_residual_ratio: float = .035,
    stable_depth_ratio: float = .0175,
    minimum_component_pixels: int = 32,
    core_geodesic_radius_ratio: float = .035,
    influence_geodesic_radius_ratio: float = .12,
    maximum_influence_fraction: float = .30,
    depth_data_weight: float = .08,
    minimum_depth_improvement: float = .08,
) -> tuple[np.ndarray, dict]:
    """Camera-conditioned mesh depth adaptation from partial depth evidence.

    Visible mesh vertices receive depth-only constraints at identical saved
    Camera-1 pixels.  Stable pixels and the outer surface are fixed; a scalar
    field is smoothly propagated on a UV-seam-welded mesh graph and converted
    to world-space motion along the Camera-1 depth axis.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    partial = np.asarray(partial, dtype=np.float64)
    zero = np.zeros_like(vertices)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    if not (0. < stable_depth_ratio < depth_residual_ratio and 0. < core_geodesic_radius_ratio <= influence_geodesic_radius_ratio
            and 0. < maximum_influence_fraction <= 1. and depth_data_weight > 0. and 0. < minimum_depth_improvement < 1.):
        raise ValueError("invalid camera-depth mesh deformation bounds")
    controls, residuals, stable, observation = _camera_depth_vertex_controls(
        vertices, partial, projector,
        minimum_depth_residual=float(depth_residual_ratio) * diagonal,
        stable_depth_residual=float(stable_depth_ratio) * diagonal,
        minimum_component_pixels=int(minimum_component_pixels),
    )
    base = {"method": "camera1_partial_depth_mesh_deformation", "strict_zero_shot": True,
            "ground_truth_cd_emd_used": False, "partial_diagonal": diagonal, **observation}
    if len(controls) < max(6, int(minimum_component_pixels)):
        return zero, {"active": False, **base, "reason": "insufficient_coherent_camera_depth_controls"}
    graph = mesh_surface_graph(vertices, faces)
    geodesic = _surface_distance(graph, controls)
    core = np.flatnonzero(geodesic <= float(core_geodesic_radius_ratio) * diagonal)
    influence = np.flatnonzero(geodesic <= float(influence_geodesic_radius_ratio) * diagonal)
    if len(core) < max(6, int(minimum_component_pixels)) or len(influence) / len(vertices) > float(maximum_influence_fraction):
        return zero, {"active": False, **base, "reason": "unsupported_or_overbroad_camera_depth_region",
                      "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence))}
    outside = np.setdiff1d(np.arange(len(vertices), dtype=np.int64), influence, assume_unique=True)
    fixed = np.unique(np.r_[stable, outside])
    movable = np.setdiff1d(influence, fixed, assume_unique=False)
    if len(movable) < max(6, int(minimum_component_pixels)):
        return zero, {"active": False, **base, "reason": "camera_depth_region_overconstrained"}
    raw_field = _screened_depth_field(graph, movable, controls, residuals, data_weight=float(depth_data_weight))
    depth_axis = np.asarray(camera_axes, dtype=np.float64)[2]
    if not np.isfinite(depth_axis).all() or abs(np.linalg.norm(depth_axis) - 1.) > 1e-4:
        raise ValueError("camera_axes must be orthonormal")
    movable_control = np.isin(controls, movable)
    observed_controls, observed_residuals = controls[movable_control], residuals[movable_control]
    before = np.abs(observed_residuals)
    accepted = None
    best_improvement = -np.inf
    last_quality: dict | None = None
    for alpha in (1., .75, .5, .35, .25, .15, .1, .075, .05, .03):
        field = float(alpha) * raw_field
        displacement = field[:, None] * depth_axis[None, :]
        after = np.abs(observed_residuals - field[observed_controls])
        improvement = 1. - float(np.median(after) / max(np.median(before), 1e-12))
        best_improvement = max(best_improvement, improvement)
        if improvement < float(minimum_depth_improvement):
            continue
        quality = _surface_orientation_quality(vertices, vertices + displacement, faces)
        last_quality = quality
        # Textured generative meshes frequently contain a handful of extremely
        # skinny seam-adjacent triangles. Treating a few of those as a global
        # veto discards an otherwise smooth, camera-supported edit. The gate
        # remains strict at surface level: only a negligible fraction may flip
        # and the edge-length tails must remain well behaved.
        if (quality["flipped_resolved_triangle_fraction"] <= 5e-5
                and quality["edge_stretch_p01"] >= .85
                and quality["edge_stretch_p999"] <= 1.5):
            accepted = alpha, displacement, improvement, quality, after
            break
    if accepted is None:
        reason = ("insufficient_camera_depth_improvement" if best_improvement < float(minimum_depth_improvement)
                  else "mesh_triangle_orientation_violation")
        return zero, {"active": False, **base, "reason": reason,
                      "best_observable_depth_improvement": float(best_improvement), **(last_quality or {})}
    alpha, displacement, improvement, quality, after = accepted
    moved = np.linalg.norm(displacement, axis=1) > 1e-8
    return displacement, {
        "active": True, **base,
        "field": "camera1_pixel_depth_screened_mesh_field",
        "camera_depth_controls": int(len(observed_controls)),
        "camera_depth_stable_vertices": int(len(stable)),
        "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence)),
        "outer_fixed_mesh_vertices": int(len(outside)), "movable_mesh_vertices": int(len(movable)),
        "topology_backtracking_alpha": float(alpha), "depth_data_weight": float(depth_data_weight),
        "depth_residual_median_before": float(np.median(before)),
        "depth_residual_median_after": float(np.median(after)),
        "depth_residual_improvement": float(improvement),
        "moved_mesh_vertices": int(moved.sum()), "moved_mesh_fraction": float(moved.mean()),
        "maximum_depth_displacement": float(np.abs(displacement @ depth_axis).max(initial=0.)),
        **quality,
    }


def deform_registered_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    registered_carrier: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    camera_axes: np.ndarray,
    auxiliary_support: np.ndarray | None = None,
    auxiliary_support_max_distance: float | None = None,
    minimum_residual_ratio: float = .15,
    minimum_component_pairs: int = 128,
    stable_residual_ratio: float = .05,
    minimum_stable_pairs: int = 512,
    core_geodesic_radius_ratio: float = .04,
    influence_geodesic_radius_ratio: float = .10,
    maximum_influence_fraction: float = .30,
    maximum_log_scale: float = .60,
    minimum_component_improvement: float = .20,
    max_pixel_distance: float = 1.,
    screen_component_radius: float = 3.,
    minimum_directional_coherence: float = .85,
    minimum_camera_axis_dominance: float = .75,
    component_direction_cosine: float = .95,
    graph_screening: float = .0015,
) -> tuple[np.ndarray, dict]:
    """Deform a globally registered mesh with attachment-preserving axial scale."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    prior = np.asarray(registered_carrier, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    zero = np.zeros_like(vertices)
    if not (0. < core_geodesic_radius_ratio <= influence_geodesic_radius_ratio
            and 0. < maximum_influence_fraction <= 1. and maximum_log_scale > 0.
            and 0. < minimum_component_improvement < 1.):
        raise ValueError("invalid mesh deformation bounds")
    evidence, base = _coherent_residual_selection(
        prior, partial, projector, np.asarray(camera_axes, dtype=np.float64),
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
    prior_ids = np.asarray(evidence["prior_ids"], dtype=np.int64)
    partial_ids = np.asarray(evidence["partial_ids"], dtype=np.int64)
    records = np.asarray(evidence["record_indices"], dtype=np.int64)
    residual_norm = np.asarray(evidence["residual_norm"], dtype=np.float64)
    graph = mesh_surface_graph(vertices, faces)
    mesh_tree = cKDTree(vertices)
    selected_vertices = np.unique(mesh_tree.query(prior[prior_ids[records]], k=1, workers=-1)[1])
    stable_vertices = np.unique(mesh_tree.query(
        prior[prior_ids[residual_norm <= float(stable_residual_ratio) * diagonal]], k=1, workers=-1,
    )[1])
    seeds = np.setdiff1d(selected_vertices, stable_vertices, assume_unique=False)
    if len(seeds) < max(6, int(minimum_component_pairs) // 2) or len(stable_vertices) < int(minimum_stable_pairs):
        return zero, {"active": False, **base, "reason": "insufficient_mesh_supported_controls"}
    geodesic = _surface_distance(graph, seeds)
    core = np.flatnonzero(geodesic <= float(core_geodesic_radius_ratio) * diagonal)
    influence = np.flatnonzero(geodesic <= float(influence_geodesic_radius_ratio) * diagonal)
    if (len(core) < max(6, int(minimum_component_pairs) // 2)
            or float(len(influence) / len(vertices)) > float(maximum_influence_fraction)):
        return zero, {"active": False, **base, "reason": "unsupported_or_overbroad_mesh_patch",
                      "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence))}
    auxiliary_vertices = np.empty(0, dtype=np.int64)
    if auxiliary_support is not None:
        support = np.asarray(auxiliary_support, dtype=np.float64)
        if support.ndim != 2 or support.shape[1] != 3 or len(support) < 6:
            raise ValueError("auxiliary_support must be a nontrivial (N, 3) cloud")
        if auxiliary_support_max_distance is None or float(auxiliary_support_max_distance) <= 0.:
            raise ValueError("auxiliary support requires a positive distance")
        distance, _ = cKDTree(support).query(vertices, k=1, workers=-1)
        auxiliary_vertices = np.flatnonzero(distance <= float(auxiliary_support_max_distance))
    boundary = _edge_boundary(graph, core)
    attachment = np.intersect1d(boundary, stable_vertices, assume_unique=False)
    if len(attachment) < 6:
        attachment = np.intersect1d(boundary, auxiliary_vertices, assume_unique=False)
    if len(attachment) < 6:
        return zero, {"active": False, **base, "reason": "missing_observed_mesh_attachment"}
    outside = np.setdiff1d(np.arange(len(vertices), dtype=np.int64), influence, assume_unique=True)
    # The partial scan is the primary geometric observation.  MoGe is an
    # auxiliary camera-consistency cue: use it to find a reliable attachment
    # when partial coverage is sparse, but do not freeze its entire visible
    # footprint inside a partial-derived residual region.  Otherwise the
    # auxiliary image prior would partition a physically continuous edit into
    # many isolated moving islands.
    fixed = np.unique(np.r_[stable_vertices, outside])
    movable = np.setdiff1d(influence, fixed, assume_unique=False)
    if len(movable) < max(6, int(minimum_component_pairs) // 2):
        return zero, {"active": False, **base, "reason": "mesh_core_overconstrained_by_support"}
    controls, targets = _partial_control_targets(
        vertices, prior, partial, prior_ids, partial_ids, records, mesh_tree,
    )
    movable_mask = np.zeros(len(vertices), dtype=bool)
    movable_mask[movable] = True
    control_mask = movable_mask[controls]
    if int(control_mask.sum()) < max(6, int(minimum_component_pairs) // 2):
        return zero, {"active": False, **base, "reason": "mesh_residual_controls_fixed_by_partial"}
    raw_displacement, arap = _arap_surface_deformation(
        vertices, graph, movable, fixed, controls, targets,
    )
    active_controls, active_targets = controls[control_mask], targets[control_mask]
    before = np.linalg.norm(vertices[active_controls] - active_targets, axis=1)
    # A local ARAP solve can still be too strong on a thin generated surface.
    # Select the strongest globally scaled update that improves observed
    # controls and preserves orientation on the resolved mesh.
    accepted = None
    best_observable_improvement = -np.inf
    last_quality: dict | None = None
    for alpha in (1., .75, .5, .35, .25, .15, .1):
        candidate_displacement = float(alpha) * raw_displacement
        candidate_deformed = vertices + candidate_displacement
        after = np.linalg.norm(candidate_deformed[active_controls] - active_targets, axis=1)
        improvement = 1. - float(np.median(after) / max(np.median(before), 1e-12))
        best_observable_improvement = max(best_observable_improvement, improvement)
        if improvement < float(minimum_component_improvement):
            continue
        quality = _surface_orientation_quality(vertices, candidate_deformed, faces)
        last_quality = quality
        if quality["flipped_resolved_triangle_fraction"] <= 1e-5:
            accepted = (alpha, improvement, candidate_displacement, quality)
            break
    if accepted is None:
        reason = ("insufficient_mesh_arap_improvement" if best_observable_improvement < float(minimum_component_improvement)
                  else "mesh_triangle_orientation_violation")
        return zero, {"active": False, **base, "reason": reason,
                      "best_observable_component_improvement": float(best_observable_improvement),
                      **(last_quality or {})}
    alpha, improvement, displacement, quality = accepted
    moved = np.linalg.norm(displacement, axis=1) > 1e-8
    return displacement, {
        "active": True,
        "method": "partial_anchored_mesh_attached_gaussian_deformation",
        "field": "uv_seam_welded_partial_anchored_arap_surface_field",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        **base,
        "selected_component_pairs": int(len(records)),
        "mesh_vertices": int(len(vertices)), "mesh_faces": int(len(faces)),
        "core_mesh_vertices": int(len(core)), "influence_mesh_vertices": int(len(influence)),
        "partial_stable_mesh_vertices": int(len(stable_vertices)),
        "auxiliary_boundary_candidate_vertices": int(len(auxiliary_vertices)),
        "attachment_mesh_vertices": int(len(attachment)), "outer_fixed_mesh_vertices": int(len(outside)),
        "topology_backtracking_alpha": float(alpha),
        "component_residual_median_before": float(np.median(before)),
        "component_residual_median_after": float(np.median(
            np.linalg.norm((vertices + displacement)[active_controls] - active_targets, axis=1))),
        "component_improvement": improvement, **arap,
        "moved_mesh_vertices": int(moved.sum()), "moved_mesh_fraction": float(moved.mean()),
        "maximum_displacement": float(np.linalg.norm(displacement, axis=1).max(initial=0.)),
        **quality,
        "carrier_slots_preserved": True,
    }
