"""Visibility-conditioned residual Sim(3) plus intrinsic mesh deformation.

The accepted v15 registration is an immutable initialization.  A small
proper-Sim(3) trust-region update explains coherent global residuals; only the
remaining observation-supported field is passed to an intrinsic mesh solve.
The local field is projected away from all similarity modes so global pose and
local shape cannot explain the same motion.
"""

from __future__ import annotations

import math

import numpy as np
import open3d as o3d
from scipy.ndimage import label
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
import trimesh

from src.ray_consistent_registration import soft_ray_correspondences, zbuffer_indices


def load_scene_mesh(path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh in {path}")
    return mesh


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    points, face_ids = trimesh.sample.sample_surface(
        mesh, int(count), seed=np.random.default_rng(int(seed)))
    return np.asarray(points, dtype=np.float64), np.asarray(face_ids, dtype=np.int64)


def build_proxy(mesh: trimesh.Trimesh, target_triangles: int) -> trimesh.Trimesh:
    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))
    source.remove_duplicated_vertices()
    source.remove_duplicated_triangles()
    source.remove_degenerate_triangles()
    source.remove_unreferenced_vertices()
    if len(source.triangles) > int(target_triangles):
        source = source.simplify_quadric_decimation(
            target_number_of_triangles=int(target_triangles),
            maximum_error=float("inf"), boundary_weight=1.0)
    source.remove_degenerate_triangles()
    source.remove_duplicated_triangles()
    source.remove_unreferenced_vertices()
    return trimesh.Trimesh(
        vertices=np.asarray(source.vertices, dtype=np.float64),
        faces=np.asarray(source.triangles, dtype=np.int64), process=False)


def mesh_edges(vertices: np.ndarray, faces: np.ndarray):
    edges = np.unique(np.sort(np.concatenate((
        faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)],
    ), axis=0), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return edges, lengths


def mesh_geodesic(vertices: np.ndarray, faces: np.ndarray, sources: np.ndarray) -> np.ndarray:
    edges, lengths = mesh_edges(vertices, faces)
    graph = sparse.coo_matrix((
        np.concatenate((lengths, lengths)),
        (np.concatenate((edges[:, 0], edges[:, 1])),
         np.concatenate((edges[:, 1], edges[:, 0]))),
    ), shape=(len(vertices), len(vertices))).tocsr()
    distance = dijkstra(
        graph, directed=False, indices=np.asarray(sources, dtype=np.int64),
        min_only=True)
    return np.asarray(distance, dtype=np.float64)


def compact_weight(distance: np.ndarray, inner: float, outer: float) -> np.ndarray:
    value = np.clip((float(outer) - distance) / max(float(outer - inner), 1e-12), 0., 1.)
    return value * value * (3. - 2. * value)


def similarity_design(vertices: np.ndarray, center: np.ndarray) -> np.ndarray:
    x = np.asarray(vertices, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    design = np.zeros((len(x), 3, 7), dtype=np.float64)
    design[:, 0, 0] = 1.; design[:, 1, 1] = 1.; design[:, 2, 2] = 1.
    design[:, 1, 3] = -x[:, 2]; design[:, 2, 3] = x[:, 1]
    design[:, 0, 4] = x[:, 2]; design[:, 2, 4] = -x[:, 0]
    design[:, 0, 5] = -x[:, 1]; design[:, 1, 5] = x[:, 0]
    design[:, :, 6] = x
    return design


def remove_similarity_modes(vertices: np.ndarray, displacement: np.ndarray,
                            support: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project a compact field away from translation, rotation, and scale."""
    vertices = np.asarray(vertices, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    active = support > 1e-6
    if int(active.sum()) < 8:
        return displacement.copy(), {"valid": False, "reason": "insufficient_support"}
    center = np.average(vertices[active], axis=0, weights=support[active])
    design = similarity_design(vertices, center)
    supported_design = support[:, None, None] * design
    matrix = supported_design[active].reshape(-1, 7)
    target = displacement[active].reshape(-1)
    coefficient = np.linalg.lstsq(matrix, target, rcond=None)[0]
    mode = np.einsum("nij,j->ni", supported_design, coefficient)
    projected = displacement - mode
    return projected, {
        "valid": True,
        "translation": coefficient[:3].tolist(),
        "rotation_vector": coefficient[3:6].tolist(),
        "isotropic_scale_increment": float(coefficient[6]),
        "removed_mode_rms": float(np.sqrt(np.mean(mode[active] ** 2))),
    }


def edge_flip_metrics(mesh: trimesh.Trimesh, candidate_vertices: np.ndarray) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges, before = mesh_edges(vertices, faces)
    after = np.linalg.norm(
        candidate_vertices[edges[:, 0]] - candidate_vertices[edges[:, 1]], axis=1)
    stretch = after / np.maximum(before, 1e-12)
    normal_before = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normal_after = np.cross(
        candidate_vertices[faces[:, 1]] - candidate_vertices[faces[:, 0]],
        candidate_vertices[faces[:, 2]] - candidate_vertices[faces[:, 0]])
    cosine = np.sum(normal_before * normal_after, axis=1) / np.maximum(
        np.linalg.norm(normal_before, axis=1) * np.linalg.norm(normal_after, axis=1), 1e-12)
    return {
        "edge_stretch_q01": float(np.quantile(stretch, .01)),
        "edge_stretch_q99": float(np.quantile(stretch, .99)),
        "flipped_face_ratio": float(np.mean(cosine < 0.)),
        "normal_cosine_q01": float(np.quantile(cosine, .01)),
    }


def screened_intrinsic_solve(mesh: trimesh.Trimesh, handle_ids: np.ndarray,
                             handle_targets: np.ndarray, anchor_ids: np.ndarray,
                             *, data_weight: float, smooth_weight: float,
                             anchor_weight: float, identity_weight: float):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges, _ = mesh_edges(vertices, faces)
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(row)), (row, col)), shape=(len(vertices), len(vertices))).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = 1. / np.sqrt(np.maximum(degree, 1.))
    laplacian = sparse.eye(len(vertices), format="csr") - (
        sparse.diags(inverse) @ adjacency @ sparse.diags(inverse))
    diagonal = np.full(len(vertices), float(identity_weight), dtype=np.float64)
    diagonal[handle_ids] += float(data_weight)
    diagonal[anchor_ids] += float(anchor_weight)
    system = (float(smooth_weight) * (laplacian.T @ laplacian)
              + sparse.diags(diagonal)).tocsr()
    rhs = np.zeros((len(vertices), 3), dtype=np.float64)
    rhs[handle_ids] = float(data_weight) * (handle_targets - vertices[handle_ids])
    preconditioner = sparse.diags(1. / np.maximum(system.diagonal(), 1e-12))
    solved, convergence = [], []
    for axis in range(3):
        value, info = cg(
            system, rhs[:, axis], M=preconditioner,
            rtol=1e-6, atol=0., maxiter=400)
        solved.append(value); convergence.append(int(info))
    return np.column_stack(solved), {
        "vertices": int(len(vertices)), "edges": int(len(edges)),
        "cg_info_xyz": convergence,
    }


def farthest_subset(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    count = min(int(count), len(points))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmax(np.linalg.norm(points - np.median(points, axis=0), axis=1)))
    nearest = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(nearest))
        nearest = np.minimum(nearest, np.sum((points - points[selected[index]]) ** 2, axis=1))
    return selected


def visible_residual_components(partial: np.ndarray, body: np.ndarray, projector,
                                *, diagonal: float, pixel_radius: float = 8.,
                                residual_quantile: float = .70,
                                residual_min_ratio: float = .018,
                                residual_max_ratio: float = .14,
                                min_points: int = 48,
                                max_components: int = 8):
    """Return strong 8-connected residual patches without merging by dilation."""
    partial_uv, partial_depth = projector.project(partial)
    body_uv, body_depth = projector.project(body)
    _, partial_mask, partial_index = zbuffer_indices(
        partial_uv, partial_depth, projector.image_shape)
    _, body_mask, body_index = zbuffer_indices(
        body_uv, body_depth, projector.image_shape, splat_radius=1)
    py, px = np.where(partial_mask); by, bx = np.where(body_mask)
    if not len(px) or not len(bx):
        return [], {"reason": "empty_projection"}
    partial_ids = partial_index[py, px]
    pixel_distance, nearest = cKDTree(np.c_[bx, by]).query(np.c_[px, py])
    body_ids = body_index[by[nearest], bx[nearest]]
    distance = np.linalg.norm(
        partial[partial_ids] - body[body_ids], axis=1) / max(float(diagonal), 1e-12)
    near = pixel_distance <= float(pixel_radius)
    threshold = max(float(residual_min_ratio), float(np.quantile(
        distance[near], float(residual_quantile))) if near.any() else float("inf"))
    high = near & (distance >= threshold) & (distance <= float(residual_max_ratio))
    raster = np.zeros(projector.image_shape, dtype=bool)
    raster[py[high], px[high]] = True
    components, count = label(raster, structure=np.ones((3, 3), dtype=np.uint8))
    result = []
    for component_id in range(1, int(count) + 1):
        selected = high & (components[py, px] == component_id)
        if int(selected.sum()) < int(min_points):
            continue
        source = body[body_ids[selected]]
        target = partial[partial_ids[selected]]
        covariance = np.cov(target.T)
        values = np.linalg.eigvalsh(covariance)[::-1]
        info = {
            "visible_partial_points": int(len(partial_ids)),
            "component_points": int(selected.sum()),
            "component_fraction": float(selected.sum() / len(partial_ids)),
            "line_likeness": float(values[0] / max(values[1] + values[2], 1e-12)),
            "target_span_ratio": float(np.linalg.norm(np.ptp(target, axis=0)) / diagonal),
            "source_span_ratio": float(np.linalg.norm(np.ptp(source, axis=0)) / diagonal),
            "median_residual_ratio": float(np.median(distance[selected])),
            "residual_threshold_ratio": float(threshold),
        }
        result.append(((source, target), info))
    result.sort(
        key=lambda item: item[1]["component_points"] * item[1]["median_residual_ratio"],
        reverse=True)
    return result[:int(max_components)], {
        "reason": "components_found" if result else "no_compact_residual_component",
        "residual_threshold_ratio": float(threshold),
        "component_count": int(len(result)),
    }


def _aggregate_handles(proxy: trimesh.Trimesh, surface: np.ndarray,
                       face_ids: np.ndarray, source: np.ndarray,
                       target: np.ndarray, *, diagonal: float,
                       max_handles: int, displacement_cap_ratio: float):
    _, sample_ids = cKDTree(surface).query(source, k=1, workers=-1)
    triangles = np.asarray(proxy.faces, dtype=np.int64)[face_ids[sample_ids]]
    triangle_vertices = np.asarray(proxy.vertices)[triangles]
    corner = np.argmin(np.linalg.norm(triangle_vertices - source[:, None, :], axis=2), axis=1)
    vertex_ids = triangles[np.arange(len(triangles)), corner]
    groups: dict[int, list[np.ndarray]] = {}
    for vertex_id, delta in zip(vertex_ids, target - source):
        groups.setdefault(int(vertex_id), []).append(delta)
    ids = np.asarray(sorted(groups), dtype=np.int64)
    if not len(ids):
        return ids, np.empty((0, 3), dtype=np.float64)
    delta = np.stack([np.median(np.stack(groups[int(index)]), axis=0) for index in ids])
    cap = float(displacement_cap_ratio) * float(diagonal)
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1., cap / np.maximum(length, 1e-12))[:, None]
    if len(ids) > int(max_handles):
        chosen = farthest_subset(np.asarray(proxy.vertices)[ids], int(max_handles))
        ids, delta = ids[chosen], delta[chosen]
    return ids, np.asarray(proxy.vertices)[ids] + delta


def apply_transform_mesh(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    result.vertices = (np.asarray(result.vertices) @ transform[:3, :3].T
                       + transform[:3, 3])
    return result


def intrinsic_local_step(
    mesh: trimesh.Trimesh,
    body: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    diagonal: float,
    seed: int,
    proxy_triangles: int = 12000,
    correspondence_samples: int = 50000,
    output_samples: int = 100000,
    max_handles: int = 96,
    min_handles: int = 6,
    active_inner_ratio: float = .055,
    active_outer_ratio: float = .125,
    anchor_ratio: float = .145,
    max_handle_displacement_ratio: float = .055,
    max_vertex_displacement_ratio: float = .045,
    max_flipped_face_ratio: float = 8e-4,
    component_policy: str = "line_priority",
    guidance_body: np.ndarray | None = None,
    guidance_max_transfer_ratio: float = .025,
    # Thin or near-contact surfaces can fail a topology check even when a
    # coarse residual direction is correct.  A fixed, shared fine tail lets
    # the same local action take a genuinely small step rather than turning a
    # safe rejection into a global re-generation.  The final acceptance still
    # requires the common saved-view and multi-view no-harm gates.
    continuation=(1., .75, .50, .35, .25, .15, .10, .05, .035, .025, .015, .01, .005),
):
    """Propose and gate one topology-aware visible residual component.

    An optional re-registered generated ``guidance_body`` contributes only
    visible residual vectors.  Handle locations, topology, and all unobserved
    support remain on the original ``mesh``/``body``.
    """
    before = soft_ray_correspondences(
        partial, body, projector, pixel_radius=5., trim_quantile=.75,
        max_distance_ratio=.14, bbox_diagonal=diagonal)
    proxy = build_proxy(mesh, int(proxy_triangles))
    surface, face_ids = sample_surface(proxy, int(correspondence_samples), int(seed))
    component_source = (surface if guidance_body is None else
                        np.asarray(guidance_body, dtype=np.float64))
    components, component_summary = visible_residual_components(
        partial, component_source, projector, diagonal=diagonal)
    if not components:
        return mesh.copy(), np.asarray(body).copy(), {
            **component_summary, "accepted": False,
            "guidance_body_used": guidance_body is not None,
            "before": before, "after": before}
    vertices = np.asarray(proxy.vertices, dtype=np.float64)
    faces = np.asarray(proxy.faces, dtype=np.int64)
    admissible_components = []
    rejected_components = []
    for component_rank, (pair, component) in enumerate(components):
        source, target = pair
        if guidance_body is not None:
            # A generated proposal can have an incorrect hidden part or a
            # shifted thin surface.  Its displacement is informative only
            # where that proposal already lies close to the source mesh; do
            # not create handles by snapping through a gap to another sheet.
            transfer_distance = cKDTree(surface).query(source, k=1, workers=-1)[0]
            transferable = transfer_distance <= float(guidance_max_transfer_ratio) * diagonal
            if int(transferable.sum()) < int(min_handles):
                rejected_components.append({
                    **component, "component_rank": int(component_rank),
                    "reason": "insufficient_nearby_guidance",
                    "transferable_correspondences": int(transferable.sum()),
                    "guidance_max_transfer_ratio": float(guidance_max_transfer_ratio),
                })
                continue
            source, target = source[transferable], target[transferable]
        handles, targets = _aggregate_handles(
            proxy, surface, face_ids, source, target, diagonal=diagonal,
            max_handles=max_handles,
            displacement_cap_ratio=max_handle_displacement_ratio)
        if len(handles) < int(min_handles):
            rejected_components.append({
                **component, "component_rank": int(component_rank),
                "reason": "insufficient_handles", "handles": int(len(handles))})
            continue
        geodesic = mesh_geodesic(vertices, faces, handles)
        support = compact_weight(
            geodesic, float(active_inner_ratio) * diagonal,
            float(active_outer_ratio) * diagonal)
        anchors = np.flatnonzero(geodesic >= float(anchor_ratio) * diagonal)
        extrema = np.unique(np.concatenate((
            np.argmin(vertices, axis=0), np.argmax(vertices, axis=0))))
        anchors = np.unique(np.concatenate((anchors, extrema)))
        anchors = anchors[~np.isin(anchors, handles)]
        if len(anchors) > 2400:
            anchors = anchors[np.linspace(0, len(anchors) - 1, 2400, dtype=np.int64)]
        support_fraction = float(np.mean(support > 0.))
        if len(anchors) < 64 or support_fraction > .34:
            rejected_components.append({
                **component, "component_rank": int(component_rank),
                "reason": "invalid_intrinsic_support", "handles": int(len(handles)),
                "anchors": int(len(anchors)), "support_fraction": support_fraction})
            continue
        priority = float(component["component_fraction"]) * max(
            1., float(component["line_likeness"]))
        admissible_components.append((
            priority, component_rank, component, handles, targets,
            geodesic, support, anchors))
    if not admissible_components:
        return mesh.copy(), np.asarray(body).copy(), {
            **component_summary, "accepted": False,
            "reason": "no_intrinsically_compact_component",
            "guidance_body_used": guidance_body is not None,
            "rejected_components": rejected_components,
            "before": before, "after": before}
    if component_policy == "line_priority":
        selected_component = max(admissible_components, key=lambda item: item[0])
    elif component_policy == "residual_priority":
        # Components arrive in descending visible residual mass order.  This
        # alternative is useful for broad surfaces where line-likeness should
        # not dominate the routing decision.
        selected_component = admissible_components[0]
    else:
        raise ValueError(f"Unknown component policy: {component_policy}")
    (_, component_rank, component, handles, targets,
     geodesic, support, anchors) = selected_component

    displacement, solver = screened_intrinsic_solve(
        proxy, handles, targets, anchors, data_weight=65., smooth_weight=12.,
        anchor_weight=120., identity_weight=.40)
    displacement *= support[:, None]
    displacement, removed = remove_similarity_modes(vertices, displacement, support)
    displacement *= support[:, None]
    cap = float(max_vertex_displacement_ratio) * diagonal
    length = np.linalg.norm(displacement, axis=1)
    displacement *= np.minimum(1., cap / np.maximum(length, 1e-12))[:, None]

    # Transfer by nearest proxy vertex.  Since the proxy is a decimation of the
    # same mesh, this preserves intrinsic component separation better than KNN
    # averaging across nearby thin surfaces.
    _, nearest_proxy = cKDTree(vertices).query(np.asarray(mesh.vertices), k=1, workers=-1)
    high_displacement = displacement[nearest_proxy]
    _, body_proxy = cKDTree(vertices).query(np.asarray(body), k=1, workers=-1)
    body_displacement = displacement[body_proxy]
    candidates = []
    for fraction in continuation:
        candidate_mesh = mesh.copy()
        candidate_mesh.vertices = np.asarray(mesh.vertices) + float(fraction) * high_displacement
        quality = edge_flip_metrics(mesh, np.asarray(candidate_mesh.vertices))
        candidate_body = np.asarray(body) + float(fraction) * body_displacement
        score = soft_ray_correspondences(
            partial, candidate_body, projector, pixel_radius=5.,
            trim_quantile=.75, max_distance_ratio=.14,
            bbox_diagonal=diagonal)
        candidates.append((score["objective"], float(fraction), candidate_mesh,
                           candidate_body, score, quality))
    line_search = [{
        "fraction": float(item[1]),
        "objective": float(item[0]),
        "grid_coverage": float(item[4]["grid_coverage"]),
        "matched_coverage": float(item[4]["matched_coverage"]),
        "edge_stretch_q01": float(item[5]["edge_stretch_q01"]),
        "edge_stretch_q99": float(item[5]["edge_stretch_q99"]),
        "flipped_face_ratio": float(item[5]["flipped_face_ratio"]),
    } for item in candidates]
    topology_safe = [item for item in candidates if (
        item[5]["flipped_face_ratio"] <= float(max_flipped_face_ratio)
        and item[5]["edge_stretch_q01"] >= .78
        and item[5]["edge_stretch_q99"] <= 1.28)]
    selection_pool = topology_safe if topology_safe else candidates
    objective, fraction, candidate_mesh, candidate_body, after, quality = min(
        selection_pool, key=lambda item: item[0])
    accepted = bool(
        np.isfinite(objective)
        # The inner solver only establishes a strictly improving local
        # geometric proposal.  Saved-camera and fixed-frame multi-view
        # no-harm thresholds live in the shared outer controller, so imposing
        # a second arbitrary 0.05% margin here can discard a topology-safe
        # correction before it is evaluated by the real acceptance policy.
        and objective < before["objective"]
        and after["grid_coverage"] >= before["grid_coverage"] - .02
        and after["matched_coverage"] >= before["matched_coverage"] - .015
        and quality["flipped_face_ratio"] <= float(max_flipped_face_ratio)
        and quality["edge_stretch_q01"] >= .78
        and quality["edge_stretch_q99"] <= 1.28)
    return ((candidate_mesh, candidate_body) if accepted else
            (mesh.copy(), np.asarray(body).copy())) + ({
        **component, "accepted": accepted,
        "guidance_body_used": guidance_body is not None,
        "guidance_max_transfer_ratio": (float(guidance_max_transfer_ratio)
                                        if guidance_body is not None else None),
        "reason": "intrinsic_visible_gate_passed" if accepted else "intrinsic_visible_gate_rejected",
        "handles": int(len(handles)), "anchors": int(len(anchors)),
        "component_rank": int(component_rank),
        "component_policy": str(component_policy),
        "rejected_components": rejected_components,
        "support_fraction": float(np.mean(support > 0.)),
        "selected_fraction": float(fraction) if accepted else 0.,
        "topology_safe_candidates": int(len(topology_safe)),
        "line_search": line_search,
        "displacement_p99_ratio": float(np.quantile(
            np.linalg.norm(high_displacement, axis=1), .99) / diagonal),
        "max_flipped_face_ratio": float(max_flipped_face_ratio),
        "similarity_modes_removed": removed, "solver": solver,
        "quality": quality, "before": before, "after": after,
        "mesh_vertices_deleted": 0,
    },)
