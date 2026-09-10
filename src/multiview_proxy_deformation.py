"""Shared broad-to-narrow deformation from several saved partial views.

The module deliberately keeps the deformation small and auditable: every
view contributes pixel-indexed 3-D residuals to one carrier graph, stable
observations become soft anchors, and a single smooth displacement field is
transferred to the textured mesh.  No category, part label, GT, or offline
completion metric is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy import ndimage
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

from src.zbuffer import zbuffer_depth_with_indices


@dataclass(frozen=True)
class OrbitProjector:
    """Perspective projector for a saved OpenGL camera-to-world pose."""

    camera_pose: np.ndarray
    field_of_view_degrees: float
    image_shape: tuple[int, int]

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        pose = np.asarray(self.camera_pose, dtype=np.float64)
        inverse = np.linalg.inv(pose)
        camera = points @ inverse[:3, :3].T + inverse[:3, 3]
        depth = -camera[:, 2]
        height, width = self.image_shape
        focal = 0.5 * height / math.tan(math.radians(self.field_of_view_degrees) * 0.5)
        uv = np.empty((len(points), 2), dtype=np.float64)
        denominator = np.maximum(depth, 1e-12)
        uv[:, 0] = focal * camera[:, 0] / denominator + (width - 1) * 0.5
        uv[:, 1] = -focal * camera[:, 1] / denominator + (height - 1) * 0.5
        return uv, depth


def _visible_pairs(
    prior: np.ndarray,
    partial: np.ndarray,
    projector: OrbitProjector,
    *,
    max_pixel_distance: float,
) -> dict:
    prior_uv, prior_depth = projector.project(prior)
    partial_uv, partial_depth = projector.project(partial)
    _, prior_mask, prior_index = zbuffer_depth_with_indices(
        prior_uv, prior_depth, projector.image_shape, splat_radius=1,
    )
    _, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=1,
    )
    prior_yx = np.argwhere(prior_mask)
    partial_yx = np.argwhere(partial_mask)
    empty = {
        "prior_ids": np.empty(0, dtype=np.int64),
        "partial_ids": np.empty(0, dtype=np.int64),
        "pixel_distance": np.empty(0, dtype=np.float64),
        "residual": np.empty((0, 3), dtype=np.float64),
        "partial_visible_pixels": int(len(partial_yx)),
        "prior_visible_pixels": int(len(prior_yx)),
        "partial_coverage": 0.0,
    }
    if min(len(prior_yx), len(partial_yx)) == 0:
        return empty
    distance, nearest = cKDTree(prior_yx.astype(np.float64)).query(
        partial_yx.astype(np.float64), k=1,
        distance_upper_bound=float(max_pixel_distance), workers=-1,
    )
    valid = np.isfinite(distance) & (nearest < len(prior_yx))
    if not valid.any():
        return empty
    selected_partial_yx = partial_yx[valid]
    selected_prior_yx = prior_yx[nearest[valid]]
    prior_ids = prior_index[selected_prior_yx[:, 0], selected_prior_yx[:, 1]]
    partial_ids = partial_index[selected_partial_yx[:, 0], selected_partial_yx[:, 1]]
    valid_ids = (prior_ids >= 0) & (partial_ids >= 0)
    prior_ids, partial_ids = prior_ids[valid_ids], partial_ids[valid_ids]
    distance = distance[valid][valid_ids]
    return {
        "prior_ids": prior_ids,
        "partial_ids": partial_ids,
        "pixel_distance": distance,
        "residual": partial[partial_ids] - prior[prior_ids],
        "partial_visible_pixels": int(len(partial_yx)),
        "prior_visible_pixels": int(len(prior_yx)),
        "partial_coverage": float(len(prior_ids) / max(len(partial_yx), 1)),
    }


def _silhouette_pairs(
    prior: np.ndarray,
    projector: OrbitProjector,
    target_mask: np.ndarray,
    *,
    max_pixel_distance: float,
) -> dict:
    """Lift edited silhouette motion to the current visible prior surface.

    The edit supplies only an image-plane displacement. Depth remains that of
    the current prior here; physical depth motion is supplied independently by
    the partial 3-D correspondences in :func:`_visible_pairs`.
    """
    target = np.asarray(target_mask, dtype=bool)
    if target.shape != tuple(projector.image_shape):
        raise ValueError("edited silhouette and projector image_shape must match")
    prior_uv, prior_depth = projector.project(prior)
    _, prior_mask, prior_index = zbuffer_depth_with_indices(
        prior_uv, prior_depth, projector.image_shape, splat_radius=1,
    )
    structure = np.ones((3, 3), dtype=bool)
    prior_boundary = prior_mask & ~ndimage.binary_erosion(prior_mask, structure=structure)
    target_boundary = target & ~ndimage.binary_erosion(target, structure=structure)
    source_yx = np.argwhere(prior_boundary)
    target_yx = np.argwhere(target_boundary)
    empty = {
        "prior_ids": np.empty(0, dtype=np.int64),
        "residual": np.empty((0, 3), dtype=np.float64),
        "pixel_distance": np.empty(0, dtype=np.float64),
    }
    if min(len(source_yx), len(target_yx)) == 0:
        return empty

    source_tree = cKDTree(source_yx.astype(np.float64))
    target_tree = cKDTree(target_yx.astype(np.float64))
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # Expansion evidence: a target contour absent from the current render is
    # attached to its nearest visible prior contour.
    expansion_yx = target_yx[~prior_mask[target_yx[:, 0], target_yx[:, 1]]]
    if len(expansion_yx):
        distance, nearest = source_tree.query(expansion_yx, k=1, workers=-1)
        keep = distance <= float(max_pixel_distance)
        if keep.any():
            records.append((source_yx[nearest[keep]], expansion_yx[keep], distance[keep]))

    # Contraction evidence: a rendered contour outside the edited silhouette
    # is moved toward the nearest edited contour.
    contraction_yx = source_yx[~target[source_yx[:, 0], source_yx[:, 1]]]
    if len(contraction_yx):
        distance, nearest = target_tree.query(contraction_yx, k=1, workers=-1)
        keep = distance <= float(max_pixel_distance)
        if keep.any():
            records.append((contraction_yx[keep], target_yx[nearest[keep]], distance[keep]))
    if not records:
        return empty

    source_pixels = np.concatenate([item[0] for item in records], axis=0)
    target_pixels = np.concatenate([item[1] for item in records], axis=0)
    pixel_distance = np.concatenate([item[2] for item in records], axis=0)
    prior_ids = prior_index[source_pixels[:, 0], source_pixels[:, 1]]
    valid = (prior_ids >= 0) & (prior_depth[prior_ids] > 0.0)
    prior_ids = prior_ids[valid]
    target_pixels = target_pixels[valid]
    pixel_distance = pixel_distance[valid]
    if not len(prior_ids):
        return empty

    target_uv = target_pixels[:, ::-1].astype(np.float64)
    source_uv = prior_uv[prior_ids]
    depth = prior_depth[prior_ids]
    height, _ = projector.image_shape
    focal = 0.5 * height / math.tan(math.radians(projector.field_of_view_degrees) * 0.5)
    camera_delta = np.column_stack((
        (target_uv[:, 0] - source_uv[:, 0]) * depth / focal,
        -(target_uv[:, 1] - source_uv[:, 1]) * depth / focal,
        np.zeros(len(prior_ids), dtype=np.float64),
    ))
    pose = np.asarray(projector.camera_pose, dtype=np.float64)
    return {
        "prior_ids": prior_ids,
        "residual": camera_delta @ pose[:3, :3].T,
        "pixel_distance": pixel_distance,
    }
def multiview_consistency(
    prior: np.ndarray,
    partial: np.ndarray,
    projectors: list[OrbitProjector],
    *,
    max_pixel_distance: float = 5.0,
) -> dict:
    """Return one-sided visible consistency; complete hidden support is free."""
    records = []
    all_residuals = []
    for projector in projectors:
        pairs = _visible_pairs(
            prior, partial, projector, max_pixel_distance=max_pixel_distance,
        )
        norm = np.linalg.norm(pairs["residual"], axis=1)
        all_residuals.append(norm)
        records.append({
            "partial_visible_pixels": pairs["partial_visible_pixels"],
            "prior_visible_pixels": pairs["prior_visible_pixels"],
            "matched_pixels": int(len(norm)),
            "partial_coverage": pairs["partial_coverage"],
            "surface_residual_median": float(np.median(norm)) if len(norm) else None,
            "surface_residual_p90": float(np.quantile(norm, .90)) if len(norm) else None,
        })
    concatenated = np.concatenate([item for item in all_residuals if len(item)], axis=0)
    return {
        "views": records,
        "mean_partial_coverage": float(np.mean([item["partial_coverage"] for item in records])),
        "surface_residual_median": float(np.median(concatenated)) if len(concatenated) else None,
        "surface_residual_p90": float(np.quantile(concatenated, .90)) if len(concatenated) else None,
    }


def _voxel_anchors(points: np.ndarray, target_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-9)
    low, high = diagonal / 180.0, diagonal / 6.0
    best = np.arange(min(len(points), target_count), dtype=np.int64)
    for _ in range(18):
        size = 0.5 * (low + high)
        keys = np.floor((points - points.min(axis=0)) / size).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        best = np.sort(indices)
        if len(best) > target_count * 1.15:
            low = size
        elif len(best) < target_count * .70:
            high = size
        else:
            break
    return best


def _anchor_graph(points: np.ndarray, neighbours: int = 12) -> sparse.csr_matrix:
    count = len(points)
    k = min(max(3, int(neighbours)) + 1, count)
    distance, index = cKDTree(points).query(points, k=k, workers=-1)
    row = np.repeat(np.arange(count, dtype=np.int64), k - 1)
    col = index[:, 1:].reshape(-1)
    values = np.exp(-np.square(distance[:, 1:].reshape(-1) /
                               max(float(np.median(distance[:, 1:])), 1e-12)))
    adjacency = sparse.coo_matrix((values, (row, col)), shape=(count, count)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    return sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency


def _interpolate_field(query: np.ndarray, anchors: np.ndarray, field: np.ndarray, k: int = 4) -> np.ndarray:
    k = min(int(k), len(anchors))
    distance, index = cKDTree(anchors).query(query, k=k, workers=-1)
    if k == 1:
        return field[index]
    weights = 1.0 / np.maximum(distance, 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(field[index] * weights[..., None], axis=1)


def _pixel_distance_schedule(
    fine_distance: float,
    coarse_distance: float,
    coarse_iterations: int,
) -> tuple[float, ...]:
    """Return a geometric broad-to-narrow correspondence schedule.

    Distances are expressed in pixels by the caller, but the coarse endpoint
    is derived from the image size.  This makes the capture range independent
    of object category and physical scale.  A geometric decay gives one truly
    broad structural step followed by an intermediate step before the normal
    local correspondence radius is restored.
    """
    fine = float(fine_distance)
    coarse = max(float(coarse_distance), fine)
    count = max(int(coarse_iterations), 0)
    if count == 0:
        return ()
    if count == 1:
        return (coarse,)
    return tuple(float(value) for value in np.geomspace(coarse, fine, count + 1)[:-1])


def _solve_iteration(
    carrier: np.ndarray,
    partial: np.ndarray,
    projectors: list[OrbitProjector],
    *,
    anchor_count: int,
    max_pixel_distance: float,
    smoothness_weight: float,
    data_weight: float,
    stable_weight: float,
    edited_silhouettes: list[np.ndarray] | None,
    silhouette_weight: float,
    max_silhouette_pixel_distance: float,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    anchor_ids = _voxel_anchors(carrier, int(anchor_count))
    anchors = carrier[anchor_ids]
    anchor_tree = cKDTree(anchors)
    observations: list[tuple[int, np.ndarray, float, bool]] = []
    per_view = []
    all_norm = []
    raw_pairs = []
    for view_index, projector in enumerate(projectors):
        pairs = _visible_pairs(
            carrier, partial, projector, max_pixel_distance=max_pixel_distance,
        )
        norm = np.linalg.norm(pairs["residual"], axis=1)
        raw_pairs.append((view_index, pairs, norm))
        all_norm.append(norm)
    valid_norm = np.concatenate([item for item in all_norm if len(item)], axis=0)
    if len(valid_norm) < 32:
        raise ValueError("insufficient multiview visible correspondences")
    spacing = float(np.median(cKDTree(partial).query(partial, k=2, workers=-1)[0][:, 1]))
    stable_threshold = max(2.5 * spacing, float(np.quantile(valid_norm, .25)))
    high_threshold = min(float(np.quantile(valid_norm, .985)), .35 * diagonal)
    for view_index, pairs, norm in raw_pairs:
        keep = np.isfinite(norm) & (norm <= high_threshold)
        prior_ids = pairs["prior_ids"][keep]
        residual = pairs["residual"][keep]
        pixel_distance = pairs["pixel_distance"][keep]
        anchor = anchor_tree.query(carrier[prior_ids], k=1, workers=-1)[1]
        confidence = np.exp(-.5 * np.square(pixel_distance / max(max_pixel_distance * .55, 1e-6)))
        for aid, vector, weight, is_stable in zip(anchor, residual, confidence, norm[keep] <= stable_threshold):
            observations.append((int(aid), vector, float(weight), bool(is_stable)))
        silhouette_controls = 0
        if edited_silhouettes is not None:
            silhouette = _silhouette_pairs(
                carrier, projectors[view_index], edited_silhouettes[view_index],
                max_pixel_distance=max_silhouette_pixel_distance,
            )
            if len(silhouette["prior_ids"]):
                semantic_anchor = anchor_tree.query(
                    carrier[silhouette["prior_ids"]], k=1, workers=-1,
                )[1]
                semantic_confidence = float(silhouette_weight) * np.exp(
                    -.5 * np.square(
                        silhouette["pixel_distance"]
                        / max(max_silhouette_pixel_distance * .55, 1e-6)
                    )
                )
                for aid, vector, weight in zip(
                    semantic_anchor, silhouette["residual"], semantic_confidence,
                ):
                    observations.append((int(aid), vector, float(weight), False))
                silhouette_controls = int(len(semantic_anchor))
        per_view.append({
            "view_index": int(view_index),
            "visible_pairs": int(len(norm)),
            "retained_pairs": int(keep.sum()),
            "partial_coverage": pairs["partial_coverage"],
            "residual_median": float(np.median(norm)) if len(norm) else None,
            "edited_silhouette_controls": silhouette_controls,
        })
    grouped: dict[int, list[tuple[np.ndarray, float, bool]]] = {}
    for aid, vector, weight, stable in observations:
        grouped.setdefault(aid, []).append((vector, weight, stable))
    control_ids, targets, confidence, stable_ids = [], [], [], []
    for aid, group in grouped.items():
        vectors = np.stack([item[0] for item in group])
        weights = np.asarray([item[1] for item in group], dtype=np.float64)
        stable_fraction = float(np.mean([item[2] for item in group]))
        target = np.median(vectors, axis=0)
        support = min(len(group) / 4.0, 1.0)
        if stable_fraction >= .55:
            stable_ids.append(aid)
        else:
            control_ids.append(aid)
            targets.append(target)
            confidence.append(float(np.mean(weights)) * support)
    control_ids = np.asarray(control_ids, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    confidence = np.asarray(confidence, dtype=np.float64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    if len(control_ids) < 8:
        raise ValueError("insufficient non-stable multiview controls")
    laplacian = _anchor_graph(anchors)
    diagonal_data = np.full(len(anchors), 1e-5, dtype=np.float64)
    rhs = np.zeros((len(anchors), 3), dtype=np.float64)
    np.add.at(diagonal_data, control_ids, float(data_weight) * confidence)
    np.add.at(rhs, control_ids, float(data_weight) * confidence[:, None] * targets)
    if len(stable_ids):
        np.add.at(diagonal_data, stable_ids, float(stable_weight))
    system = float(smoothness_weight) * laplacian + sparse.diags(diagonal_data)
    field = np.asarray(sparse_linalg.spsolve(system.tocsc(), rhs), dtype=np.float64)
    carrier_field = _interpolate_field(carrier, anchors, field)
    status = np.zeros(len(carrier), dtype=np.uint8)
    nearest_anchor = anchor_tree.query(carrier, k=1, workers=-1)[1]
    status[np.isin(nearest_anchor, control_ids)] = 1
    status[np.isin(nearest_anchor, stable_ids)] = 2
    return carrier_field, {
        "anchor_count": int(len(anchors)),
        "control_anchors": int(len(control_ids)),
        "stable_anchors": int(len(stable_ids)),
        "stable_threshold": stable_threshold,
        "high_residual_threshold": high_threshold,
        "partial_spacing": spacing,
        "per_view": per_view,
    }, status, {
        "anchors": anchors,
        "anchor_field": field,
        "stable_anchor_ids": stable_ids,
    }


def _mesh_laplacian(vertices: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
    """Build the fixed, seam-aware surface operator used by every iteration."""
    from src.mesh_attached_gaussian_deformation import mesh_surface_graph

    graph = mesh_surface_graph(vertices, faces)
    positive = graph.data[graph.data > np.finfo(np.float64).eps]
    scale = max(float(np.median(positive)), 1e-12)
    adjacency = graph.copy().astype(np.float64)
    adjacency.data = np.exp(-adjacency.data / scale)
    return (sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()


def _mesh_harmonic_transfer(
    vertices: np.ndarray,
    mesh_laplacian: sparse.csr_matrix,
    anchors: np.ndarray,
    anchor_field: np.ndarray,
    stable_anchor_ids: np.ndarray,
) -> np.ndarray:
    """Transfer the carrier field through mesh connectivity, not spatial KNN.

    Spatial interpolation can blend opposite sides of a thin surface and flip
    triangles.  Here carrier anchors are attached to their nearest mesh
    vertices and extended by one screened Laplacian solve on the welded mesh
    graph, so nearby-but-disconnected sheets remain independent.
    """
    mesh_ids = cKDTree(vertices).query(anchors, k=1, workers=-1)[1]
    unique, inverse = np.unique(mesh_ids, return_inverse=True)
    target = np.zeros((len(unique), 3), dtype=np.float64)
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(target, inverse, anchor_field)
    target /= count[:, None]
    diagonal = np.full(len(vertices), 1e-6, dtype=np.float64)
    rhs = np.zeros((len(vertices), 3), dtype=np.float64)
    attachment_weight = 8.0
    np.add.at(diagonal, unique, attachment_weight)
    np.add.at(rhs, unique, attachment_weight * target)
    if len(stable_anchor_ids):
        stable_mesh = np.unique(mesh_ids[np.asarray(stable_anchor_ids, dtype=np.int64)])
        np.add.at(diagonal, stable_mesh, 24.0)
    system = (mesh_laplacian + sparse.diags(diagonal)).tocsr()
    result = np.zeros((len(vertices), 3), dtype=np.float64)
    for axis in range(3):
        solved, info = sparse_linalg.cg(system, rhs[:, axis], rtol=2e-5, atol=0.0, maxiter=600)
        if info != 0:
            # Coarse structural controls can make the screened system more
            # ill-conditioned than the local stage. A sparse direct solve is
            # an exact deterministic fallback for the same objective rather
            # than a change in geometry or acceptance policy.
            solved = sparse_linalg.spsolve(system.tocsc(), rhs[:, axis])
            if not np.isfinite(solved).all():
                raise RuntimeError(
                    f"mesh harmonic transfer failed to converge on axis {axis}: {info}"
                )
        result[:, axis] = solved
    return result


def deform_shared_multiview(
    carrier: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    partial: np.ndarray,
    projectors: list[OrbitProjector],
    *,
    iterations: int = 2,
    anchor_count: int = 6000,
    max_pixel_distance: float = 5.0,
    smoothness_weight: float = 1.0,
    data_weight: float = 6.0,
    stable_weight: float = 18.0,
    maximum_local_log_scale: float = .35,
    edited_silhouettes: list[np.ndarray] | None = None,
    silhouette_weight: float = .45,
    max_silhouette_pixel_distance: float = 36.0,
    coarse_pixel_distance_ratio: float = 0.0,
    coarse_iterations: int = 0,
    coarse_anchor_ratio: float = .25,
    coarse_smoothness_multiplier: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Deform one carrier/mesh with bounded local scale and fixed connectivity."""
    from src.mesh_attached_gaussian_deformation import _surface_orientation_quality

    carrier = np.asarray(carrier, dtype=np.float64).copy()
    mesh = np.asarray(mesh_vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh_faces, dtype=np.int64)
    initial_mesh = mesh.copy()
    mesh_laplacian = _mesh_laplacian(initial_mesh, faces)
    minimum_local_scale = float(np.exp(-maximum_local_log_scale))
    maximum_local_scale = float(np.exp(maximum_local_log_scale))
    if edited_silhouettes is not None and len(edited_silhouettes) != len(projectors):
        raise ValueError("one edited silhouette is required for every projector")
    before = multiview_consistency(carrier, partial, projectors,
                                   max_pixel_distance=max_pixel_distance)
    image_extent = min(projectors[0].image_shape)
    coarse_schedule = _pixel_distance_schedule(
        max_pixel_distance,
        float(coarse_pixel_distance_ratio) * float(image_extent),
        coarse_iterations,
    )
    records = []
    final_status = np.zeros(len(carrier), dtype=np.uint8)
    for iteration in range(int(iterations)):
        coarse = iteration < len(coarse_schedule)
        iteration_pixel_distance = (
            coarse_schedule[iteration] if coarse else float(max_pixel_distance)
        )
        iteration_anchor_count = (
            max(256, int(round(anchor_count * float(coarse_anchor_ratio))))
            if coarse else int(anchor_count)
        )
        iteration_smoothness = float(smoothness_weight) * (
            float(coarse_smoothness_multiplier) if coarse else 1.0
        )
        field, record, status, control_state = _solve_iteration(
            carrier, partial, projectors, anchor_count=iteration_anchor_count,
            max_pixel_distance=iteration_pixel_distance,
            smoothness_weight=iteration_smoothness, data_weight=data_weight,
            stable_weight=stable_weight,
            edited_silhouettes=edited_silhouettes,
            silhouette_weight=silhouette_weight,
            max_silhouette_pixel_distance=max(
                float(max_silhouette_pixel_distance), iteration_pixel_distance,
            ),
        )
        mesh_field = _mesh_harmonic_transfer(
            mesh, mesh_laplacian, control_state["anchors"], control_state["anchor_field"],
            control_state["stable_anchor_ids"],
        )
        accepted = None
        candidates = []
        baseline = multiview_consistency(
            carrier, partial, projectors,
            max_pixel_distance=iteration_pixel_distance,
        )
        # Large residuals are absorbed over several topology-safe increments.
        # The small tail is important for thin structures: it lets deformation
        # accumulate without accepting a locally inverted surface in any step.
        for alpha in (1.0, .75, .5, .35, .25, .15, .10, .075,
                      .05, .04, .03, .025, .02, .015, .01):
            candidate_carrier = carrier + float(alpha) * field
            candidate_mesh = mesh + float(alpha) * mesh_field
            step_quality = _surface_orientation_quality(mesh, candidate_mesh, faces)
            total_quality = _surface_orientation_quality(initial_mesh, candidate_mesh, faces)
            score = multiview_consistency(
                candidate_carrier, partial, projectors,
                max_pixel_distance=iteration_pixel_distance,
            )
            residual_ok = (score["surface_residual_median"] is not None and
                           score["surface_residual_median"] < baseline["surface_residual_median"])
            topology_ok = (
                step_quality["flipped_resolved_triangle_fraction"] <= 1e-5
                and total_quality["edge_stretch_p01"] >= minimum_local_scale
                and total_quality["edge_stretch_p99"] <= maximum_local_scale
            )
            candidates.append({
                "alpha": float(alpha),
                "surface_residual_median": score["surface_residual_median"],
                "surface_residual_p90": score["surface_residual_p90"],
                "mean_partial_coverage": score["mean_partial_coverage"],
                "residual_ok": bool(residual_ok), "topology_ok": bool(topology_ok),
                "flipped_triangle_fraction": step_quality["flipped_resolved_triangle_fraction"],
                "initial_normal_reversal_fraction": total_quality["flipped_resolved_triangle_fraction"],
                "step_edge_stretch_p01": step_quality["edge_stretch_p01"],
                "step_edge_stretch_p99": step_quality["edge_stretch_p99"],
                "total_edge_stretch_p01": total_quality["edge_stretch_p01"],
                "total_edge_stretch_p50": total_quality["edge_stretch_p50"],
                "total_edge_stretch_p99": total_quality["edge_stretch_p99"],
            })
            if residual_ok and topology_ok:
                accepted = alpha, candidate_carrier, candidate_mesh, step_quality, total_quality, score
                break
        if accepted is None:
            records.append({"iteration": iteration, "active": False,
                            "line_search_candidates": candidates, **record})
            break
        alpha, carrier, mesh, step_quality, total_quality, score = accepted
        records.append({
            "iteration": iteration, "active": True, "alpha": float(alpha),
            "phase": "coarse_structural" if coarse else "fine_residual",
            "max_pixel_distance": float(iteration_pixel_distance),
            "smoothness_weight": float(iteration_smoothness),
            "consistency": score,
            "step_surface_quality": step_quality,
            "total_surface_quality": total_quality,
            "line_search_candidates": candidates, **record,
        })
        final_status = status
    after = multiview_consistency(carrier, partial, projectors,
                                  max_pixel_distance=max_pixel_distance)
    total_quality = _surface_orientation_quality(initial_mesh, mesh, faces)
    return carrier, mesh, final_status, {
        "method": "shared_multiview_low_frequency_proxy_deformation",
        "ground_truth_used": False,
        "category_or_part_rules_used": False,
        "local_scale_allowed": True,
        "local_scale_bounds": [minimum_local_scale, maximum_local_scale],
        "edited_silhouette_guidance": edited_silhouettes is not None,
        "edited_silhouette_weight": float(silhouette_weight),
        "coarse_pixel_distance_ratio": float(coarse_pixel_distance_ratio),
        "coarse_pixel_distance_schedule": list(coarse_schedule),
        "coarse_iterations": int(coarse_iterations),
        "before": before,
        "after": after,
        "iterations": records,
        "total_surface_quality": total_quality,
    }
