"""Joint object/camera refinement using NVIDIA's differentiable rasterizer.

This module deliberately does not implement rasterization.  ``nvdiffrast``
provides triangle coverage and analytic antialiasing gradients; PyTorch3D
provides the SO(3) exponential/logarithm maps.  GenPC++ only contributes the
registration model: one shared residual Sim(3), small auxiliary-camera orbit
corrections, and a partial-supported 3-D anchor term.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from pytorch3d.transforms import matrix_to_axis_angle, so3_exp_map

from src.bidirectional_cycle_registration import sim3_parts
from src.ray_consistent_registration import apply_transform
from src.trellis_condition_registration import ConditionCamera, _mask_moments


@dataclass(frozen=True)
class RasterRefinementConfig:
    """Category-independent trust region for one refinement level."""

    resolution: int
    steps: int
    learning_rate: float
    max_object_rotation_degrees: float
    max_log_scale: float
    max_translation_ratio: float
    max_auxiliary_rotation_degrees: float
    stable_fraction: float = 0.75
    stable_samples: int = 4096
    partial_weight: float = 0.35
    camera1_rgb_weight: float = 0.0


@dataclass(frozen=True)
class RasterRefinementResult:
    transform: np.ndarray
    auxiliary_rotation_vectors: dict[str, np.ndarray]
    diagnostics: dict


def capture_shared_multiview_translation(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    transform: np.ndarray,
    cameras: list[ConditionCamera],
    field_of_view_degrees: float,
    auxiliary_rotation_vectors: dict[str, np.ndarray],
    *,
    rounds: int = 2,
    max_translation_ratio: float = 0.10,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """Solve one shared 3-D translation from multi-view mask centroids.

    The renderer supplies a finite-difference projection Jacobian, so this
    remains valid for arbitrary calibrated view layouts.  A short line search
    accepts only an improvement of the same no-GT centroid/IoU objective.
    Rotation, scale, geometry, and auxiliary camera orbits stay fixed.
    """
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    faces = np.asarray(mesh_faces, dtype=np.int32)
    current = np.asarray(transform, dtype=np.float64).copy()
    world = apply_transform(vertices, current)
    diagonal = max(float(np.linalg.norm(np.ptp(world, axis=0))), 1e-8)
    epsilon = 0.005 * diagonal
    target_centres = np.stack([_mask_moments(camera.mask)[0] for camera in cameras])

    def evaluate(candidate: np.ndarray) -> tuple[float, np.ndarray, list[dict]]:
        masks = render_multiview_masks(
            vertices, faces, candidate, cameras, field_of_view_degrees,
            auxiliary_rotation_vectors, device=device,
        )
        predicted_centres, records = [], []
        for camera in cameras:
            predicted = np.asarray(masks[camera.name], dtype=bool)
            target = np.asarray(camera.mask, dtype=bool)
            predicted_centre = _mask_moments(predicted)[0]
            predicted_centres.append(predicted_centre)
            intersection = int(np.count_nonzero(predicted & target))
            union = int(np.count_nonzero(predicted | target))
            records.append({
                "name": camera.name,
                "iou": float(intersection / max(union, 1)),
                "target_centre": _mask_moments(target)[0].tolist(),
                "predicted_centre": predicted_centre.tolist(),
            })
        predicted_centres = np.stack(predicted_centres)
        centre_error = np.linalg.norm(target_centres - predicted_centres, axis=1)
        ious = np.asarray([item["iou"] for item in records])
        objective = (
            float(centre_error.mean())
            + 0.25 * float(centre_error.max())
            + 0.10 * float(1.0 - ious.mean())
        )
        return objective, predicted_centres, records

    history = []
    for round_index in range(int(rounds)):
        before, predicted_centres, before_records = evaluate(current)
        jacobian = np.empty((2 * len(cameras), 3), dtype=np.float64)
        for axis in range(3):
            perturbed = current.copy()
            perturbed[axis, 3] += epsilon
            _, shifted_centres, _ = evaluate(perturbed)
            jacobian[:, axis] = ((shifted_centres - predicted_centres) / epsilon).reshape(-1)
        residual = (target_centres - predicted_centres).reshape(-1)
        regularizer = 1e-4 * np.eye(3, dtype=np.float64)
        delta = np.linalg.solve(jacobian.T @ jacobian + regularizer, jacobian.T @ residual)
        length = float(np.linalg.norm(delta))
        limit = float(max_translation_ratio) * diagonal
        if length > limit:
            delta *= limit / length
        candidates = []
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
            candidate = current.copy()
            candidate[:3, 3] += float(fraction) * delta
            objective, _, records = evaluate(candidate)
            candidates.append((objective, float(fraction), candidate, records))
        selected = min(candidates, key=lambda item: item[0])
        history.append({
            "round": round_index,
            "objective_before": before,
            "objective_after": selected[0],
            "jacobian": jacobian.tolist(),
            "solved_translation": delta.tolist(),
            "selected_fraction": selected[1],
            "per_view_before": before_records,
            "per_view_after": selected[3],
        })
        current = selected[2]
        if selected[1] == 0.0:
            break
    return current, {
        "method": "finite_difference_multiview_centroid_translation",
        "ground_truth_used": False,
        "rounds": history,
        "transform": current.tolist(),
    }


def initialise_from_condition_camera(
    source_points: np.ndarray,
    selection_json: Path,
    render_manifest: dict,
) -> tuple[np.ndarray, dict]:
    """Analytically map a selected TRELLIS view to the condition front view."""
    selection = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    proposal = selection.get("selected")
    if proposal is None:
        proposal = selection["top"][0]
    front = next(item for item in render_manifest["views"] if item["name"] == "front")
    source_basis = np.asarray(proposal["camera_pose"], dtype=np.float64)[:3, :3]
    target_basis = np.asarray(front["camera_pose"], dtype=np.float64)[:3, :3]
    rotation = target_basis @ source_basis.T
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    source = np.asarray(source_points, dtype=np.float64)
    source_centre = np.median(source, axis=0)
    source_radius = float(np.quantile(np.linalg.norm(source - source_centre, axis=1), 0.995))
    target_centre = np.asarray(render_manifest["centre"], dtype=np.float64)
    target_radius = float(render_manifest["radius"])
    scale = target_radius / max(source_radius, 1e-12)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_centre - scale * rotation @ source_centre
    return transform, {
        "method": "analytic_selected_view_to_condition_front",
        "selection_json": str(Path(selection_json).resolve()),
        "source_radius": source_radius,
        "target_radius": target_radius,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": transform[:3, 3].tolist(),
        "transform": transform.tolist(),
    }


def initialise_from_multiview_condition_cameras(
    source_points: np.ndarray,
    selection_jsons: dict[str, Path],
    render_manifest: dict,
    *,
    shortlist: int | None = None,
    consistency_scale_degrees: float = 30.0,
    beam_width: int = 8,
    max_beam_states: int = 4096,
) -> tuple[np.ndarray, dict]:
    """Resolve canonical-frame ambiguity with one camera-consistent orbit.

    Independent image retrieval is ambiguous for repeated or nearly symmetric
    structures. Search the complete discrete camera bank jointly instead: a
    valid tuple must induce nearly the same source-to-target object rotation
    from every condition view. A small beam avoids a Cartesian-product search.
    This remains a no-GT discrete basin-selection step; continuous pose and
    scale are left to the downstream geometric optimiser.
    """
    names = tuple(selection_jsons)
    manifest_views = {item["name"]: item for item in render_manifest["views"]}
    if len(names) < 2 or any(name not in manifest_views for name in names):
        raise ValueError("multi-view initialization needs named manifest views")
    records: dict[str, list[dict]] = {}
    rotations: dict[str, np.ndarray] = {}
    visual_regrets: dict[str, np.ndarray] = {}
    for name, path in selection_jsons.items():
        selection = json.loads(Path(path).read_text(encoding="utf-8"))
        candidates = selection.get("candidates")
        if candidates is None:
            candidates = selection.get("top")
        if candidates is None:
            candidates = [selection["selected"]]
        if shortlist is not None:
            candidates = sorted(
                candidates, key=lambda item: float(item["joint_score"]), reverse=True,
            )[: int(shortlist)]
        records[name] = candidates
        scores = np.asarray([item["joint_score"] for item in candidates], dtype=np.float64)
        low = float(np.quantile(scores, 0.05))
        high = float(scores.max())
        visual_regrets[name] = np.clip(
            (high - scores) / max(high - low, 1e-8), 0.0, 1.5,
        )
        induced = []
        for candidate in candidates:
            source_basis = np.asarray(candidate["camera_pose"], dtype=np.float64)[:3, :3]
            target_basis = np.asarray(
                manifest_views[name]["camera_pose"], dtype=np.float64,
            )[:3, :3]
            rotation = target_basis @ source_basis.T
            u, _, vh = np.linalg.svd(rotation)
            rotation = u @ vh
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vh
            induced.append(rotation)
        rotations[name] = np.stack(induced)

    def rotation_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        relative = left @ np.swapaxes(right, -1, -2)
        traces = np.trace(relative, axis1=-2, axis2=-1)
        return np.degrees(np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)))

    first = names[0]
    states = [
        ([index], float(regret), 0.0)
        for index, regret in enumerate(visual_regrets[first])
    ]
    for view_index, name in enumerate(names[1:], start=1):
        expanded = []
        for indices, visual_sum, pairwise_sum in states:
            distances = np.zeros(len(records[name]), dtype=np.float64)
            for previous_index, previous_name in zip(indices, names[:view_index]):
                distances += rotation_distances(
                    rotations[name], rotations[previous_name][previous_index],
                )
            local = visual_regrets[name] + distances / float(consistency_scale_degrees)
            keep = np.argsort(local)[: min(int(beam_width), len(local))]
            for candidate_index in keep:
                expanded.append((
                    [*indices, int(candidate_index)],
                    visual_sum + float(visual_regrets[name][candidate_index]),
                    pairwise_sum + float(distances[candidate_index]),
                ))
        pair_count = view_index * (view_index + 1) / 2
        expanded.sort(key=lambda state: (
            state[1] / (view_index + 1)
            + (state[2] / max(pair_count, 1.0)) / float(consistency_scale_degrees)
        ))
        states = expanded[: min(int(max_beam_states), len(expanded))]

    pair_count = len(names) * (len(names) - 1) / 2
    indices, visual_sum, pairwise_sum = min(
        states,
        key=lambda state: (
            state[1] / len(names)
            + (state[2] / max(pair_count, 1.0)) / float(consistency_scale_degrees)
        ),
    )
    selected_candidates = [records[name][index] for name, index in zip(names, indices)]
    selected_rotations = [rotations[name][index] for name, index in zip(names, indices)]
    pairwise_degrees = [
        float(rotation_distances(selected_rotations[i], selected_rotations[j]))
        for i in range(len(selected_rotations))
        for j in range(i + 1, len(selected_rotations))
    ]
    consistency_degrees = float(np.mean(pairwise_degrees)) if pairwise_degrees else 0.0
    image_regret = float(visual_sum / len(names))
    objective = image_regret + consistency_degrees / float(consistency_scale_degrees)

    # The first/front view is the gauge that defines the exported object
    # frame. Differences in generated auxiliary views are initialized as
    # camera residuals instead of rotating the object away from Camera-1.
    rotation = selected_rotations[0]
    auxiliary_vectors = {names[0]: np.zeros(3, dtype=np.float64)}
    for name, view_rotation in zip(names[1:], selected_rotations[1:]):
        orbit = view_rotation @ rotation.T
        auxiliary_vectors[name] = Rotation.from_matrix(orbit).as_rotvec()

    source = np.asarray(source_points, dtype=np.float64)
    source_centre = np.median(source, axis=0)
    source_radius = float(np.quantile(np.linalg.norm(source - source_centre, axis=1), 0.995))
    target_centre = np.asarray(render_manifest["centre"], dtype=np.float64)
    target_radius = float(render_manifest["radius"])
    scale = target_radius / max(source_radius, 1e-12)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_centre - scale * rotation @ source_centre
    return transform, {
        "method": "full_bank_camera_orbit_consistent_rotation",
        "selection_jsons": {
            name: str(Path(path).resolve()) for name, path in selection_jsons.items()
        },
        "candidate_count_per_view": {name: len(records[name]) for name in names},
        "shortlist": None if shortlist is None else int(shortlist),
        "beam_width": int(beam_width),
        "max_beam_states": int(max_beam_states),
        "consistency_scale_degrees": float(consistency_scale_degrees),
        "selected_objective": objective,
        "selected_image_regret": image_regret,
        "selected_consistency_degrees": consistency_degrees,
        "selected_pairwise_degrees": pairwise_degrees,
        "selected_views": {
            name: {
                key: candidate[key]
                for key in ("yaw_degrees", "pitch_degrees", "roll_degrees", "joint_score")
            }
            for name, candidate in zip(names, selected_candidates)
        },
        "initial_auxiliary_rotation_vectors": {
            name: vector.tolist() for name, vector in auxiliary_vectors.items()
        },
        "source_radius": source_radius,
        "target_radius": target_radius,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": transform[:3, 3].tolist(),
        "transform": transform.tolist(),
    }


def render_multiview_masks(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    transform: np.ndarray,
    cameras: list[ConditionCamera],
    field_of_view_degrees: float,
    auxiliary_rotation_vectors: dict[str, np.ndarray],
    *,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Rasterize final hard masks for diagnostics, using the fitted cameras."""
    vertices_world = apply_transform(np.asarray(mesh_vertices), transform)
    # Camera orbit calibration is defined around the rendered mesh centre.
    pivot = np.median(vertices_world, axis=0)
    vertices = torch.as_tensor(vertices_world, dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(mesh_faces), dtype=torch.int32, device=device)
    pivot_t = torch.as_tensor(pivot, dtype=torch.float32, device=device)
    context = dr.RasterizeCudaContext(device=device)
    rendered = {}
    with torch.no_grad():
        for camera in cameras:
            vector = torch.as_tensor(
                auxiliary_rotation_vectors.get(camera.name, np.zeros(3)),
                dtype=torch.float32, device=device,
            )
            orbit = so3_exp_map(vector[None])[0]
            view_vertices = (vertices - pivot_t) @ orbit.T + pivot_t
            mask = rasterize_silhouette(
                context, view_vertices, faces, camera,
                field_of_view_degrees, int(camera.mask.shape[0]),
            )
            rendered[camera.name] = (mask >= 0.5).cpu().numpy()
    return rendered


def compose_pivoted_sim3(
    base_transform: np.ndarray,
    pivot: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    """Left-compose a residual Sim(3) around a world-space pivot."""
    linear = float(scale) * np.asarray(rotation, dtype=np.float64)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = linear
    delta[:3, 3] = (
        np.asarray(pivot, dtype=np.float64)
        + np.asarray(translation, dtype=np.float64)
        - linear @ np.asarray(pivot, dtype=np.float64)
    )
    return delta @ np.asarray(base_transform, dtype=np.float64)


def _project_clip(
    world_vertices: torch.Tensor,
    camera: ConditionCamera,
    field_of_view_degrees: float,
) -> torch.Tensor:
    """Convert world vertices to nvdiffrast clip coordinates."""
    dtype, device = world_vertices.dtype, world_vertices.device
    pose = torch.as_tensor(camera.pose, dtype=dtype, device=device)
    camera_vertices = (world_vertices - pose[:3, 3]) @ pose[:3, :3]
    depth = -camera_vertices[:, 2]
    tangent = math.tan(math.radians(float(field_of_view_degrees)) * 0.5)
    offset = torch.as_tensor(camera.image_offset, dtype=dtype, device=device)
    # nvdiffrast writes rows in image order for this CUDA path.  Negating the
    # camera y coordinate therefore matches the top-left image convention.
    x_clip = camera_vertices[:, 0] / tangent + offset[0] * depth
    y_clip = -camera_vertices[:, 1] / tangent + offset[1] * depth
    z_clip = torch.zeros_like(depth)
    return torch.stack((x_clip, y_clip, z_clip, depth), dim=1)


def rasterize_silhouette(
    context: dr.RasterizeCudaContext,
    world_vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: ConditionCamera,
    field_of_view_degrees: float,
    resolution: int,
) -> torch.Tensor:
    """Render an antialiased silhouette with official nvdiffrast operators."""
    clip = _project_clip(world_vertices, camera, field_of_view_degrees)
    raster, _ = dr.rasterize(
        context, clip[None], faces, [int(resolution), int(resolution)], grad_db=True,
    )
    coverage = torch.clamp(raster[..., 3:4], 0.0, 1.0)
    return dr.antialias(coverage, raster, clip[None], faces)[0, ..., 0]


def rasterize_vertex_rgb(
    context: dr.RasterizeCudaContext,
    world_vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_rgb: torch.Tensor,
    camera: ConditionCamera,
    field_of_view_degrees: float,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render vertex colour and alpha with official nvdiffrast operators."""
    clip = _project_clip(world_vertices, camera, field_of_view_degrees)
    raster, _ = dr.rasterize(
        context, clip[None], faces, [int(resolution), int(resolution)], grad_db=True,
    )
    colour, _ = dr.interpolate(vertex_rgb[None], raster, faces)
    alpha = torch.clamp(raster[..., 3:4], 0.0, 1.0)
    colour = dr.antialias(colour * alpha, raster, clip[None], faces)[0]
    alpha = dr.antialias(alpha, raster, clip[None], faces)[0, ..., 0]
    return colour.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


def _robust_camera1_rgb_loss(
    predicted_rgb: torch.Tensor,
    predicted_alpha: torch.Tensor,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Match low-frequency semantic colour layout, not exact illumination."""
    def blur(image: torch.Tensor) -> torch.Tensor:
        value = image.permute(2, 0, 1)[None]
        value = F.avg_pool2d(value, kernel_size=11, stride=1, padding=5)
        return value[0].permute(1, 2, 0)

    predicted = blur(predicted_rgb)
    target = blur(target_rgb)
    predicted_sum = predicted.sum(dim=-1, keepdim=True).clamp_min(0.08)
    target_sum = target.sum(dim=-1, keepdim=True).clamp_min(0.08)
    predicted_chroma = predicted / predicted_sum
    target_chroma = target / target_sum
    chroma_error = F.smooth_l1_loss(
        predicted_chroma, target_chroma, beta=0.03, reduction="none",
    ).mean(dim=-1)
    luma_error = F.smooth_l1_loss(
        predicted.mean(dim=-1), target.mean(dim=-1), beta=0.05, reduction="none",
    )
    weight = (predicted_alpha * target_mask).detach()
    return ((chroma_error + 0.20 * luma_error) * weight).sum() / weight.sum().clamp_min(1.0)


def _soft_mask_loss(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    intersection = (predicted * target).sum()
    predicted_mass = predicted.sum().clamp_min(1.0)
    target_mass = target.sum().clamp_min(1.0)
    union = predicted_mass + target_mass - intersection
    iou = intersection / union.clamp_min(1.0)
    coverage = intersection / target_mass
    leakage = (predicted * (1.0 - target)).sum() / predicted_mass

    coordinate = torch.linspace(
        -1.0, 1.0, predicted.shape[0], dtype=predicted.dtype, device=predicted.device,
    )
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)

    def moments(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mass = mask.sum().clamp_min(1.0)
        centre = (mask[..., None] * grid).sum(dim=(0, 1)) / mass
        variance = (
            mask[..., None] * torch.square(grid - centre[None, None])
        ).sum(dim=(0, 1)) / mass
        return centre, torch.sqrt(variance.clamp_min(1e-6))

    predicted_centre, predicted_spread = moments(predicted)
    target_centre, target_spread = moments(target)
    centre_error = F.smooth_l1_loss(predicted_centre, target_centre, beta=0.02)
    spread_error = F.smooth_l1_loss(
        torch.log(predicted_spread), torch.log(target_spread), beta=0.02,
    )
    loss = (
        0.58 * (1.0 - iou)
        + 0.16 * (1.0 - coverage)
        + 0.16 * leakage
        + 0.10 * (centre_error + 0.5 * spread_error)
    )
    return loss, {
        "iou": iou,
        "coverage": coverage,
        "leakage": leakage,
        "centre_error": centre_error,
        "spread_error": spread_error,
    }


def _stable_pairs(
    source_world: np.ndarray,
    partial: np.ndarray,
    *,
    fraction: float,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Freeze only the initially supported part of the partial observation."""
    distances, indices = cKDTree(source_world).query(partial, k=1, workers=-1)
    threshold = float(np.quantile(distances, float(fraction)))
    selected = np.flatnonzero(distances <= threshold)
    rng = np.random.default_rng(seed)
    if len(selected) > int(samples):
        selected = rng.choice(selected, int(samples), replace=False)
    return source_world[indices[selected]], partial[selected], {
        "fraction": float(fraction),
        "threshold": threshold,
        "pairs": int(len(selected)),
    }


def refine_joint_object_and_cameras(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    source_points: np.ndarray,
    partial: np.ndarray,
    initial_transform: np.ndarray,
    cameras: list[ConditionCamera],
    field_of_view_degrees: float,
    *,
    config: RasterRefinementConfig,
    initial_auxiliary_rotation_vectors: dict[str, np.ndarray] | None = None,
    mesh_vertex_colors: np.ndarray | None = None,
    device: str = "cuda",
    seed: int = 6145,
) -> RasterRefinementResult:
    """Fit shared object Sim(3) and small auxiliary camera-orbit corrections.

    The first condition view is the gauge camera and remains fixed.  Auxiliary
    camera rotations explain image-generation view drift; object pose, scale,
    and translation remain shared by every view and by the partial scan.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("nvdiffrast joint refinement requires a CUDA device")
    if len(cameras) < 2:
        raise ValueError("joint camera/object refinement needs at least two views")

    initial_transform = np.asarray(initial_transform, dtype=np.float64)
    vertices_world = apply_transform(np.asarray(mesh_vertices), initial_transform)
    source_world = apply_transform(np.asarray(source_points), initial_transform)
    partial = np.asarray(partial, dtype=np.float64)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    pivot = np.median(vertices_world, axis=0)
    anchor_source, anchor_target, pair_info = _stable_pairs(
        source_world, partial,
        fraction=config.stable_fraction,
        samples=config.stable_samples,
        seed=seed,
    )

    dtype = torch.float32
    vertices = torch.as_tensor(vertices_world - pivot, dtype=dtype, device=device)
    faces = torch.as_tensor(np.asarray(mesh_faces), dtype=torch.int32, device=device)
    pivot_t = torch.as_tensor(pivot, dtype=dtype, device=device)
    anchor_source_t = torch.as_tensor(anchor_source - pivot, dtype=dtype, device=device)
    anchor_target_t = torch.as_tensor(anchor_target, dtype=dtype, device=device)
    target_masks = [
        torch.as_tensor(camera.mask, dtype=dtype, device=device) for camera in cameras
    ]
    use_camera1_rgb = (
        float(config.camera1_rgb_weight) > 0.0
        and cameras[0].rgb is not None
        and mesh_vertex_colors is not None
    )
    target_camera1_rgb = None
    vertex_rgb = None
    if use_camera1_rgb:
        target_camera1_rgb = torch.as_tensor(
            cameras[0].rgb, dtype=dtype, device=device,
        )
        vertex_rgb = torch.as_tensor(
            np.asarray(mesh_vertex_colors)[:, :3], dtype=dtype, device=device,
        ).clamp(0.0, 1.0)

    auxiliary_names = [camera.name for camera in cameras[1:]]
    initial_auxiliary_rotation_vectors = initial_auxiliary_rotation_vectors or {}
    base_auxiliary = []
    for name in auxiliary_names:
        vector = np.asarray(
            initial_auxiliary_rotation_vectors.get(name, np.zeros(3)), dtype=np.float32,
        )
        base_auxiliary.append(vector)
    base_auxiliary_t = torch.as_tensor(
        np.stack(base_auxiliary), dtype=dtype, device=device,
    )
    base_auxiliary_rotation = so3_exp_map(base_auxiliary_t)

    object_raw = torch.zeros(7, dtype=dtype, device=device, requires_grad=True)
    auxiliary_raw = torch.zeros(
        (len(auxiliary_names), 3), dtype=dtype, device=device, requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [object_raw, auxiliary_raw], lr=float(config.learning_rate),
    )
    context = dr.RasterizeCudaContext(device=device)
    best = None
    history = []
    max_object_angle = math.radians(float(config.max_object_rotation_degrees))
    max_auxiliary_angle = math.radians(float(config.max_auxiliary_rotation_degrees))

    for iteration in range(int(config.steps)):
        optimizer.zero_grad(set_to_none=True)
        object_rotvec = torch.tanh(object_raw[:3]) * (max_object_angle / math.sqrt(3.0))
        object_rotation = so3_exp_map(object_rotvec[None])[0]
        object_scale = torch.exp(torch.tanh(object_raw[3]) * float(config.max_log_scale))
        object_translation = torch.tanh(object_raw[4:7]) * (
            float(config.max_translation_ratio) * diagonal
        )
        world = (vertices @ object_rotation.T) * object_scale + pivot_t + object_translation

        auxiliary_delta = torch.tanh(auxiliary_raw) * (
            max_auxiliary_angle / math.sqrt(3.0)
        )
        auxiliary_rotation = so3_exp_map(auxiliary_delta) @ base_auxiliary_rotation
        view_losses = []
        view_statistics = []
        camera1_rgb_loss = torch.zeros((), dtype=dtype, device=device)
        for index, (camera, target) in enumerate(zip(cameras, target_masks)):
            view_world = world
            if index > 0:
                orbit = auxiliary_rotation[index - 1]
                view_world = (world - pivot_t) @ orbit.T + pivot_t
            predicted = rasterize_silhouette(
                context, view_world, faces, camera,
                field_of_view_degrees, config.resolution,
            )
            view_loss, statistics = _soft_mask_loss(predicted, target)
            view_losses.append(view_loss)
            view_statistics.append(statistics)
            if index == 0 and use_camera1_rgb:
                predicted_rgb, predicted_alpha = rasterize_vertex_rgb(
                    context, view_world, faces, vertex_rgb, camera,
                    field_of_view_degrees, config.resolution,
                )
                camera1_rgb_loss = _robust_camera1_rgb_loss(
                    predicted_rgb, predicted_alpha, target_camera1_rgb, target,
                )

        stacked = torch.stack(view_losses)
        # Smooth worst-view emphasis prevents an easy front view from hiding
        # a side/back error while preserving stable gradients.
        mask_loss = 0.65 * stacked.mean() + 0.35 * torch.logsumexp(8.0 * stacked, dim=0) / 8.0

        moved_anchor = (
            (anchor_source_t @ object_rotation.T) * object_scale
            + pivot_t + object_translation
        )
        anchor_distance = torch.linalg.norm(moved_anchor - anchor_target_t, dim=1) / diagonal
        partial_loss = torch.sqrt(torch.square(anchor_distance) + 1e-6).mean()
        regularizer = (
            0.004 * torch.square(object_rotvec / max(max_object_angle, 1e-8)).sum()
            + 0.003 * torch.square(torch.log(object_scale))
            + 0.002 * torch.square(object_translation / diagonal).sum()
            + 0.006 * torch.square(auxiliary_delta / max(max_auxiliary_angle, 1e-8)).sum()
        )
        loss = (
            mask_loss
            + float(config.partial_weight) * partial_loss
            + float(config.camera1_rgb_weight) * camera1_rgb_loss
            + regularizer
        )
        loss.backward()
        optimizer.step()

        value = float(loss.detach().cpu())
        if best is None or value < best["loss"]:
            best = {
                "loss": value,
                "mask_loss": float(mask_loss.detach().cpu()),
                "partial_loss": float(partial_loss.detach().cpu()),
                "camera1_rgb_loss": float(camera1_rgb_loss.detach().cpu()),
                "rotation": object_rotation.detach().cpu().numpy().astype(np.float64),
                "scale": float(object_scale.detach().cpu()),
                "translation": object_translation.detach().cpu().numpy().astype(np.float64),
                "auxiliary_rotation": auxiliary_rotation.detach().cpu().clone(),
                "view_statistics": [
                    {key: float(item.detach().cpu()) for key, item in statistics.items()}
                    for statistics in view_statistics
                ],
            }
        if iteration % 25 == 0 or iteration + 1 == int(config.steps):
            history.append({
                "iteration": int(iteration),
                "loss": value,
                "mask_loss": float(mask_loss.detach().cpu()),
                "partial_loss": float(partial_loss.detach().cpu()),
                "camera1_rgb_loss": float(camera1_rgb_loss.detach().cpu()),
            })

    assert best is not None
    transform = compose_pivoted_sim3(
        initial_transform, pivot, best["rotation"], best["scale"], best["translation"],
    )
    auxiliary_vectors = {cameras[0].name: np.zeros(3, dtype=np.float64)}
    vectors = matrix_to_axis_angle(best["auxiliary_rotation"]).cpu().numpy().astype(np.float64)
    auxiliary_vectors.update(dict(zip(auxiliary_names, vectors)))
    return RasterRefinementResult(
        transform=transform,
        auxiliary_rotation_vectors=auxiliary_vectors,
        diagnostics={
            "renderer": "nvdiffrast",
            "rotation_parameterization": "pytorch3d.so3_exp_map",
            "resolution": int(config.resolution),
            "steps": int(config.steps),
            "best_loss": best["loss"],
            "mask_loss": best["mask_loss"],
            "partial_anchor_loss": best["partial_loss"],
            "camera1_rgb_loss": best["camera1_rgb_loss"],
            "camera1_rgb_enabled": bool(use_camera1_rgb),
            "world_pivot": pivot.tolist(),
            "residual_object_rotation_degrees": float(
                np.degrees(np.linalg.norm(matrix_to_axis_angle(
                    torch.as_tensor(best["rotation"], dtype=dtype)[None]
                )[0].numpy()))
            ),
            "residual_object_scale": best["scale"],
            "residual_object_translation": best["translation"].tolist(),
            "per_view": {
                camera.name: statistics
                for camera, statistics in zip(cameras, best["view_statistics"])
            },
            "stable_pairs": pair_info,
            "history": history,
        },
    )
