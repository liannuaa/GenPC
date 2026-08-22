"""Registration, deformation-graph, and fusion ablation for GenPC.

The script is intentionally standalone. It never overwrites the frozen source
workspace or changes the default pipeline. Candidate selection is GT-free;
ground truth is loaded only by the existing metric function after all outputs
have been written.
"""

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import yaml
from munch import Munch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_render_to_moge_sim3 import (  # noqa: E402
    apply_sim3,
    axis_aligned_rotations,
    bbox_extent,
    compose_complete_to_partial,
    evaluate_2d_acceptance,
    evaluate_transform,
    initial_candidates,
    maybe_subsample,
    normalized_intrinsic_to_pixel,
    parse_float_list,
    project_points,
    rank_refined_sim3_candidates,
    score_2d_gate_objective,
    score_depth_render,
    zbuffer_depth,
    zbuffer_depth_with_indices,
)
from utils.runtime import normalize_runtime_config  # noqa: E402


DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT / "workspace" / "redwood_onestage_rawdepth_512_stage2_20260714"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_SOURCE_ROOT / "_ablation_registration_graph_fusion_20260820"
)
VARIANTS = (
    "sim3_baseline",
    "raw_partial_rerank",
    "graph_se3",
    "graph_scale",
    "graph_scale_partial_priority_fusion",
)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2))


def load_points(path):
    path = Path(path)
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return points


def write_points(path, points, color=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if color is not None:
        cloud.paint_uniform_color(color)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f"Failed to write point cloud: {path}")


def write_compare(path, partial, generated):
    partial_cloud = o3d.geometry.PointCloud()
    partial_cloud.points = o3d.utility.Vector3dVector(np.asarray(partial, dtype=np.float64))
    partial_cloud.paint_uniform_color([0.55, 0.55, 0.55])
    generated_cloud = o3d.geometry.PointCloud()
    generated_cloud.points = o3d.utility.Vector3dVector(
        np.asarray(generated, dtype=np.float64)
    )
    generated_cloud.paint_uniform_color([0.9, 0.1, 0.1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), partial_cloud + generated_cloud):
        raise IOError(f"Failed to write comparison point cloud: {path}")


def estimate_normals(points, radius):
    points = np.asarray(points, dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(float(radius), 1e-5), max_nn=30
        )
    )
    normals = np.asarray(cloud.normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-8
    normals[valid] /= lengths[valid]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normals


def transform_normals(normals, transform):
    normals = np.asarray(normals, dtype=np.float64)
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    moved = normals @ np.linalg.inv(linear)
    lengths = np.linalg.norm(moved, axis=1, keepdims=True)
    return moved / np.maximum(lengths, 1e-12)


class SavedCameraProjector:
    """Project raw-frame points with the exact saved DepthPrompting camera."""

    def __init__(
        self,
        camera,
        center_xy,
        scale_xy,
        *,
        padding,
        image_shape,
        device="cpu",
    ):
        self.camera = camera
        self.center_xy = np.asarray(center_xy, dtype=np.float64)
        self.scale_xy = float(scale_xy)
        self.padding = float(padding)
        self.image_shape = tuple(int(value) for value in image_shape)
        self.device = str(device)

    @classmethod
    def from_partial(
        cls,
        partial_points,
        camera_path,
        *,
        padding,
        image_shape,
        device="cpu",
    ):
        camera = torch.load(
            str(camera_path), map_location=device, weights_only=False
        )
        points = torch.as_tensor(
            np.asarray(partial_points), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            camera_points = camera.transform(points).detach().float().cpu().numpy()
        xy_min = camera_points[:, :2].min(axis=0)
        xy_max = camera_points[:, :2].max(axis=0)
        center = (xy_min + xy_max) * 0.5
        scale = max(float((xy_max - xy_min).max()), 1e-8)
        return cls(
            camera,
            center,
            scale,
            padding=padding,
            image_shape=image_shape,
            device=device,
        )

    def project(self, points):
        points = np.asarray(points, dtype=np.float64)
        transformed_chunks = []
        chunk_size = 200000
        with torch.no_grad():
            for start in range(0, len(points), chunk_size):
                tensor = torch.as_tensor(
                    points[start : start + chunk_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                transformed_chunks.append(
                    self.camera.transform(tensor).detach().float().cpu().numpy()
                )
        camera_points = np.concatenate(transformed_chunks, axis=0)
        uv = (camera_points[:, :2] - self.center_xy) / self.scale_xy
        uv = uv * (1.0 - 2.0 * self.padding) + 0.5
        uv[:, 1] = 1.0 - uv[:, 1]
        height, width = self.image_shape
        pixel = uv * np.array([width - 1, height - 1], dtype=np.float64)
        return pixel, camera_points[:, 2]


def grid_coverage(pixel_xy, matched, image_shape, grid_size=8):
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    matched = np.asarray(matched, dtype=bool)
    height, width = int(image_shape[0]), int(image_shape[1])
    if len(pixel_xy) == 0:
        return 0.0, 0, 0
    gx = np.clip((pixel_xy[:, 0] * int(grid_size) / max(width, 1)).astype(int), 0, int(grid_size) - 1)
    gy = np.clip((pixel_xy[:, 1] * int(grid_size) / max(height, 1)).astype(int), 0, int(grid_size) - 1)
    occupied = set(zip(gx.tolist(), gy.tolist()))
    covered = set(zip(gx[matched].tolist(), gy[matched].tolist()))
    return (
        float(len(covered) / max(len(occupied), 1)),
        int(len(covered)),
        int(len(occupied)),
    )


def evaluate_raw_gate_from_projection(
    partial_points,
    complete_points,
    partial_uv,
    partial_depth,
    complete_uv,
    complete_depth,
    *,
    image_shape,
    bbox_diagonal,
    partial_normals=None,
    complete_normals=None,
    pixel_radius=2.0,
    splat_radius=1,
    distance_ratio=0.04,
    normal_cosine=0.5,
    min_coverage=0.70,
    min_grid_coverage=0.70,
    max_depth_mean_ratio=0.03,
    max_depth_p95_ratio=0.08,
):
    """Evaluate same-ray partial-to-complete consistency without GT."""
    partial_points = np.asarray(partial_points, dtype=np.float64)
    complete_points = np.asarray(complete_points, dtype=np.float64)
    diagonal = max(float(bbox_diagonal), 1e-8)
    partial_render, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, image_shape, splat_radius=0
    )
    complete_render, complete_mask, complete_index = zbuffer_depth_with_indices(
        complete_uv, complete_depth, image_shape, splat_radius=splat_radius
    )
    py, px = np.where(partial_mask)
    visible_count = int(len(px))
    if visible_count == 0 or not complete_mask.any():
        info = {
            "accepted": False,
            "failed": ["visible_pixels"],
            "visible_partial_pixels": visible_count,
            "matched_pixels": 0,
            "coverage": 0.0,
            "grid_coverage": 0.0,
            "depth_mean": None,
            "depth_p95": None,
            "distance_mean": None,
            "distance_p95": None,
            "cost": None,
        }
        return info, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    cy, cx = np.where(complete_mask)
    pixel_tree = cKDTree(np.c_[cx, cy])
    pixel_distances, nearest_pixel = pixel_tree.query(np.c_[px, py], k=1)
    near_pixel = pixel_distances <= float(pixel_radius)
    nearest_x = cx[nearest_pixel]
    nearest_y = cy[nearest_pixel]
    partial_ids = partial_index[py, px]
    complete_ids = complete_index[nearest_y, nearest_x]

    point_distances = np.linalg.norm(
        partial_points[partial_ids] - complete_points[complete_ids], axis=1
    )
    point_ok = point_distances <= float(distance_ratio) * diagonal
    normal_ok = np.ones(len(partial_ids), dtype=bool)
    if partial_normals is not None and complete_normals is not None:
        dots = np.abs(
            np.sum(
                np.asarray(partial_normals)[partial_ids]
                * np.asarray(complete_normals)[complete_ids],
                axis=1,
            )
        )
        normal_ok = dots >= float(normal_cosine)

    geometric_match = point_ok & normal_ok
    accepted_match = near_pixel & geometric_match
    accepted_partial = partial_ids[accepted_match]
    accepted_complete = complete_ids[accepted_match]
    coverage = float(accepted_match.sum() / max(visible_count, 1))
    grid_value, covered_cells, occupied_cells = grid_coverage(
        np.c_[px, py], accepted_match, image_shape
    )

    near_depth_errors = np.abs(
        partial_render[py[near_pixel], px[near_pixel]]
        - complete_render[nearest_y[near_pixel], nearest_x[near_pixel]]
    )
    if accepted_match.any():
        depth_errors = np.abs(
            partial_render[py[accepted_match], px[accepted_match]]
            - complete_render[
                nearest_y[accepted_match], nearest_x[accepted_match]
            ]
        )
        accepted_distances = point_distances[accepted_match]
        depth_mean = float(depth_errors.mean())
        depth_p95 = float(np.percentile(depth_errors, 95))
        distance_mean = float(accepted_distances.mean())
        distance_p95 = float(np.percentile(accepted_distances, 95))
    else:
        depth_mean = float("inf")
        depth_p95 = float("inf")
        distance_mean = float("inf")
        distance_p95 = float("inf")

    failed = []
    if coverage < float(min_coverage):
        failed.append("coverage")
    if grid_value < float(min_grid_coverage):
        failed.append("grid_coverage")
    if depth_mean > float(max_depth_mean_ratio) * diagonal:
        failed.append("depth_mean")
    if depth_p95 > float(max_depth_p95_ratio) * diagonal:
        failed.append("depth_p95")
    cost = (
        depth_mean
        + 0.5 * depth_p95
        + 0.25 * diagonal * (1.0 - coverage)
        + 0.25 * diagonal * (1.0 - grid_value)
    )
    info = {
        "accepted": not failed,
        "reason": "raw_partial_gate_met" if not failed else "raw_partial_gate_not_met",
        "failed": failed,
        "visible_partial_pixels": visible_count,
        "same_ray_pixels": int(near_pixel.sum()),
        "matched_pixels": int(accepted_match.sum()),
        "same_ray_coverage": float(near_pixel.sum() / max(visible_count, 1)),
        "coverage": coverage,
        "grid_coverage": grid_value,
        "covered_cells": covered_cells,
        "occupied_cells": occupied_cells,
        "pixel_distance_mean": float(pixel_distances[accepted_match].mean())
        if accepted_match.any()
        else None,
        "same_ray_depth_mean": float(near_depth_errors.mean())
        if near_pixel.any()
        else None,
        "same_ray_depth_p95": float(np.percentile(near_depth_errors, 95))
        if near_pixel.any()
        else None,
        "same_ray_distance_pass_ratio": float(point_ok[near_pixel].mean())
        if near_pixel.any()
        else 0.0,
        "same_ray_normal_pass_ratio": float(normal_ok[near_pixel].mean())
        if near_pixel.any()
        else 0.0,
        "same_ray_geometric_pass_ratio": float(geometric_match[near_pixel].mean())
        if near_pixel.any()
        else 0.0,
        "depth_mean": depth_mean,
        "depth_p95": depth_p95,
        "depth_mean_ratio": depth_mean / diagonal,
        "depth_p95_ratio": depth_p95 / diagonal,
        "distance_mean": distance_mean,
        "distance_p95": distance_p95,
        "distance_mean_ratio": distance_mean / diagonal,
        "distance_p95_ratio": distance_p95 / diagonal,
        "cost": cost,
        "thresholds": {
            "pixel_radius": float(pixel_radius),
            "distance_ratio": float(distance_ratio),
            "normal_cosine": float(normal_cosine),
            "min_coverage": float(min_coverage),
            "min_grid_coverage": float(min_grid_coverage),
            "max_depth_mean_ratio": float(max_depth_mean_ratio),
            "max_depth_p95_ratio": float(max_depth_p95_ratio),
        },
    }
    return info, accepted_partial, accepted_complete


def evaluate_raw_partial_gate(
    partial_points,
    complete_points,
    projector,
    *,
    bbox_diagonal,
    partial_normals=None,
    complete_normals=None,
    distance_ratio=0.04,
):
    partial_uv, partial_depth = projector.project(partial_points)
    complete_uv, complete_depth = projector.project(complete_points)
    return evaluate_raw_gate_from_projection(
        partial_points,
        complete_points,
        partial_uv,
        partial_depth,
        complete_uv,
        complete_depth,
        image_shape=projector.image_shape,
        bbox_diagonal=bbox_diagonal,
        partial_normals=partial_normals,
        complete_normals=complete_normals,
        distance_ratio=distance_ratio,
    )


def load_moge_context(sample_dir, flag):
    info_path = sample_dir / f"{flag}_render_to_moge_sim3_info.json"
    info = json.loads(info_path.read_text())
    image_shape = tuple(int(value) for value in info["moge"]["image_hw"])
    intrinsic_px = normalized_intrinsic_to_pixel(
        np.asarray(info["moge"]["output_keys"]["intrinsics"], dtype=np.float64),
        image_shape,
    )
    moge_points = load_points(sample_dir / f"{flag}_moge_object_only.ply")
    uv, valid = project_points(moge_points, intrinsic_px, image_shape)
    target_depth, target_mask = zbuffer_depth(
        uv[valid], moge_points[valid, 2], image_shape, splat_radius=1
    )
    moge_to_partial = np.load(
        sample_dir
        / f"{flag}_moge_to_raw_partial_moge_to_raw_partial_transform.npy"
    )
    return {
        "info": info,
        "image_shape": image_shape,
        "intrinsic_px": intrinsic_px,
        "moge_points": moge_points,
        "target_depth": target_depth,
        "target_mask": target_mask,
        "moge_to_partial": np.asarray(moge_to_partial, dtype=np.float64),
        "partial_to_moge": np.linalg.inv(
            np.asarray(moge_to_partial, dtype=np.float64)
        ),
    }


def evaluate_moge_points(points_in_partial, context, gate_thresholds=None):
    points_in_moge = apply_sim3(points_in_partial, context["partial_to_moge"])
    uv, valid = project_points(
        points_in_moge, context["intrinsic_px"], context["image_shape"]
    )
    depth, mask = zbuffer_depth(
        uv[valid],
        points_in_moge[valid, 2],
        context["image_shape"],
        splat_radius=1,
    )
    score = score_depth_render(
        depth, mask, context["target_depth"], context["target_mask"]
    )
    thresholds = gate_thresholds or {}
    gate = evaluate_2d_acceptance(
        score,
        enabled=True,
        min_iou=thresholds.get("min_iou", 0.82),
        min_coverage=thresholds.get("min_coverage", 0.84),
        max_leakage=thresholds.get("max_leakage", 0.10),
        min_edge_iou=thresholds.get("min_edge_iou", 0.025),
        max_edge_chamfer_px=thresholds.get("max_edge_chamfer_px", 18.0),
    )
    return score, gate, mask


def centered_delta(scale, rotation, translation, center):
    linear = float(scale) * np.asarray(rotation, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = center + np.asarray(translation) - linear @ center
    return transform


def bounded_raw_sim3_refinement(
    native_points,
    native_normals,
    initial_transform,
    partial_points,
    partial_normals,
    projector,
    context,
    *,
    bbox_diagonal,
):
    """Coordinate-search a small raw-frame delta Sim3 under both hard gates."""
    initial_transform = np.asarray(initial_transform, dtype=np.float64)
    initial_center = apply_sim3(native_points, initial_transform).mean(axis=0)

    def evaluate(transform):
        points = apply_sim3(native_points, transform)
        normals = transform_normals(native_normals, transform)
        raw, _, _ = evaluate_raw_partial_gate(
            partial_points,
            points,
            projector,
            bbox_diagonal=bbox_diagonal,
            partial_normals=partial_normals,
            complete_normals=normals,
        )
        complete_to_moge = context["partial_to_moge"] @ transform
        score = evaluate_transform(
            native_points,
            complete_to_moge,
            context["intrinsic_px"],
            context["target_depth"],
            context["target_mask"],
            splat_radius=1,
        )
        gate = evaluate_2d_acceptance(
            score,
            enabled=True,
            min_iou=0.82,
            min_coverage=0.84,
            max_leakage=0.10,
            min_edge_iou=0.025,
            max_edge_chamfer_px=18.0,
        )
        return points, raw, score, gate

    current = initial_transform.copy()
    current_points, current_raw, current_score, current_gate = evaluate(current)
    history = []
    for rotation_step, translation_ratio, scale_step in (
        (4.0, 0.02, 1.05),
        (2.0, 0.01, 1.02),
    ):
        for _ in range(2):
            proposals = []
            center = current_points.mean(axis=0)
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    translation = np.zeros(3, dtype=np.float64)
                    translation[axis] = sign * translation_ratio * bbox_diagonal
                    proposals.append(
                        centered_delta(1.0, np.eye(3), translation, center) @ current
                    )
                    rotation = Rotation.from_rotvec(
                        np.eye(3)[axis] * math.radians(sign * rotation_step)
                    ).as_matrix()
                    proposals.append(
                        centered_delta(1.0, rotation, np.zeros(3), center) @ current
                    )
            for factor in (1.0 / scale_step, scale_step):
                proposals.append(
                    centered_delta(factor, np.eye(3), np.zeros(3), center) @ current
                )

            best = None
            for proposal in proposals:
                delta = proposal @ np.linalg.inv(initial_transform)
                delta_scale = float(np.cbrt(np.linalg.det(delta[:3, :3])))
                delta_rotation = delta[:3, :3] / max(delta_scale, 1e-12)
                rotation_degrees = math.degrees(
                    Rotation.from_matrix(delta_rotation).magnitude()
                )
                center_shift = np.linalg.norm(
                    apply_sim3(native_points.mean(axis=0, keepdims=True), proposal)[0]
                    - initial_center
                )
                if not (0.90 <= delta_scale <= 1.10):
                    continue
                if rotation_degrees > 8.0 + 1e-6:
                    continue
                if center_shift > 0.05 * bbox_diagonal + 1e-8:
                    continue
                points, raw, score, gate = evaluate(proposal)
                if not raw["accepted"] or not gate["accepted"]:
                    continue
                if raw["cost"] >= current_raw["cost"] - 1e-10:
                    continue
                item = (raw["cost"], -score_2d_gate_objective(score), proposal, points, raw, score, gate)
                if best is None or item[:2] < best[:2]:
                    best = item
            if best is None:
                break
            _, _, current, current_points, current_raw, current_score, current_gate = best
            history.append(
                {
                    "rotation_step_deg": rotation_step,
                    "translation_ratio": translation_ratio,
                    "scale_step": scale_step,
                    "raw_gate": current_raw,
                    "moge_score": current_score,
                }
            )

    return current, current_points, {
        "accepted": bool(history),
        "reason": "raw_cost_improved" if history else "no_bounded_improvement",
        "history": history,
        "raw_gate": current_raw,
        "moge_score": current_score,
        "moge_gate": current_gate,
    }


def farthest_point_nodes(points, spacing, min_nodes=64, max_nodes=256):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Cannot build a deformation graph from no points")
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    selected = [first]
    min_sq = np.sum((points - points[first]) ** 2, axis=1)
    while len(selected) < min(int(max_nodes), len(points)):
        next_index = int(np.argmax(min_sq))
        if len(selected) >= int(min_nodes) and math.sqrt(float(min_sq[next_index])) <= float(spacing):
            break
        if next_index in selected:
            break
        selected.append(next_index)
        min_sq = np.minimum(
            min_sq, np.sum((points - points[next_index]) ** 2, axis=1)
        )
    return points[np.asarray(selected, dtype=np.int64)]


def influence_weights(points, nodes, k):
    k = min(int(k), len(nodes))
    distances, indices = cKDTree(nodes).query(points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    epsilon = max(float(np.median(distances[:, -1])) * 0.05, 1e-6)
    weights = 1.0 / np.maximum(distances + epsilon, 1e-8) ** 2
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return indices.astype(np.int64), weights.astype(np.float64)


def build_deformation_graph(
    points,
    *,
    node_spacing_ratio=0.05,
    min_nodes=64,
    max_nodes=256,
    graph_knn=6,
    influence_k=4,
):
    points = np.asarray(points, dtype=np.float64)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    diagonal = max(float(np.linalg.norm(hi - lo)), 1e-8)
    normalized = (points - center) / diagonal
    nodes = farthest_point_nodes(
        normalized,
        float(node_spacing_ratio),
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )
    point_nodes, point_weights = influence_weights(
        normalized, nodes, influence_k
    )
    node_nodes, node_weights = influence_weights(nodes, nodes, influence_k)
    edge_k = min(int(graph_knn) + 1, len(nodes))
    _, neighbors = cKDTree(nodes).query(nodes, k=edge_k)
    if edge_k == 1:
        neighbors = neighbors[:, None]
    edges = set()
    for node_index, row in enumerate(neighbors):
        for neighbor in np.atleast_1d(row):
            neighbor = int(neighbor)
            if neighbor == node_index:
                continue
            edges.add(tuple(sorted((int(node_index), neighbor))))
    return {
        "center": center,
        "diagonal": diagonal,
        "points": normalized,
        "nodes": nodes,
        "point_nodes": point_nodes,
        "point_weights": point_weights,
        "node_nodes": node_nodes,
        "node_weights": node_weights,
        "edges": np.asarray(sorted(edges), dtype=np.int64),
    }


def batched_rodrigues(rotvec):
    theta = torch.linalg.norm(rotvec, dim=1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / theta
    x, y, z = axis.unbind(dim=1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(
        len(rotvec), -1, -1
    )
    angle = theta[:, :, None]
    return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def deform_graph_torch(
    points,
    nodes,
    node_indices,
    weights,
    rotvec,
    translations,
    log_scales,
):
    rotations = batched_rodrigues(rotvec)
    selected_nodes = nodes[node_indices]
    relative = points[:, None, :] - selected_nodes
    local_scale = torch.exp(log_scales[node_indices])[..., None]
    selected_rotations = rotations[node_indices]
    moved = torch.matmul(
        selected_rotations, (local_scale * relative)[..., None]
    )[..., 0]
    moved = moved + selected_nodes + translations[node_indices]
    return torch.sum(weights[..., None] * moved, dim=1)


def graph_points_numpy(graph, state, points_key="points", nodes_key="point_nodes", weights_key="point_weights"):
    device = torch.device("cpu")
    with torch.no_grad():
        points = torch.as_tensor(graph[points_key], dtype=torch.float32, device=device)
        nodes = torch.as_tensor(graph["nodes"], dtype=torch.float32, device=device)
        node_indices = torch.as_tensor(graph[nodes_key], dtype=torch.long, device=device)
        weights = torch.as_tensor(graph[weights_key], dtype=torch.float32, device=device)
        rotvec = torch.as_tensor(state["rotvec"], dtype=torch.float32, device=device)
        translations = torch.as_tensor(
            state["translations"], dtype=torch.float32, device=device
        )
        log_scales = torch.as_tensor(
            state["log_scales"], dtype=torch.float32, device=device
        )
        normalized = deform_graph_torch(
            points,
            nodes,
            node_indices,
            weights,
            rotvec,
            translations,
            log_scales,
        ).cpu().numpy()
    return normalized * graph["diagonal"] + graph["center"]


def graph_edge_distortion(graph, state):
    moved_nodes = graph_points_numpy(
        graph,
        state,
        points_key="nodes",
        nodes_key="node_nodes",
        weights_key="node_weights",
    )
    normalized_moved = (moved_nodes - graph["center"]) / graph["diagonal"]
    edges = graph["edges"]
    if len(edges) == 0:
        return {"outside_ratio": 0.0, "min_ratio": 1.0, "max_ratio": 1.0}
    before = np.linalg.norm(
        graph["nodes"][edges[:, 0]] - graph["nodes"][edges[:, 1]], axis=1
    )
    after = np.linalg.norm(
        normalized_moved[edges[:, 0]] - normalized_moved[edges[:, 1]], axis=1
    )
    ratios = after / np.maximum(before, 1e-8)
    return {
        "outside_ratio": float(((ratios < 0.75) | (ratios > 1.25)).mean()),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "mean_ratio": float(ratios.mean()),
    }


def clamp_deformation_parameters(
    rotvec,
    translations,
    log_scales,
    *,
    allow_scale,
):
    """Apply the hard deformation bounds in place."""
    with torch.no_grad():
        translation_norm = torch.linalg.norm(
            translations, dim=1, keepdim=True
        ).clamp_min(1e-8)
        translations.mul_(torch.clamp(0.08 / translation_norm, max=1.0))
        rotation_norm = torch.linalg.norm(
            rotvec, dim=1, keepdim=True
        ).clamp_min(1e-8)
        rotvec.mul_(torch.clamp(0.35 / rotation_norm, max=1.0))
        if allow_scale:
            log_scales.clamp_(math.log(0.85), math.log(1.15))
        else:
            log_scales.zero_()


def optimize_deformation_graph(
    graph,
    input_points,
    partial_points,
    partial_normals,
    projector,
    context,
    baseline_raw_gate,
    *,
    allow_scale,
    initial_state=None,
    iterations=80,
    device="cuda",
    required_mean_improvement=0.02,
    seed=6145,
):
    device = torch.device(device)
    node_count = len(graph["nodes"])
    if initial_state is None:
        initial_state = {
            "rotvec": np.zeros((node_count, 3), dtype=np.float32),
            "translations": np.zeros((node_count, 3), dtype=np.float32),
            "log_scales": np.zeros(node_count, dtype=np.float32),
        }
    rotvec = torch.nn.Parameter(
        torch.as_tensor(initial_state["rotvec"], dtype=torch.float32, device=device).clone()
    )
    translations = torch.nn.Parameter(
        torch.as_tensor(
            initial_state["translations"], dtype=torch.float32, device=device
        ).clone()
    )
    log_scales = torch.nn.Parameter(
        torch.as_tensor(
            initial_state["log_scales"], dtype=torch.float32, device=device
        ).clone()
    )
    params = [rotvec, translations]
    if allow_scale:
        params.append(log_scales)

    nodes_t = torch.as_tensor(graph["nodes"], dtype=torch.float32, device=device)
    points_t = torch.as_tensor(graph["points"], dtype=torch.float32, device=device)
    point_nodes_t = torch.as_tensor(
        graph["point_nodes"], dtype=torch.long, device=device
    )
    point_weights_t = torch.as_tensor(
        graph["point_weights"], dtype=torch.float32, device=device
    )
    edges_t = torch.as_tensor(graph["edges"], dtype=torch.long, device=device)
    rng = np.random.default_rng(int(seed))
    histories = []
    failure_reason = None

    for stage_index, (distance_ratio, arap_weight) in enumerate(
        ((0.04, 50.0), (0.025, 20.0), (0.015, 8.0))
    ):
        state_now = {
            "rotvec": rotvec.detach().cpu().numpy(),
            "translations": translations.detach().cpu().numpy(),
            "log_scales": log_scales.detach().cpu().numpy(),
        }
        current_points = graph_points_numpy(graph, state_now)
        current_normals = estimate_normals(
            current_points, radius=0.03 * graph["diagonal"]
        )
        gate, partial_ids, source_ids = evaluate_raw_partial_gate(
            partial_points,
            current_points,
            projector,
            bbox_diagonal=graph["diagonal"],
            partial_normals=partial_normals,
            complete_normals=current_normals,
            distance_ratio=distance_ratio,
        )
        if len(source_ids) < 256:
            failure_reason = f"insufficient_correspondences_stage_{stage_index}"
            break
        if len(source_ids) > 12000:
            chosen = rng.choice(len(source_ids), size=12000, replace=False)
            source_ids = source_ids[chosen]
            partial_ids = partial_ids[chosen]

        source_ids_t = torch.as_tensor(source_ids, dtype=torch.long, device=device)
        partial_target = torch.as_tensor(
            (partial_points[partial_ids] - graph["center"]) / graph["diagonal"],
            dtype=torch.float32,
            device=device,
        )
        target_normals = torch.as_tensor(
            partial_normals[partial_ids], dtype=torch.float32, device=device
        )
        anchored = np.zeros(node_count, dtype=bool)
        anchored[np.unique(graph["point_nodes"][source_ids])] = True
        unobserved_t = torch.as_tensor(~anchored, dtype=torch.bool, device=device)

        optimizer = torch.optim.Adam(params, lr=0.01)
        best_loss = float("inf")
        for iteration in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True)
            moved = deform_graph_torch(
                points_t[source_ids_t],
                nodes_t,
                point_nodes_t[source_ids_t],
                point_weights_t[source_ids_t],
                rotvec,
                translations,
                log_scales if allow_scale else torch.zeros_like(log_scales),
            )
            residual = moved - partial_target
            plane = torch.abs(torch.sum(residual * target_normals, dim=1))
            threshold = torch.quantile(plane.detach(), 0.70)
            keep = plane <= threshold
            if int(keep.sum()) < 32:
                failure_reason = f"trimmed_correspondences_stage_{stage_index}"
                break
            data_plane = F.smooth_l1_loss(
                plane[keep], torch.zeros_like(plane[keep]), beta=0.005
            )
            data_point = F.smooth_l1_loss(
                residual[keep], torch.zeros_like(residual[keep]), beta=0.01
            )

            if len(edges_t):
                rotations = batched_rodrigues(rotvec)
                edge_i = edges_t[:, 0]
                edge_j = edges_t[:, 1]
                relative = nodes_t[edge_j] - nodes_t[edge_i]
                scale_i = (
                    torch.exp(log_scales[edge_i])[:, None]
                    if allow_scale
                    else torch.ones((len(edge_i), 1), device=device)
                )
                predicted = torch.matmul(
                    rotations[edge_i], (scale_i * relative)[..., None]
                )[..., 0]
                predicted = predicted + nodes_t[edge_i] + translations[edge_i]
                target = nodes_t[edge_j] + translations[edge_j]
                arap = torch.mean(torch.sum((predicted - target) ** 2, dim=1))
            else:
                arap = torch.zeros((), device=device)

            if unobserved_t.any():
                identity = (
                    rotvec[unobserved_t].square().mean()
                    + translations[unobserved_t].square().mean()
                )
            else:
                identity = torch.zeros((), device=device)
            if allow_scale:
                scale_identity = log_scales.square().mean()
                if len(edges_t):
                    scale_smooth = (
                        log_scales[edges_t[:, 0]] - log_scales[edges_t[:, 1]]
                    ).square().mean()
                else:
                    scale_smooth = torch.zeros((), device=device)
            else:
                scale_identity = torch.zeros((), device=device)
                scale_smooth = torch.zeros((), device=device)

            loss = (
                data_plane
                + 0.1 * data_point
                + arap_weight * arap
                + 10.0 * identity
                + 5.0 * scale_identity
                + 20.0 * scale_smooth
            )
            loss.backward()
            optimizer.step()
            clamp_deformation_parameters(
                rotvec, translations, log_scales, allow_scale=allow_scale
            )
            best_loss = min(best_loss, float(loss.detach().cpu()))
        histories.append(
            {
                "stage": stage_index,
                "distance_ratio": distance_ratio,
                "arap_weight": arap_weight,
                "pairs": int(len(source_ids)),
                "initial_gate": gate,
                "best_loss": best_loss,
            }
        )
        if failure_reason is not None:
            break

    state = {
        "rotvec": rotvec.detach().cpu().numpy(),
        "translations": translations.detach().cpu().numpy(),
        "log_scales": log_scales.detach().cpu().numpy(),
    }
    output_points = graph_points_numpy(graph, state)
    output_normals = estimate_normals(
        output_points, radius=0.03 * graph["diagonal"]
    )
    raw_gate, _, _ = evaluate_raw_partial_gate(
        partial_points,
        output_points,
        projector,
        bbox_diagonal=graph["diagonal"],
        partial_normals=partial_normals,
        complete_normals=output_normals,
        distance_ratio=0.04,
    )
    moge_score, moge_gate, _ = evaluate_moge_points(output_points, context)
    distortion = graph_edge_distortion(graph, state)
    scale_values = np.exp(state["log_scales"])
    scale_hit_ratio = float(
        ((scale_values <= 0.8505) | (scale_values >= 1.1495)).mean()
    )
    finite = bool(np.isfinite(output_points).all())
    before_mean = float(baseline_raw_gate.get("distance_mean", float("inf")))
    before_p95 = float(baseline_raw_gate.get("distance_p95", float("inf")))
    after_mean = float(raw_gate.get("distance_mean", float("inf")))
    after_p95 = float(raw_gate.get("distance_p95", float("inf")))
    required_mean = before_mean * (1.0 - float(required_mean_improvement))
    accepted = (
        failure_reason is None
        and finite
        and raw_gate["accepted"]
        and moge_gate["accepted"]
        and after_mean <= required_mean
        and after_p95 <= before_p95 * 1.02
        and distortion["outside_ratio"] <= 0.01
        and scale_hit_ratio <= 0.05
    )
    failed = []
    if failure_reason:
        failed.append(failure_reason)
    if not finite:
        failed.append("non_finite")
    if not raw_gate["accepted"]:
        failed.append("raw_gate")
    if not moge_gate["accepted"]:
        failed.append("moge_gate")
    if not after_mean <= required_mean:
        failed.append("mean_improvement")
    if not after_p95 <= before_p95 * 1.02:
        failed.append("p95")
    if distortion["outside_ratio"] > 0.01:
        failed.append("edge_distortion")
    if scale_hit_ratio > 0.05:
        failed.append("scale_clamp_hits")
    info = {
        "accepted": accepted,
        "reason": "deformation_accepted" if accepted else "deformation_rejected",
        "failed": failed,
        "allow_scale": bool(allow_scale),
        "required_mean_improvement": float(required_mean_improvement),
        "before_raw_gate": baseline_raw_gate,
        "after_raw_gate": raw_gate,
        "moge_score": moge_score,
        "moge_gate": moge_gate,
        "edge_distortion": distortion,
        "scale_min": float(scale_values.min()),
        "scale_max": float(scale_values.max()),
        "scale_hit_ratio": scale_hit_ratio,
        "max_translation_ratio": float(
            np.linalg.norm(state["translations"], axis=1).max()
        ),
        "history": histories,
    }
    if not accepted:
        output_points = np.asarray(input_points, dtype=np.float64)
    return output_points, info, state


def partial_priority_fusion(
    generated_points,
    partial_points,
    projector,
    *,
    bbox_diagonal,
    remove_ratio=0.015,
    voxel_ratio=0.003,
):
    generated_points = np.asarray(generated_points, dtype=np.float64)
    partial_points = np.asarray(partial_points, dtype=np.float64)
    partial_uv, partial_depth = projector.project(partial_points)
    generated_uv, generated_depth = projector.project(generated_points)
    _, partial_mask, _ = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=0
    )
    observed_mask = cv2.dilate(
        partial_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
    ).astype(bool)
    height, width = projector.image_shape
    xy = np.rint(generated_uv).astype(np.int64)
    valid = (
        np.isfinite(generated_uv).all(axis=1)
        & np.isfinite(generated_depth)
        & (generated_depth > 1e-8)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    observed = np.zeros(len(generated_points), dtype=bool)
    observed[valid] = observed_mask[xy[valid, 1], xy[valid, 0]]
    distances, _ = cKDTree(partial_points).query(generated_points, k=1)
    remove = observed & (distances <= float(remove_ratio) * bbox_diagonal)
    kept = generated_points[~remove]

    generated_cloud = o3d.geometry.PointCloud()
    generated_cloud.points = o3d.utility.Vector3dVector(kept)
    voxel_size = max(float(voxel_ratio) * bbox_diagonal, 1e-6)
    downsampled = np.asarray(
        generated_cloud.voxel_down_sample(voxel_size).points, dtype=np.float64
    )
    fused = np.concatenate([partial_points, downsampled], axis=0)
    info = {
        "partial_points": int(len(partial_points)),
        "generated_input_points": int(len(generated_points)),
        "generated_removed_points": int(remove.sum()),
        "generated_kept_before_downsample": int(len(kept)),
        "generated_kept_after_downsample": int(len(downsampled)),
        "fused_points": int(len(fused)),
        "remove_ratio": float(remove_ratio),
        "voxel_ratio": float(voxel_ratio),
        "partial_preserved_exactly": True,
    }
    return fused, info


def draw_projection_overlay(
    path,
    image_path,
    partial_points,
    generated_points,
    projector,
):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = projector.image_shape
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    partial_uv, partial_depth = projector.project(partial_points)
    generated_uv, generated_depth = projector.project(generated_points)
    _, partial_mask, _ = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=1
    )
    _, generated_mask, _ = zbuffer_depth_with_indices(
        generated_uv, generated_depth, projector.image_shape, splat_radius=1
    )
    overlay = image.copy()
    overlay[partial_mask] = (
        0.55 * overlay[partial_mask] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay[generated_mask] = (
        0.55 * overlay[generated_mask] + 0.45 * np.array([0, 80, 255])
    ).astype(np.uint8)
    overlap = partial_mask & generated_mask
    overlay[overlap] = (
        0.35 * overlay[overlap] + 0.65 * np.array([0, 255, 255])
    ).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)


def candidate_summary(item):
    return {
        key: value
        for key, value in item.items()
        if key
        not in {
            "native_points",
            "native_normals",
            "raw_points",
            "raw_points_refined",
            "raw_normals",
        }
    }


def materialize_variant(output_root, variant, flag, points):
    path = Path(output_root) / variant / str(flag) / f"{flag}_fused.ply"
    write_points(path, points)
    return path


def load_config(path):
    cfg = Munch.fromDict(yaml.safe_load(Path(path).read_text()))
    cfg.paths = getattr(cfg, "paths", Munch())
    return normalize_runtime_config(cfg)


def ensure_seed_candidates(
    cfg,
    source_root,
    output_root,
    sample_ids,
    seeds,
    *,
    generate_missing,
):
    generated_root = Path(output_root) / "generated_candidates"
    paths = {}
    missing = []
    for flag in sample_ids:
        paths[str(flag)] = {}
        for seed in seeds:
            path = generated_root / str(flag) / f"{flag}_hunyuan2.1_seed{seed}.ply"
            paths[str(flag)][int(seed)] = path
            if not path.exists():
                missing.append((str(flag), int(seed), path))
    if missing and not generate_missing:
        preview = ", ".join(str(item[2]) for item in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} seed candidates; examples: {preview}"
        )
    if not missing:
        return paths

    from tools.hunyuan3d_2 import hunyuan3d_2, release_hunyuan3d_cache

    generation_cfg = copy.deepcopy(cfg)
    generation_cfg.paths.output_dir = str(generated_root)
    normalize_runtime_config(generation_cfg)
    generation_cfg.generative_model = "hunyuan2.1"
    for flag, seed, path in missing:
        source_image = Path(source_root) / flag / "img_sam.png"
        if not source_image.exists():
            raise FileNotFoundError(f"Hunyuan input image not found: {source_image}")
        generation_cfg.hunyuan_seed = int(seed)
        generation_cfg.hunyuan_output_ply_name = path.name
        np.random.seed(int(seed))
        o3d.utility.random.seed(int(seed))
        with Image.open(source_image) as image:
            hunyuan3d_2(generation_cfg, flag, image.copy())
        if not path.exists():
            raise FileNotFoundError(f"Hunyuan candidate was not written: {path}")
    release_hunyuan3d_cache()
    return paths


def validate_inputs(source_root, sample_ids, require_img_sam):
    missing = []
    for flag in sample_ids:
        sample_dir = Path(source_root) / str(flag)
        required = [
            PROJECT_ROOT / "data" / f"{flag}.ply",
            sample_dir / "camera.pth",
            sample_dir / "img.png",
            sample_dir / f"{flag}_hunyuan2.1.ply",
            sample_dir / f"{flag}_fused.ply",
            sample_dir / f"{flag}_moge_object_only.ply",
            sample_dir / f"{flag}_render_to_moge_sim3_info.json",
            sample_dir
            / f"{flag}_moge_to_raw_partial_moge_to_raw_partial_transform.npy",
        ]
        if require_img_sam:
            required.append(sample_dir / "img_sam.png")
        missing.extend(path for path in required if not path.exists())
    if missing:
        preview = "\n".join(str(path) for path in missing[:12])
        raise FileNotFoundError(
            f"Missing {len(missing)} required inputs:\n{preview}"
        )


def evaluate_candidate_geometry(
    flag,
    label,
    candidate_path,
    partial_points,
    partial_normals,
    projector,
    context,
    *,
    seed,
    top_k,
    eval_points_count,
):
    native_points = load_points(candidate_path)
    native_diagonal = max(
        float(np.linalg.norm(bbox_extent(native_points))), 1e-8
    )
    native_normals = estimate_normals(native_points, radius=0.03 * native_diagonal)
    eval_points = maybe_subsample(native_points, eval_points_count, seed)
    transforms = initial_candidates(
        eval_points,
        context["moge_points"],
        axis_aligned_rotations(),
        parse_float_list("0.45,0.55,0.65,0.75,0.85,1.0,1.15,1.3,1.45,1.6"),
    )
    ranked = rank_refined_sim3_candidates(
        transforms,
        eval_points,
        context["intrinsic_px"],
        context["target_depth"],
        context["target_mask"],
        top_k=top_k,
        splat_radius=1,
        translation_steps=(0.12, 0.06, 0.03, 0.015),
        rotation_steps_deg=(12.0, 6.0, 3.0),
        scale_steps=(1.12, 1.06, 1.03),
        rounds=1,
        require_2d_acceptance=True,
        min_iou=0.82,
        min_coverage=0.84,
        max_leakage=0.10,
        min_edge_iou=0.025,
        max_edge_chamfer_px=18.0,
    )
    results = []
    for item in ranked:
        complete_to_partial = compose_complete_to_partial(
            context["moge_to_partial"], item["transform"]
        )
        raw_points = apply_sim3(native_points, complete_to_partial)
        raw_normals = transform_normals(native_normals, complete_to_partial)
        diagonal = max(
            float(np.linalg.norm(bbox_extent(raw_points))), 1e-8
        )
        raw_gate, _, _ = evaluate_raw_partial_gate(
            partial_points,
            raw_points,
            projector,
            bbox_diagonal=diagonal,
            partial_normals=partial_normals,
            complete_normals=raw_normals,
        )
        results.append(
            {
                "candidate_label": label,
                "candidate_path": str(candidate_path),
                "candidate_index": int(item["index"]),
                "complete_to_moge": item["transform"],
                "complete_to_partial": complete_to_partial,
                "moge_score": item["score"],
                "moge_gate": item["acceptance"],
                "moge_objective": item["objective"],
                "raw_gate": raw_gate,
                "native_points": native_points,
                "native_normals": native_normals,
                "raw_points": raw_points,
                "raw_normals": raw_normals,
            }
        )
    return results


def process_sample(
    flag,
    source_root,
    output_root,
    seed_paths,
    *,
    seed,
    top_k,
    eval_points_count,
    graph_iterations,
    device,
):
    flag = str(flag)
    sample_dir = Path(source_root) / flag
    diagnostics_dir = Path(output_root) / "diagnostics" / flag
    diagnostics_path = diagnostics_dir / f"{flag}_registration_graph_fusion_info.json"
    baseline_points = load_points(sample_dir / f"{flag}_fused.ply")
    partial_points = load_points(PROJECT_ROOT / "data" / f"{flag}.ply")
    context = load_moge_context(sample_dir, flag)
    partial_diagonal = max(
        float(np.linalg.norm(bbox_extent(partial_points))), 1e-8
    )
    partial_normals = estimate_normals(
        partial_points, radius=0.03 * partial_diagonal
    )
    projector = SavedCameraProjector.from_partial(
        partial_points,
        sample_dir / "camera.pth",
        padding=0.15,
        image_shape=context["image_shape"],
        device="cpu",
    )

    outputs = {}
    outputs["sim3_baseline"] = materialize_variant(
        output_root, "sim3_baseline", flag, baseline_points
    )

    candidate_paths = [("current", sample_dir / f"{flag}_hunyuan2.1.ply")]
    candidate_paths.extend(
        (f"seed{seed_value}", path)
        for seed_value, path in sorted(seed_paths.items())
    )
    all_candidates = []
    for candidate_offset, (label, path) in enumerate(candidate_paths):
        all_candidates.extend(
            evaluate_candidate_geometry(
                flag,
                label,
                path,
                partial_points,
                partial_normals,
                projector,
                context,
                seed=seed + candidate_offset,
                top_k=top_k,
                eval_points_count=eval_points_count,
            )
        )

    # Preserve the current saved complete-to-MoGe candidate even if it did not
    # enter the new top-K shortlist.
    current_info = context["info"]
    current_native = load_points(candidate_paths[0][1])
    current_native_diagonal = max(
        float(np.linalg.norm(bbox_extent(current_native))), 1e-8
    )
    current_normals = estimate_normals(
        current_native, radius=0.03 * current_native_diagonal
    )
    saved_complete_to_moge = np.asarray(
        current_info["complete_to_moge"], dtype=np.float64
    )
    saved_complete_to_partial = compose_complete_to_partial(
        context["moge_to_partial"], saved_complete_to_moge
    )
    saved_raw = apply_sim3(current_native, saved_complete_to_partial)
    saved_normals = transform_normals(current_normals, saved_complete_to_partial)
    saved_diagonal = max(
        float(np.linalg.norm(bbox_extent(saved_raw))), 1e-8
    )
    saved_raw_gate, _, _ = evaluate_raw_partial_gate(
        partial_points,
        saved_raw,
        projector,
        bbox_diagonal=saved_diagonal,
        partial_normals=partial_normals,
        complete_normals=saved_normals,
    )
    saved_score = evaluate_transform(
        current_native,
        saved_complete_to_moge,
        context["intrinsic_px"],
        context["target_depth"],
        context["target_mask"],
        splat_radius=1,
    )
    saved_gate = evaluate_2d_acceptance(
        saved_score,
        enabled=True,
        min_iou=0.82,
        min_coverage=0.84,
        max_leakage=0.10,
        min_edge_iou=0.025,
        max_edge_chamfer_px=18.0,
    )
    all_candidates.append(
        {
            "candidate_label": "current_saved_transform",
            "candidate_path": str(candidate_paths[0][1]),
            "candidate_index": -1,
            "complete_to_moge": saved_complete_to_moge,
            "complete_to_partial": saved_complete_to_partial,
            "moge_score": saved_score,
            "moge_gate": saved_gate,
            "moge_objective": score_2d_gate_objective(saved_score),
            "raw_gate": saved_raw_gate,
            "native_points": current_native,
            "native_normals": current_normals,
            "raw_points": saved_raw,
            "raw_normals": saved_normals,
        }
    )

    passing = [
        item
        for item in all_candidates
        if item["moge_gate"]["accepted"] and item["raw_gate"]["accepted"]
    ]
    passing.sort(
        key=lambda item: (
            item["raw_gate"]["cost"],
            -item["moge_objective"],
            item["candidate_label"],
            item["candidate_index"],
        )
    )
    registration_failed = not passing
    if registration_failed:
        reranked_points = baseline_points.copy()
        selected = None
        refinement_info = {
            "accepted": False,
            "reason": "no_candidate_passed_both_gates",
        }
    else:
        selected = passing[0]
        selected_transform, reranked_points, refinement_info = (
            bounded_raw_sim3_refinement(
                selected["native_points"],
                selected["native_normals"],
                selected["complete_to_partial"],
                partial_points,
                partial_normals,
                projector,
                context,
                bbox_diagonal=max(
                    float(np.linalg.norm(bbox_extent(selected["raw_points"]))),
                    1e-8,
                ),
            )
        )
        selected["complete_to_partial_refined"] = selected_transform
        selected["raw_points_refined"] = reranked_points
    outputs["raw_partial_rerank"] = materialize_variant(
        output_root, "raw_partial_rerank", flag, reranked_points
    )

    if registration_failed:
        graph_se3_points = baseline_points.copy()
        graph_scale_points = baseline_points.copy()
        fused_points = baseline_points.copy()
        graph_se3_info = {
            "accepted": False,
            "reason": "registration_failed",
        }
        graph_scale_info = {
            "accepted": False,
            "reason": "registration_failed",
        }
        fusion_info = {
            "accepted": False,
            "reason": "registration_failed",
        }
    else:
        reranked_diagonal = max(
            float(np.linalg.norm(bbox_extent(reranked_points))), 1e-8
        )
        reranked_normals = estimate_normals(
            reranked_points, radius=0.03 * reranked_diagonal
        )
        baseline_gate, _, _ = evaluate_raw_partial_gate(
            partial_points,
            reranked_points,
            projector,
            bbox_diagonal=reranked_diagonal,
            partial_normals=partial_normals,
            complete_normals=reranked_normals,
        )
        graph = build_deformation_graph(reranked_points)
        graph_se3_points, graph_se3_info, se3_state = optimize_deformation_graph(
            graph,
            reranked_points,
            partial_points,
            partial_normals,
            projector,
            context,
            baseline_gate,
            allow_scale=False,
            iterations=graph_iterations,
            device=device,
            required_mean_improvement=0.02,
            seed=seed,
        )
        if graph_se3_info["accepted"]:
            scale_baseline = graph_se3_info["after_raw_gate"]
            graph_scale_points, graph_scale_info, _ = optimize_deformation_graph(
                graph,
                graph_se3_points,
                partial_points,
                partial_normals,
                projector,
                context,
                scale_baseline,
                allow_scale=True,
                initial_state=se3_state,
                iterations=graph_iterations,
                device=device,
                required_mean_improvement=0.01,
                seed=seed + 1,
            )
            if not graph_scale_info["accepted"]:
                graph_scale_points = graph_se3_points.copy()
        else:
            graph_scale_points = reranked_points.copy()
            graph_scale_info = {
                "accepted": False,
                "reason": "se3_deformation_rejected",
            }
        fused_points, fusion_info = partial_priority_fusion(
            graph_scale_points,
            partial_points,
            projector,
            bbox_diagonal=max(
                float(np.linalg.norm(bbox_extent(graph_scale_points))), 1e-8
            ),
        )
        fusion_info["accepted"] = True
        fusion_info["reason"] = "partial_priority_fusion_written"

    outputs["graph_se3"] = materialize_variant(
        output_root, "graph_se3", flag, graph_se3_points
    )
    outputs["graph_scale"] = materialize_variant(
        output_root, "graph_scale", flag, graph_scale_points
    )
    outputs["graph_scale_partial_priority_fusion"] = materialize_variant(
        output_root,
        "graph_scale_partial_priority_fusion",
        flag,
        fused_points,
    )

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    write_compare(
        diagnostics_dir / f"{flag}_partial_gray_reranked_red.ply",
        partial_points,
        reranked_points,
    )
    draw_projection_overlay(
        diagnostics_dir / f"{flag}_raw_partial_projection_overlay.png",
        sample_dir / "img.png",
        partial_points,
        reranked_points,
        projector,
    )
    if selected is not None:
        np.save(
            diagnostics_dir / f"{flag}_selected_complete_to_partial.npy",
            selected.get(
                "complete_to_partial_refined", selected["complete_to_partial"]
            ),
        )
    info = {
        "flag": flag,
        "method": "multi_seed_raw_partial_gate_deformation_graph_partial_priority_fusion",
        "inputs": {
            "source_root": str(source_root),
            "partial": str(PROJECT_ROOT / "data" / f"{flag}.ply"),
            "baseline": str(sample_dir / f"{flag}_fused.ply"),
            "camera": str(sample_dir / "camera.pth"),
        },
        "registration_failed": registration_failed,
        "candidate_count": int(len(all_candidates)),
        "passing_candidate_count": int(len(passing)),
        "selected_candidate": candidate_summary(selected) if selected else None,
        "bounded_refinement": refinement_info,
        "candidates": [candidate_summary(item) for item in all_candidates],
        "graph_se3": graph_se3_info,
        "graph_scale": graph_scale_info,
        "fusion": fusion_info,
        "outputs": outputs,
    }
    save_json(diagnostics_path, info)
    return info


def run_official_metrics(cfg, output_root, sample_ids, sample_infos):
    from main import metric, resolve_prompt_label, write_metric_results

    summary_rows = []
    combined_rows = []
    failure_count = sum(
        bool(info.get("registration_failed")) for info in sample_infos.values()
    )
    for variant in VARIANTS:
        metric_cfg = copy.deepcopy(cfg)
        metric_cfg.paths.output_dir = str(Path(output_root) / variant)
        normalize_runtime_config(metric_cfg)
        results = []
        for flag in sample_ids:
            cd, emd = metric(str(flag), metric_cfg)
            result = {
                "sample_id": str(flag),
                "flag": resolve_prompt_label(str(flag), metric_cfg),
                "cd": float(cd),
                "emd": float(emd),
            }
            results.append(result)
            combined_rows.append(
                {
                    "variant": variant,
                    "sample_id": str(flag),
                    "cd_l1_x1e2": float(cd) * 100.0,
                    "emd_x1e2": float(emd) * 100.0,
                    "registration_failed": bool(
                        sample_infos[str(flag)].get("registration_failed")
                    ),
                }
            )
        write_metric_results(metric_cfg, results)
        summary_rows.append(
            {
                "variant": variant,
                "samples": len(results),
                "registration_failures": failure_count
                if variant != "sim3_baseline"
                else 0,
                "mean_cd_l1_x1e2": 100.0
                * float(np.mean([item["cd"] for item in results])),
                "mean_emd_x1e2": 100.0
                * float(np.mean([item["emd"] for item in results])),
            }
        )

    summary_path = Path(output_root) / "metrics_ablation_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    combined_path = Path(output_root) / "metrics_ablation_samples.csv"
    with combined_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 102, 103])
    parser.add_argument(
        "--generate-missing-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-metrics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--eval-points", type=int, default=60000)
    parser.add_argument("--graph-iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=6145)
    return parser.parse_args()


def run(args):
    cfg = load_config(args.config)
    sample_ids = [
        str(value)
        for value in (
            args.samples
            if args.samples
            else getattr(cfg, "sample_ids", [])
        )
    ]
    if not sample_ids:
        raise ValueError("No sample ids configured")
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    validate_inputs(
        source_root,
        sample_ids,
        require_img_sam=bool(args.generate_missing_seeds),
    )
    seed_paths = ensure_seed_candidates(
        cfg,
        source_root,
        output_root,
        sample_ids,
        args.seeds,
        generate_missing=bool(args.generate_missing_seeds),
    )

    sample_infos = {}
    for flag in sample_ids:
        diagnostics_path = (
            output_root
            / "diagnostics"
            / flag
            / f"{flag}_registration_graph_fusion_info.json"
        )
        expected = [
            output_root / variant / flag / f"{flag}_fused.ply"
            for variant in VARIANTS
        ]
        if (
            bool(args.skip_existing)
            and diagnostics_path.exists()
            and all(path.exists() for path in expected)
        ):
            print(f"Skip completed ablation sample {flag}")
            sample_infos[flag] = json.loads(diagnostics_path.read_text())
            continue
        print(f"Running registration/deformation/fusion ablation for {flag}")
        sample_infos[flag] = process_sample(
            flag,
            source_root,
            output_root,
            seed_paths[flag],
            seed=args.seed,
            top_k=args.top_k,
            eval_points_count=args.eval_points,
            graph_iterations=args.graph_iterations,
            device=args.device,
        )

    batch_info = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "samples": sample_ids,
        "seeds": [int(value) for value in args.seeds],
        "registration_failures": [
            flag
            for flag, info in sample_infos.items()
            if info.get("registration_failed")
        ],
        "variants": list(VARIANTS),
    }
    save_json(output_root / "batch_info.json", batch_info)
    if bool(args.run_metrics):
        batch_info["metrics"] = run_official_metrics(
            cfg, output_root, sample_ids, sample_infos
        )
        save_json(output_root / "batch_info.json", batch_info)
    print(json.dumps(jsonable(batch_info), indent=2))
    return batch_info


if __name__ == "__main__":
    run(parse_args())

