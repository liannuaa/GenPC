import argparse
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_moge_pixel_index_bridge import (  # noqa: E402
    filter_moge_points_by_object_mask,
    load_alpha_mask,
    run_moge_with_pixels,
    write_pcd,
)


DEFAULT_SAMPLE_ROOT = PROJECT_ROOT / "workspace" / "redwood_stage1_qwen_refine_preview"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "workspace"
DEFAULT_MOGE_MODEL = PROJECT_ROOT / "models" / "moge-2-vitl"


@dataclass(frozen=True)
class RenderToMogePaths:
    sample_dir: Path
    out_dir: Path
    image_path: Path
    complete_path: Path
    object_mask_path: Path
    moge_to_partial_path: Path


def infer_paths(
    *,
    flag,
    sample_root,
    sample_dir,
    out_root,
    out_dir,
    image_name,
    complete_name,
    object_mask_name,
    moge_to_partial_name,
):
    flag = str(flag)
    sample_dir = Path(sample_dir) if sample_dir else Path(sample_root) / flag
    out_dir = Path(out_dir) if out_dir else Path(out_root) / f"render_to_moge_sim3_{flag}"
    image_name = image_name or "img.png"
    complete_name = complete_name or f"{flag}_hunyuan2.1.ply"
    object_mask_name = object_mask_name or f"{flag}_moge_to_raw_partial_object_mask.png"
    moge_to_partial_name = (
        moge_to_partial_name or f"{flag}_moge_to_raw_partial_moge_to_raw_partial_transform.npy"
    )
    return RenderToMogePaths(
        sample_dir=sample_dir,
        out_dir=out_dir,
        image_path=sample_dir / image_name,
        complete_path=sample_dir / complete_name,
        object_mask_path=sample_dir / object_mask_name,
        moge_to_partial_path=sample_dir / moge_to_partial_name,
    )


def make_sim3(scale, rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def apply_sim3(points, transform):
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    hom = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (transform @ hom.T).T[:, :3]


def compose_complete_to_partial(moge_to_partial, complete_to_moge):
    return np.asarray(moge_to_partial, dtype=np.float64) @ np.asarray(
        complete_to_moge, dtype=np.float64
    )


def decompose_sim3(transform):
    transform = np.asarray(transform, dtype=np.float64)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    if abs(scale) < 1e-12:
        raise ValueError("Cannot decompose transform with near-zero scale")
    rotation = transform[:3, :3] / scale
    translation = transform[:3, 3].copy()
    return scale, rotation, translation


def zbuffer_depth(uv, depth, image_shape, splat_radius=1):
    height, width = int(image_shape[0]), int(image_shape[1])
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    rendered = np.full((height, width), np.inf, dtype=np.float64)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
    if not valid.any():
        return rendered, np.zeros((height, width), dtype=bool)

    xy = np.rint(uv[valid]).astype(np.int64)
    z = depth[valid]
    radius = max(0, int(splat_radius))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            xx = xy[:, 0] + dx
            yy = xy[:, 1] + dy
            keep = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            if keep.any():
                np.minimum.at(rendered, (yy[keep], xx[keep]), z[keep])

    mask = np.isfinite(rendered)
    rendered[~mask] = 0.0
    return rendered, mask


def mask_boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask & ~eroded


def boundary_chamfer_pixels(source_boundary, target_boundary):
    source_boundary = np.asarray(source_boundary, dtype=bool)
    target_boundary = np.asarray(target_boundary, dtype=bool)
    if not source_boundary.any() or not target_boundary.any():
        return float("inf")
    distance = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[source_boundary].mean())


def score_depth_render(rendered_depth, rendered_mask, target_depth, target_mask):
    rendered_mask = np.asarray(rendered_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    target_depth = np.asarray(target_depth, dtype=np.float64)
    rendered_depth = np.asarray(rendered_depth, dtype=np.float64)

    intersection = rendered_mask & target_mask
    union = rendered_mask | target_mask
    iou = float(intersection.sum() / max(int(union.sum()), 1))
    coverage = float(intersection.sum() / max(int(target_mask.sum()), 1))
    leakage = float((rendered_mask & ~target_mask).sum() / max(int(rendered_mask.sum()), 1))
    rendered_edge = mask_boundary(rendered_mask)
    target_edge = mask_boundary(target_mask)
    edge_intersection = rendered_edge & target_edge
    edge_union = rendered_edge | target_edge
    edge_iou = float(edge_intersection.sum() / max(int(edge_union.sum()), 1))
    forward_edge_chamfer = boundary_chamfer_pixels(rendered_edge, target_edge)
    backward_edge_chamfer = boundary_chamfer_pixels(target_edge, rendered_edge)
    if np.isfinite(forward_edge_chamfer) and np.isfinite(backward_edge_chamfer):
        edge_chamfer = 0.5 * (forward_edge_chamfer + backward_edge_chamfer)
        edge_chamfer_norm = float(edge_chamfer / max(np.linalg.norm(target_mask.shape), 1.0))
    else:
        edge_chamfer = float("inf")
        edge_chamfer_norm = 1.0

    if intersection.any():
        depth_error = np.abs(rendered_depth[intersection] - target_depth[intersection])
        target_span = max(
            float(np.percentile(target_depth[target_mask], 95) - np.percentile(target_depth[target_mask], 5)),
            1e-6,
        )
        depth_mae = float(np.mean(depth_error))
        depth_p95 = float(np.percentile(depth_error, 95))
        depth_norm = depth_mae / target_span
    else:
        depth_mae = float("inf")
        depth_p95 = float("inf")
        depth_norm = 10.0

    score = float(
        1.25 * iou
        + 0.55 * coverage
        + 0.35 * edge_iou
        - 0.85 * leakage
        - 0.45 * edge_chamfer_norm
        - 0.25 * min(depth_norm, 10.0)
    )
    return {
        "score": score,
        "iou": iou,
        "coverage": coverage,
        "leakage": leakage,
        "edge_iou": edge_iou,
        "edge_chamfer_px": edge_chamfer,
        "edge_chamfer_norm": edge_chamfer_norm,
        "depth_mae": depth_mae,
        "depth_p95": depth_p95,
        "rendered_pixels": int(rendered_mask.sum()),
        "target_pixels": int(target_mask.sum()),
        "intersection_pixels": int(intersection.sum()),
        "rendered_edge_pixels": int(rendered_edge.sum()),
        "target_edge_pixels": int(target_edge.sum()),
    }


def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd, points


def normalized_intrinsic_to_pixel(intrinsic, image_shape):
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    height, width = int(image_shape[0]), int(image_shape[1])
    pixel = intrinsic.copy()
    pixel[0, :] *= width
    pixel[1, :] *= height
    pixel[2, :] = [0.0, 0.0, 1.0]
    return pixel


def project_points(points, intrinsic_px, image_shape):
    points = np.asarray(points, dtype=np.float64)
    intrinsic_px = np.asarray(intrinsic_px, dtype=np.float64)
    height, width = int(image_shape[0]), int(image_shape[1])
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-8)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if valid.any():
        projected = (intrinsic_px @ points[valid].T).T
        uv[valid] = projected[:, :2] / projected[:, 2:3]
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid


def target_depth_from_moge(moge_points, moge_pixel_xy, image_shape):
    return zbuffer_depth(
        uv=np.asarray(moge_pixel_xy, dtype=np.float64),
        depth=np.asarray(moge_points, dtype=np.float64)[:, 2],
        image_shape=image_shape,
        splat_radius=0,
    )


def axis_aligned_rotations():
    rotations = []
    for perm in permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        for row, col in enumerate(perm):
            base[row, col] = 1.0
        for signs in product([-1.0, 1.0], repeat=3):
            rot = base * np.asarray(signs, dtype=np.float64)[:, None]
            if np.linalg.det(rot) > 0.5:
                rotations.append(rot)
    return rotations


def parse_float_list(value):
    return [float(item) for item in str(value).split(",") if item.strip()]


def parse_scale_triplets(value):
    triplets = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(part) for part in item.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Expected scale triplet 'sx,sy,sz', got: {item}")
        triplets.append(tuple(parts))
    return triplets


def maybe_subsample(points, max_points, seed):
    points = np.asarray(points, dtype=np.float64)
    if max_points is None or len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(max_points), replace=False)]


def bbox_extent(points):
    points = np.asarray(points, dtype=np.float64)
    return points.max(axis=0) - points.min(axis=0)


def evaluate_transform(points, transform, intrinsic_px, target_depth, target_mask, splat_radius):
    moved = apply_sim3(points, transform)
    uv, valid = project_points(moved, intrinsic_px, target_depth.shape)
    rendered_depth, rendered_mask = zbuffer_depth(
        uv[valid],
        moved[valid, 2],
        image_shape=target_depth.shape,
        splat_radius=splat_radius,
    )
    return score_depth_render(rendered_depth, rendered_mask, target_depth, target_mask)


def initial_candidates(source_points, target_points, rotations, scale_multipliers):
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    target_diag = max(float(np.linalg.norm(bbox_extent(target_points))), 1e-8)
    candidates = []
    for rotation in rotations:
        rotated = (rotation @ source_points.T).T
        source_diag = max(float(np.linalg.norm(bbox_extent(rotated))), 1e-8)
        base_scale = target_diag / source_diag
        for mult in scale_multipliers:
            scale = base_scale * float(mult)
            translation = target_center - scale * (rotation @ source_center)
            candidates.append(make_sim3(scale, rotation, translation))
    return candidates


def select_best_candidate(
    candidates,
    source_points,
    intrinsic_px,
    target_depth,
    target_mask,
    splat_radius,
):
    best = None
    for index, transform in enumerate(candidates):
        score = evaluate_transform(
            source_points,
            transform,
            intrinsic_px,
            target_depth,
            target_mask,
            splat_radius=splat_radius,
        )
        item = {"index": int(index), "transform": transform, "score": score}
        if best is None or score["score"] > best["score"]["score"]:
            best = item
    return best


def refine_transform_coordinate_search(
    initial_transform,
    source_points,
    intrinsic_px,
    target_depth,
    target_mask,
    *,
    splat_radius,
    translation_steps,
    rotation_steps_deg,
    scale_steps,
    rounds,
    selection_objective="score",
):
    def objective_value(score):
        if selection_objective == "2d_gate":
            return score_2d_gate_objective(score)
        return float(score.get("score", float("-inf")))

    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    best_score = evaluate_transform(
        source_points,
        transform,
        intrinsic_px,
        target_depth,
        target_mask,
        splat_radius=splat_radius,
    )
    history = [{"round": -1, "score": best_score}]

    for round_idx in range(int(rounds)):
        improved = True
        while improved:
            improved = False
            scale, rotation, translation = decompose_sim3(transform)
            proposals = []
            for step in translation_steps:
                for axis in range(3):
                    delta = np.zeros(3, dtype=np.float64)
                    delta[axis] = float(step)
                    proposals.append(make_sim3(scale, rotation, translation + delta))
                    proposals.append(make_sim3(scale, rotation, translation - delta))
            for step in scale_steps:
                proposals.append(make_sim3(scale * float(step), rotation, translation))
                proposals.append(make_sim3(scale / float(step), rotation, translation))
            for degrees in rotation_steps_deg:
                radians = math.radians(float(degrees))
                for axis in np.eye(3, dtype=np.float64):
                    for sign in (-1.0, 1.0):
                        delta_rot = Rotation.from_rotvec(axis * radians * sign).as_matrix()
                        proposals.append(make_sim3(scale, delta_rot @ rotation, translation))

            local_best_transform = transform
            local_best_score = best_score
            local_best_value = objective_value(best_score)
            for proposal in proposals:
                score = evaluate_transform(
                    source_points,
                    proposal,
                    intrinsic_px,
                    target_depth,
                    target_mask,
                    splat_radius=splat_radius,
                )
                value = objective_value(score)
                if value > local_best_value:
                    local_best_score = score
                    local_best_value = value
                    local_best_transform = proposal
            if local_best_value > objective_value(best_score) + 1e-8:
                transform = local_best_transform
                best_score = local_best_score
                improved = True
        history.append({"round": int(round_idx), "score": best_score})
    return transform, best_score, history


def visible_source_indices(points, transform, intrinsic_px, image_shape, target_mask, max_depth_delta):
    moved = apply_sim3(points, transform)
    uv, valid = project_points(moved, intrinsic_px, image_shape)
    if not valid.any():
        return np.empty(0, dtype=np.int64), moved
    valid_indices = np.where(valid)[0]
    xy = np.rint(uv[valid]).astype(np.int64)
    height, width = image_shape
    inside_mask = target_mask[xy[:, 1], xy[:, 0]]
    valid_indices = valid_indices[inside_mask]
    xy = xy[inside_mask]
    z = moved[valid_indices, 2]
    if len(valid_indices) == 0:
        return valid_indices, moved

    flat = xy[:, 1] * width + xy[:, 0]
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    first = np.r_[True, flat_sorted[1:] != flat_sorted[:-1]]
    front_order = order[first]
    front_indices = valid_indices[front_order]
    if max_depth_delta > 0:
        front_z_by_flat = dict(zip(flat[front_order].tolist(), z[front_order].tolist()))
        keep = np.array(
            [abs(float(zz) - front_z_by_flat[int(ff)]) <= float(max_depth_delta) for zz, ff in zip(z, flat)],
            dtype=bool,
        )
        return valid_indices[keep], moved
    return front_indices, moved


def umeyama_similarity(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 4:
        raise ValueError("Need at least four correspondences")
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    cov = (tgt_centered.T @ src_centered) / len(source)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular) @ sign) / max(variance, 1e-12))
    translation = tgt_mean - scale * rotation @ src_mean
    return make_sim3(scale, rotation, translation)


def refine_visible_icp(
    transform,
    source_points,
    target_points,
    intrinsic_px,
    image_shape,
    target_mask,
    *,
    iterations,
    trim_quantile,
    max_pairs,
    seed,
):
    rng = np.random.default_rng(int(seed))
    target_tree = cKDTree(target_points)
    current = np.asarray(transform, dtype=np.float64).copy()
    history = []
    for iteration in range(int(iterations)):
        visible_indices, moved = visible_source_indices(
            source_points,
            current,
            intrinsic_px,
            image_shape,
            target_mask,
            max_depth_delta=0.015,
        )
        if len(visible_indices) < 16:
            break
        if len(visible_indices) > int(max_pairs):
            visible_indices = rng.choice(visible_indices, size=int(max_pairs), replace=False)
        distances, nearest = target_tree.query(moved[visible_indices], k=1)
        threshold = np.quantile(distances, float(trim_quantile))
        keep = distances <= threshold
        if int(keep.sum()) < 16:
            break
        delta = umeyama_similarity(moved[visible_indices][keep], target_points[nearest[keep]])
        current = delta @ current
        history.append(
            {
                "iteration": int(iteration),
                "visible_points": int(len(visible_indices)),
                "kept_pairs": int(keep.sum()),
                "mean_distance": float(distances[keep].mean()),
                "p95_distance": float(np.percentile(distances[keep], 95)),
            }
        )
    return current, history


def choose_icp_refinement(
    coordinate_refined,
    refined_score,
    icp_refined,
    icp_score,
    *,
    rollback_on_score_drop=True,
    min_score_gain=0.0,
):
    min_score_gain = float(min_score_gain)
    if bool(rollback_on_score_drop) and icp_score["score"] < refined_score["score"] + min_score_gain:
        return (
            np.asarray(coordinate_refined, dtype=np.float64),
            refined_score,
            {
                "accepted": False,
                "reason": "render_score_drop",
                "min_score_gain": min_score_gain,
                "baseline_score": refined_score,
                "candidate_score": icp_score,
            },
        )
    return (
        np.asarray(icp_refined, dtype=np.float64),
        icp_score,
        {
            "accepted": True,
            "reason": "render_score_preserved",
            "min_score_gain": min_score_gain,
            "baseline_score": refined_score,
            "candidate_score": icp_score,
        },
    )


def evaluate_2d_acceptance(
    score,
    *,
    enabled,
    min_iou,
    min_coverage,
    max_leakage,
    min_edge_iou,
    max_edge_chamfer_px,
):
    score = dict(score or {})
    thresholds = {
        "min_iou": float(min_iou),
        "min_coverage": float(min_coverage),
        "max_leakage": float(max_leakage),
        "min_edge_iou": float(min_edge_iou),
        "max_edge_chamfer_px": float(max_edge_chamfer_px),
    }
    values = {
        "iou": float(score.get("iou", float("-inf"))),
        "coverage": float(score.get("coverage", float("-inf"))),
        "leakage": float(score.get("leakage", float("inf"))),
        "edge_iou": float(score.get("edge_iou", float("-inf"))),
        "edge_chamfer_px": float(score.get("edge_chamfer_px", float("inf"))),
    }
    if not bool(enabled):
        return {
            "enabled": False,
            "accepted": True,
            "reason": "disabled",
            "thresholds": thresholds,
            "values": values,
            "failed": [],
            "warnings": [],
        }

    failed = []
    if values["iou"] < thresholds["min_iou"]:
        failed.append("iou")
    if values["coverage"] < thresholds["min_coverage"]:
        failed.append("coverage")
    if values["leakage"] > thresholds["max_leakage"]:
        failed.append("leakage")
    if values["edge_iou"] < thresholds["min_edge_iou"]:
        failed.append("edge_iou")
    if values["edge_chamfer_px"] > thresholds["max_edge_chamfer_px"]:
        failed.append("edge_chamfer_px")

    return {
        "enabled": True,
        "accepted": not failed,
        "reason": "2d_threshold_met" if not failed else "2d_threshold_not_met",
        "thresholds": thresholds,
        "values": values,
        "failed": failed,
        "warnings": [],
    }


def score_2d_gate_objective(score):
    score = dict(score or {})
    return float(
        2.0 * float(score.get("iou", 0.0))
        + float(score.get("coverage", 0.0))
        - 2.0 * float(score.get("leakage", 1.0))
        + 3.0 * float(score.get("edge_iou", 0.0))
        - 0.025 * float(score.get("edge_chamfer_px", 100.0))
    )


def score_2d_anchor_objective(score, anchor_stats, *, anchor_scale, anchor_weight):
    anchor_stats = dict(anchor_stats or {})
    anchor_to_complete = dict(anchor_stats.get("anchor_to_complete") or {})
    anchor_mean = float(anchor_to_complete.get("mean", float("inf")))
    if not np.isfinite(anchor_mean):
        return score_2d_gate_objective(score)
    anchor_scale = max(float(anchor_scale), 1e-6)
    return float(score_2d_gate_objective(score) - float(anchor_weight) * (anchor_mean / anchor_scale))


def choose_visible_3d_refinement(
    baseline_transform,
    baseline_score,
    candidate_transform,
    candidate_score,
    optimization_info,
    *,
    max_score_drop,
    min_distance_improvement,
):
    info = dict(optimization_info)
    max_score_drop = float(max_score_drop)
    min_distance_improvement = float(min_distance_improvement)
    initial_distance = info.get("initial_distance") or {}
    candidate_distance = info.get("candidate_distance") or {}
    initial_mean = float(initial_distance.get("mean", float("inf")))
    candidate_mean = float(candidate_distance.get("mean", float("inf")))
    required_score = float(baseline_score["score"]) - max_score_drop
    required_distance = initial_mean * (1.0 - min_distance_improvement)

    score_preserved = float(candidate_score["score"]) >= required_score
    distance_improved = candidate_mean <= required_distance
    info.update(
        {
            "max_score_drop": max_score_drop,
            "min_distance_improvement": min_distance_improvement,
            "required_score": required_score,
            "required_distance_mean": required_distance,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
        }
    )
    if score_preserved and distance_improved:
        info.update({"accepted": True, "reason": "render_score_preserved_visible_distance_improved"})
        return np.asarray(candidate_transform, dtype=np.float64), candidate_score, info

    if not score_preserved:
        reason = "render_score_drop"
    elif not distance_improved:
        reason = "visible_3d_distance_not_improved"
    else:
        reason = "rejected"
    info.update({"accepted": False, "reason": reason})
    return np.asarray(baseline_transform, dtype=np.float64), baseline_score, info


def torch_rodrigues(rotvec):
    theta = torch.linalg.norm(rotvec) + 1e-8
    x, y, z = rotvec
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + (torch.sin(theta) / theta) * k + ((1.0 - torch.cos(theta)) / (theta * theta)) * (k @ k)


def resize_target_mask(mask, render_size):
    mask = np.asarray(mask, dtype=np.float32)
    resized = cv2.resize(mask, (int(render_size), int(render_size)), interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def render_soft_silhouette_and_depth_torch(
    points,
    intrinsic_px,
    image_shape,
    render_size,
    splat_radius,
    sigma,
    opacity,
):
    height, width = int(image_shape[0]), int(image_shape[1])
    render_size = int(render_size)
    z = points[:, 2]
    valid = torch.isfinite(points).all(dim=1) & (z > 1e-6)
    projected = points @ intrinsic_px.T
    uv = projected[:, :2] / torch.clamp(projected[:, 2:3], min=1e-6)
    uv = uv * torch.tensor(
        [render_size / max(width, 1), render_size / max(height, 1)],
        dtype=points.dtype,
        device=points.device,
    )
    valid = valid & (uv[:, 0] >= 0) & (uv[:, 0] <= render_size - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= render_size - 1)
    uv = uv[valid]
    z = z[valid]
    if uv.numel() == 0:
        empty = torch.zeros((render_size, render_size), dtype=points.dtype, device=points.device)
        return empty, empty

    base = torch.floor(uv).long()
    flat_size = render_size * render_size
    density = torch.zeros(flat_size, dtype=points.dtype, device=points.device)
    weighted_depth = torch.zeros(flat_size, dtype=points.dtype, device=points.device)
    radius = max(1, int(splat_radius))
    sigma_sq = max(float(sigma) ** 2, 1e-6)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xy = base + torch.tensor([dx, dy], dtype=torch.long, device=points.device)
            keep = (xy[:, 0] >= 0) & (xy[:, 0] < render_size) & (xy[:, 1] >= 0) & (xy[:, 1] < render_size)
            if not bool(keep.any()):
                continue
            center = xy[keep].to(points.dtype) + 0.5
            dist2 = torch.sum((uv[keep] - center) ** 2, dim=1)
            weights = torch.exp(-0.5 * dist2 / sigma_sq)
            indices = xy[keep, 1] * render_size + xy[keep, 0]
            density.index_add_(0, indices, weights)
            weighted_depth.index_add_(0, indices, weights * z[keep])
    density = density.reshape(render_size, render_size)
    weighted_depth = weighted_depth.reshape(render_size, render_size)
    silhouette = 1.0 - torch.exp(-float(opacity) * density)
    depth = weighted_depth / torch.clamp(density, min=1e-6)
    return silhouette, depth


def render_soft_silhouette_torch(points, intrinsic_px, image_shape, render_size, splat_radius, sigma, opacity):
    silhouette, _ = render_soft_silhouette_and_depth_torch(
        points,
        intrinsic_px,
        image_shape,
        render_size,
        splat_radius,
        sigma,
        opacity,
    )
    return silhouette


def optimize_silhouette_delta_sim3(
    initial_transform,
    source_points,
    intrinsic_px,
    image_shape,
    target_mask,
    target_depth=None,
    *,
    render_size,
    max_points,
    iterations,
    lr,
    splat_radius,
    sigma,
    opacity,
    leakage_weight,
    miss_weight,
    outside_distance_weight,
    depth_weight=0.0,
    boundary_weight=0.0,
    area_weight,
    center_weight,
    transform_reg_weight,
    seed,
    device,
):
    if int(iterations) <= 0:
        return np.asarray(initial_transform, dtype=np.float64), {"enabled": False}

    rng = np.random.default_rng(int(seed))
    points = np.asarray(source_points, dtype=np.float64)
    if len(points) > int(max_points):
        points = points[rng.choice(len(points), size=int(max_points), replace=False)]

    base_points = apply_sim3(points, initial_transform)
    target_np = resize_target_mask(target_mask, render_size)
    outside_distance = cv2.distanceTransform((target_np <= 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    outside_distance = outside_distance / max(float(np.linalg.norm(target_np.shape)), 1.0)
    target_edge_np = mask_boundary(target_np > 0.5).astype(np.float32)
    target_edge_distance = cv2.distanceTransform((target_edge_np <= 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    target_edge_distance = target_edge_distance / max(float(np.linalg.norm(target_np.shape)), 1.0)
    if target_depth is not None and float(depth_weight) > 0.0:
        target_depth_np = cv2.resize(
            np.asarray(target_depth, dtype=np.float32),
            (int(render_size), int(render_size)),
            interpolation=cv2.INTER_AREA,
        )
        valid_depth_np = (target_np > 0.5) & np.isfinite(target_depth_np) & (target_depth_np > 0.0)
    else:
        target_depth_np = np.zeros_like(target_np, dtype=np.float32)
        valid_depth_np = np.zeros_like(target_np, dtype=bool)

    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    base = torch.as_tensor(base_points, dtype=dtype, device=torch_device)
    intrinsic = torch.as_tensor(np.asarray(intrinsic_px, dtype=np.float32), dtype=dtype, device=torch_device)
    target = torch.as_tensor(target_np, dtype=dtype, device=torch_device)
    outside = torch.as_tensor(outside_distance.astype(np.float32), dtype=dtype, device=torch_device)
    target_edge = torch.as_tensor(target_edge_np.astype(np.float32), dtype=dtype, device=torch_device)
    target_edge_dist = torch.as_tensor(target_edge_distance.astype(np.float32), dtype=dtype, device=torch_device)
    target_depth_t = torch.as_tensor(target_depth_np.astype(np.float32), dtype=dtype, device=torch_device)
    valid_depth = torch.as_tensor(valid_depth_np.astype(np.float32), dtype=dtype, device=torch_device)
    target_sum = torch.clamp(target.sum(), min=1.0)
    target_edge_sum = torch.clamp(target_edge.sum(), min=1.0)
    valid_depth_sum = torch.clamp(valid_depth.sum(), min=1.0)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, int(render_size), dtype=dtype, device=torch_device),
        torch.linspace(0.0, 1.0, int(render_size), dtype=dtype, device=torch_device),
        indexing="ij",
    )
    target_cx = (target * xx).sum() / target_sum
    target_cy = (target * yy).sum() / target_sum

    log_scale = torch.nn.Parameter(torch.zeros((), dtype=dtype, device=torch_device))
    rotvec = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    translation = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    optimizer = torch.optim.Adam([log_scale, rotvec, translation], lr=float(lr))

    best = {
        "loss": float("inf"),
        "log_scale": 0.0,
        "rotvec": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
    }
    eps = 1e-6
    for iteration in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale)
        rotation = torch_rodrigues(rotvec)
        moved = scale * (base @ rotation.T) + translation
        silhouette, rendered_depth = render_soft_silhouette_and_depth_torch(
            moved,
            intrinsic,
            image_shape,
            render_size=render_size,
            splat_radius=splat_radius,
            sigma=sigma,
            opacity=opacity,
        )
        pred_sum = torch.clamp(silhouette.sum(), min=eps)
        intersection = (silhouette * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        leakage = (silhouette * (1.0 - target)).sum() / pred_sum
        miss = (target * (1.0 - silhouette)).sum() / target_sum
        outside_loss = (silhouette * outside).sum() / pred_sum
        if float(boundary_weight) > 0.0 and bool(target_edge_np.any()):
            dx = torch.nn.functional.pad(torch.abs(silhouette[:, 1:] - silhouette[:, :-1]), (0, 1, 0, 0))
            dy = torch.nn.functional.pad(torch.abs(silhouette[1:, :] - silhouette[:-1, :]), (0, 0, 0, 1))
            pred_edge = torch.clamp(dx + dy, min=0.0)
            pred_edge_sum = torch.clamp(pred_edge.sum(), min=eps)
            pred_to_target_edge = (pred_edge * target_edge_dist).sum() / pred_edge_sum
            local_silhouette = torch.nn.functional.max_pool2d(
                silhouette[None, None],
                kernel_size=3,
                stride=1,
                padding=1,
            )[0, 0]
            target_edge_covered = (target_edge * (1.0 - local_silhouette)).sum() / target_edge_sum
            boundary_loss = pred_to_target_edge + target_edge_covered
        else:
            boundary_loss = torch.zeros((), dtype=dtype, device=torch_device)
        if float(depth_weight) > 0.0 and bool(valid_depth_np.any()):
            depth_weight_map = valid_depth * silhouette.detach().clamp(min=0.05)
            depth_residual = torch.nn.functional.smooth_l1_loss(
                rendered_depth * depth_weight_map,
                target_depth_t * depth_weight_map,
                beta=0.03,
                reduction="sum",
            )
            depth_loss = depth_residual / valid_depth_sum
        else:
            depth_loss = torch.zeros((), dtype=dtype, device=torch_device)
        area_loss = ((pred_sum - target_sum) / target_sum) ** 2
        pred_cx = (silhouette * xx).sum() / pred_sum
        pred_cy = (silhouette * yy).sum() / pred_sum
        center_loss = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        transform_reg = log_scale.square() + 0.25 * rotvec.square().sum() + 0.25 * translation.square().sum()
        loss = (
            dice_loss
            + float(leakage_weight) * leakage
            + float(miss_weight) * miss
            + float(outside_distance_weight) * outside_loss
            + float(depth_weight) * depth_loss
            + float(boundary_weight) * boundary_loss
            + float(area_weight) * area_loss
            + float(center_weight) * center_loss
            + float(transform_reg_weight) * transform_reg
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best["loss"]:
            best = {
                "loss": loss_value,
                "iteration": int(iteration),
                "log_scale": float(log_scale.detach().cpu()),
                "rotvec": rotvec.detach().cpu().numpy().astype(float).tolist(),
                "translation": translation.detach().cpu().numpy().astype(float).tolist(),
                "dice_loss": float(dice_loss.detach().cpu()),
                "leakage": float(leakage.detach().cpu()),
                "miss": float(miss.detach().cpu()),
                "outside_loss": float(outside_loss.detach().cpu()),
                "depth_loss": float(depth_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
                "area_loss": float(area_loss.detach().cpu()),
                "center_loss": float(center_loss.detach().cpu()),
            }

    with torch.no_grad():
        best_rotvec = torch.as_tensor(best["rotvec"], dtype=dtype, device=torch_device)
        best_rotation = torch_rodrigues(best_rotvec).detach().cpu().numpy()
    delta = make_sim3(
        scale=float(math.exp(best["log_scale"])),
        rotation=best_rotation,
        translation=best["translation"],
    )
    optimized = delta @ np.asarray(initial_transform, dtype=np.float64)
    info = {
        "enabled": True,
        "accepted": None,
        "render_size": int(render_size),
        "points": int(len(points)),
        "iterations": int(iterations),
        "lr": float(lr),
        "splat_radius": int(splat_radius),
        "sigma": float(sigma),
        "opacity": float(opacity),
        "depth_weight": float(depth_weight),
        "boundary_weight": float(boundary_weight),
        "best": best,
        "delta": delta.tolist(),
    }
    return optimized, info


def apply_silhouette_refinement(
    transform,
    baseline_score,
    complete_points,
    eval_points,
    intrinsic_px,
    image_shape,
    target_mask,
    target_depth,
    *,
    args,
    reason_prefix="",
):
    optimized, info = optimize_silhouette_delta_sim3(
        transform,
        complete_points,
        intrinsic_px,
        image_shape,
        target_mask,
        target_depth=target_depth,
        render_size=args.silhouette_opt_render_size,
        max_points=args.silhouette_opt_max_points,
        iterations=args.silhouette_opt_iterations,
        lr=args.silhouette_opt_lr,
        splat_radius=args.silhouette_opt_splat_radius,
        sigma=args.silhouette_opt_sigma,
        opacity=args.silhouette_opt_opacity,
        leakage_weight=args.silhouette_opt_leakage_weight,
        miss_weight=args.silhouette_opt_miss_weight,
        outside_distance_weight=args.silhouette_opt_outside_distance_weight,
        depth_weight=args.silhouette_opt_depth_weight,
        boundary_weight=args.silhouette_opt_boundary_weight,
        area_weight=args.silhouette_opt_area_weight,
        center_weight=args.silhouette_opt_center_weight,
        transform_reg_weight=args.silhouette_opt_transform_reg_weight,
        seed=args.seed,
        device=args.device,
    )
    if not info.get("enabled"):
        return np.asarray(transform, dtype=np.float64), baseline_score, info

    candidate_score = evaluate_transform(
        eval_points,
        optimized,
        intrinsic_px,
        target_depth,
        target_mask,
        splat_radius=args.splat_radius,
    )
    min_gain = float(args.silhouette_opt_min_score_gain)
    prefix = f"{reason_prefix}_" if reason_prefix else ""
    if candidate_score["score"] > baseline_score["score"] + min_gain:
        info.update(
            {
                "accepted": True,
                "reason": f"{prefix}render_score_improved",
                "min_score_gain": min_gain,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
            }
        )
        return np.asarray(optimized, dtype=np.float64), candidate_score, info

    info.update(
        {
            "accepted": False,
            "reason": f"{prefix}render_score_not_improved",
            "min_score_gain": min_gain,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
        }
    )
    return np.asarray(transform, dtype=np.float64), baseline_score, info


def visible_3d_correspondences(
    transform,
    source_points,
    target_points,
    intrinsic_px,
    image_shape,
    target_mask,
    *,
    max_depth_delta,
    max_pairs,
    trim_quantile,
    seed,
):
    visible_indices, moved = visible_source_indices(
        source_points,
        transform,
        intrinsic_px,
        image_shape,
        target_mask,
        max_depth_delta=max_depth_delta,
    )
    if len(visible_indices) < 16:
        return None
    rng = np.random.default_rng(int(seed))
    if len(visible_indices) > int(max_pairs):
        visible_indices = rng.choice(visible_indices, size=int(max_pairs), replace=False)
    distances, nearest = cKDTree(target_points).query(moved[visible_indices], k=1)
    threshold = np.quantile(distances, float(trim_quantile))
    keep = distances <= threshold
    if int(keep.sum()) < 16:
        return None
    return {
        "source_points": np.asarray(source_points, dtype=np.float64)[visible_indices][keep],
        "base_points": moved[visible_indices][keep],
        "target_points": np.asarray(target_points, dtype=np.float64)[nearest[keep]],
        "initial_distances": distances[keep],
        "visible_points": int(len(visible_indices)),
        "kept_pairs": int(keep.sum()),
        "trim_threshold": float(threshold),
    }


def visible_distance_stats(points, targets):
    distances = np.linalg.norm(np.asarray(points, dtype=np.float64) - np.asarray(targets, dtype=np.float64), axis=1)
    return {
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(distances.max()),
        "count": int(len(distances)),
    }


def load_bridge_anchor_points(
    partial_points,
    *,
    sample_dir,
    flag,
    moge_to_partial,
    max_points,
    seed,
    index_name=None,
):
    sample_dir = Path(sample_dir)
    index_path = sample_dir / (
        index_name or f"{flag}_moge_to_raw_partial_partial_to_moge_index.npy"
    )
    if not index_path.exists():
        return np.empty((0, 3), dtype=np.float64), {
            "enabled": False,
            "reason": "missing_partial_to_moge_index",
            "index_path": str(index_path),
        }

    partial_points = np.asarray(partial_points, dtype=np.float64)
    partial_to_moge = np.load(index_path)
    if len(partial_to_moge) != len(partial_points):
        return np.empty((0, 3), dtype=np.float64), {
            "enabled": False,
            "reason": "index_length_mismatch",
            "index_path": str(index_path),
            "partial_points": int(len(partial_points)),
            "index_length": int(len(partial_to_moge)),
        }

    valid = np.asarray(partial_to_moge, dtype=np.int64) >= 0
    valid_count = int(valid.sum())
    if valid_count == 0:
        return np.empty((0, 3), dtype=np.float64), {
            "enabled": False,
            "reason": "not_enough_bridge_matches",
            "index_path": str(index_path),
            "valid_matches": valid_count,
            "match_ratio": float(valid_count / max(len(partial_points), 1)),
        }

    inverse_moge_to_partial = np.linalg.inv(np.asarray(moge_to_partial, dtype=np.float64))
    anchors = apply_sim3(partial_points[valid], inverse_moge_to_partial)
    rng = np.random.default_rng(int(seed))
    if max_points is not None and len(anchors) > int(max_points):
        chosen = rng.choice(len(anchors), size=int(max_points), replace=False)
        anchors = anchors[chosen]

    return anchors, {
        "enabled": True,
        "index_path": str(index_path),
        "partial_points": int(len(partial_points)),
        "valid_matches": valid_count,
        "match_ratio": float(valid_count / max(len(partial_points), 1)),
        "anchors": int(len(anchors)),
    }


def anchor_alignment_stats(source_points, anchor_points, transform, *, max_points, seed):
    rng = np.random.default_rng(int(seed))
    source_points = np.asarray(source_points, dtype=np.float64)
    anchor_points = np.asarray(anchor_points, dtype=np.float64)
    if len(source_points) > int(max_points):
        source_points = source_points[rng.choice(len(source_points), size=int(max_points), replace=False)]
    if len(anchor_points) > int(max_points):
        anchor_points = anchor_points[rng.choice(len(anchor_points), size=int(max_points), replace=False)]
    moved = apply_sim3(source_points, transform)
    complete_tree = cKDTree(moved)
    anchor_tree = cKDTree(anchor_points)
    anchor_to_complete, _ = complete_tree.query(anchor_points, k=1)
    complete_to_anchor, _ = anchor_tree.query(moved, k=1)
    complete_trim35 = complete_to_anchor[complete_to_anchor <= np.quantile(complete_to_anchor, 0.35)]
    complete_trim70 = complete_to_anchor[complete_to_anchor <= np.quantile(complete_to_anchor, 0.70)]
    return {
        "anchor_to_complete": {
            "mean": float(anchor_to_complete.mean()),
            "median": float(np.median(anchor_to_complete)),
            "p95": float(np.percentile(anchor_to_complete, 95)),
            "max": float(anchor_to_complete.max()),
            "count": int(len(anchor_to_complete)),
        },
        "complete_to_anchor_trim35": {
            "mean": float(complete_trim35.mean()),
            "count": int(len(complete_trim35)),
        },
        "complete_to_anchor_trim70": {
            "mean": float(complete_trim70.mean()),
            "count": int(len(complete_trim70)),
        },
    }


def partial_alignment_stats(source_points, target_points, transform, *, max_points, seed):
    rng = np.random.default_rng(int(seed))
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if len(source_points) > int(max_points):
        source_points = source_points[rng.choice(len(source_points), size=int(max_points), replace=False)]
    if len(target_points) > int(max_points):
        target_points = target_points[rng.choice(len(target_points), size=int(max_points), replace=False)]
    moved = apply_sim3(source_points, transform)
    source_tree = cKDTree(moved)
    target_tree = cKDTree(target_points)
    partial_to_complete, _ = source_tree.query(target_points, k=1)
    complete_to_partial, _ = target_tree.query(moved, k=1)
    complete_trim35 = complete_to_partial[complete_to_partial <= np.quantile(complete_to_partial, 0.35)]
    complete_trim70 = complete_to_partial[complete_to_partial <= np.quantile(complete_to_partial, 0.70)]
    return {
        "partial_to_complete": {
            "mean": float(partial_to_complete.mean()),
            "median": float(np.median(partial_to_complete)),
            "p95": float(np.percentile(partial_to_complete, 95)),
            "max": float(partial_to_complete.max()),
            "count": int(len(partial_to_complete)),
        },
        "complete_to_partial_trim35": {
            "mean": float(complete_trim35.mean()),
            "count": int(len(complete_trim35)),
        },
        "complete_to_partial_trim70": {
            "mean": float(complete_trim70.mean()),
            "count": int(len(complete_trim70)),
        },
    }


def refine_symmetric_partial_icp(
    initial_transform,
    source_points,
    target_points,
    *,
    iterations,
    max_pairs,
    complete_trim_quantile,
    partial_trim_quantile,
    partial_weight,
    max_step_translation,
    min_step_scale,
    max_step_scale,
    seed,
):
    if int(iterations) <= 0:
        return np.asarray(initial_transform, dtype=np.float64), {"enabled": False}

    rng = np.random.default_rng(int(seed))
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if len(source_points) > int(max_pairs):
        source_points = source_points[rng.choice(len(source_points), size=int(max_pairs), replace=False)]
    if len(target_points) > int(max_pairs):
        target_points = target_points[rng.choice(len(target_points), size=int(max_pairs), replace=False)]

    current = np.asarray(initial_transform, dtype=np.float64).copy()
    initial_stats = partial_alignment_stats(
        source_points,
        target_points,
        current,
        max_points=max_pairs,
        seed=seed,
    )
    history = []
    weight_repeats = max(1, int(round(float(partial_weight))))
    for iteration in range(int(iterations)):
        moved = apply_sim3(source_points, current)
        target_tree = cKDTree(target_points)
        source_tree = cKDTree(moved)
        source_corr = []
        target_corr = []

        complete_distances, complete_nearest = target_tree.query(moved, k=1)
        complete_threshold = np.quantile(complete_distances, float(complete_trim_quantile))
        complete_keep = complete_distances <= complete_threshold
        if int(complete_keep.sum()) >= 16:
            source_corr.append(moved[complete_keep])
            target_corr.append(target_points[complete_nearest[complete_keep]])

        partial_distances, partial_nearest = source_tree.query(target_points, k=1)
        partial_threshold = np.quantile(partial_distances, float(partial_trim_quantile))
        partial_keep = partial_distances <= partial_threshold
        if int(partial_keep.sum()) >= 16:
            for _ in range(weight_repeats):
                source_corr.append(moved[partial_nearest[partial_keep]])
                target_corr.append(target_points[partial_keep])

        if not source_corr:
            break

        source_corr = np.concatenate(source_corr, axis=0)
        target_corr = np.concatenate(target_corr, axis=0)
        delta = umeyama_similarity(source_corr, target_corr)
        step_scale, _, step_translation = decompose_sim3(delta)
        step_translation_norm = float(np.linalg.norm(step_translation))
        if (
            step_translation_norm > float(max_step_translation)
            or step_scale < float(min_step_scale)
            or step_scale > float(max_step_scale)
        ):
            history.append(
                {
                    "iteration": int(iteration),
                    "accepted_step": False,
                    "reason": "step_out_of_bounds",
                    "step_scale": float(step_scale),
                    "step_translation_norm": step_translation_norm,
                    "complete_pairs": int(complete_keep.sum()),
                    "partial_pairs": int(partial_keep.sum()),
                }
            )
            break

        current = delta @ current
        stats = partial_alignment_stats(
            source_points,
            target_points,
            current,
            max_points=max_pairs,
            seed=seed,
        )
        history.append(
            {
                "iteration": int(iteration),
                "accepted_step": True,
                "step_scale": float(step_scale),
                "step_translation_norm": step_translation_norm,
                "complete_pairs": int(complete_keep.sum()),
                "partial_pairs": int(partial_keep.sum()),
                "stats": stats,
            }
        )

    candidate_stats = partial_alignment_stats(
        source_points,
        target_points,
        current,
        max_points=max_pairs,
        seed=seed,
    )
    delta_total = current @ np.linalg.inv(np.asarray(initial_transform, dtype=np.float64))
    info = {
        "enabled": True,
        "accepted": None,
        "method": "symmetric_partial_icp",
        "iterations": int(iterations),
        "max_pairs": int(max_pairs),
        "complete_trim_quantile": float(complete_trim_quantile),
        "partial_trim_quantile": float(partial_trim_quantile),
        "partial_weight": float(partial_weight),
        "initial_alignment": initial_stats,
        "candidate_alignment": candidate_stats,
        "initial_distance": initial_stats["partial_to_complete"],
        "candidate_distance": candidate_stats["partial_to_complete"],
        "history": history,
        "delta": delta_total.tolist(),
    }
    return current, info


def pca_axes_for_transform(source_points, transform):
    source_points = np.asarray(source_points, dtype=np.float64)
    centered = source_points - source_points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(centered.T))
    projected = centered @ axes
    order = np.argsort(np.ptp(projected, axis=0))[::-1]
    axes = axes[:, order]
    _, rotation, _ = decompose_sim3(transform)
    return rotation @ axes


def apply_axis_scale_delta(transform, center, axes, factors):
    axes = np.asarray(axes, dtype=np.float64)
    factors = np.asarray(factors, dtype=np.float64)
    linear = axes @ np.diag(factors) @ axes.T
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = linear
    delta[:3, 3] = np.asarray(center, dtype=np.float64) - linear @ np.asarray(center, dtype=np.float64)
    return delta @ np.asarray(transform, dtype=np.float64)


def refine_pca_anisotropic_partial(
    initial_transform,
    source_points,
    target_points,
    *,
    scale_triplets,
    pre_icp_iterations,
    icp_iterations,
    max_pairs,
    complete_trim_quantile,
    partial_trim_quantile,
    partial_weight,
    max_step_translation,
    min_step_scale,
    max_step_scale,
    objective_pc_p95_weight,
    objective_cp70_weight,
    objective_scale_reg_weight,
    seed,
):
    if not scale_triplets:
        return np.asarray(initial_transform, dtype=np.float64), {"enabled": False}

    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    initial_transform = np.asarray(initial_transform, dtype=np.float64)
    initial_stats = partial_alignment_stats(
        source_points,
        target_points,
        initial_transform,
        max_points=max_pairs,
        seed=seed,
    )
    if int(pre_icp_iterations) > 0:
        warmup_transform, warmup_info = refine_symmetric_partial_icp(
            initial_transform,
            source_points,
            target_points,
            iterations=pre_icp_iterations,
            max_pairs=max_pairs,
            complete_trim_quantile=complete_trim_quantile,
            partial_trim_quantile=partial_trim_quantile,
            partial_weight=partial_weight,
            max_step_translation=max_step_translation,
            min_step_scale=min_step_scale,
            max_step_scale=max_step_scale,
            seed=seed,
        )
    else:
        warmup_transform = initial_transform
        warmup_info = {"enabled": False}
    center = apply_sim3(source_points, warmup_transform).mean(axis=0)
    axes = pca_axes_for_transform(source_points, warmup_transform)
    best = None
    candidates = []
    for index, factors in enumerate(scale_triplets):
        factors = tuple(float(v) for v in factors)
        scaled = apply_axis_scale_delta(warmup_transform, center, axes, factors)
        refined, icp_info = refine_symmetric_partial_icp(
            scaled,
            source_points,
            target_points,
            iterations=icp_iterations,
            max_pairs=max_pairs,
            complete_trim_quantile=complete_trim_quantile,
            partial_trim_quantile=partial_trim_quantile,
            partial_weight=partial_weight,
            max_step_translation=max_step_translation,
            min_step_scale=min_step_scale,
            max_step_scale=max_step_scale,
            seed=seed,
        )
        stats = icp_info.get("candidate_alignment") or partial_alignment_stats(
            source_points,
            target_points,
            refined,
            max_points=max_pairs,
            seed=seed,
        )
        scale_reg = float(np.sum(np.square(np.log(np.asarray(factors, dtype=np.float64)))))
        objective = float(
            stats["partial_to_complete"]["mean"]
            + float(objective_pc_p95_weight) * stats["partial_to_complete"]["p95"]
            + float(objective_cp70_weight) * stats["complete_to_partial_trim70"]["mean"]
            + float(objective_scale_reg_weight) * scale_reg
        )
        item = {
            "index": int(index),
            "factors": list(factors),
            "objective": objective,
            "scale_reg": scale_reg,
            "alignment": stats,
            "icp": icp_info,
        }
        candidates.append(item)
        if best is None or objective < best["info"]["objective"]:
            best = {"transform": refined, "info": item}

    if best is None:
        return initial_transform.copy(), {
            "enabled": True,
            "accepted": False,
            "reason": "no_anisotropic_candidates",
        }

    info = {
        "enabled": True,
        "accepted": None,
        "method": "pca_anisotropic_scale_then_symmetric_partial_icp",
        "pre_icp": warmup_info,
        "scale_triplets": [list(item) for item in scale_triplets],
        "initial_alignment": initial_stats,
        "candidate_alignment": best["info"]["alignment"],
        "initial_distance": initial_stats["partial_to_complete"],
        "candidate_distance": best["info"]["alignment"]["partial_to_complete"],
        "best": best["info"],
        "candidates": sorted(candidates, key=lambda item: item["objective"])[:12],
        "delta": (best["transform"] @ np.linalg.inv(initial_transform)).tolist(),
    }
    return best["transform"], info


def symmetric_partial_correspondence_pairs(
    transform,
    source_points,
    target_points,
    *,
    max_pairs,
    complete_trim_quantile,
    partial_trim_quantile,
    partial_weight,
    seed,
):
    rng = np.random.default_rng(int(seed))
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_indices = np.arange(len(source_points))
    target_indices = np.arange(len(target_points))
    if len(source_indices) > int(max_pairs):
        source_indices = rng.choice(source_indices, size=int(max_pairs), replace=False)
    if len(target_indices) > int(max_pairs):
        target_indices = rng.choice(target_indices, size=int(max_pairs), replace=False)

    source_sample = source_points[source_indices]
    target_sample = target_points[target_indices]
    moved = apply_sim3(source_sample, transform)
    source_tree = cKDTree(moved)
    target_tree = cKDTree(target_sample)
    bases = []
    targets = []
    weights = []

    complete_distances, complete_nearest = target_tree.query(moved, k=1)
    complete_threshold = np.quantile(complete_distances, float(complete_trim_quantile))
    complete_keep = complete_distances <= complete_threshold
    if int(complete_keep.sum()) >= 16:
        bases.append(moved[complete_keep])
        targets.append(target_sample[complete_nearest[complete_keep]])
        weights.append(np.ones(int(complete_keep.sum()), dtype=np.float64))

    partial_distances, partial_nearest = source_tree.query(target_sample, k=1)
    partial_threshold = np.quantile(partial_distances, float(partial_trim_quantile))
    partial_keep = partial_distances <= partial_threshold
    if int(partial_keep.sum()) >= 16:
        bases.append(moved[partial_nearest[partial_keep]])
        targets.append(target_sample[partial_keep])
        weights.append(np.full(int(partial_keep.sum()), float(partial_weight), dtype=np.float64))

    if not bases:
        return None
    base = np.concatenate(bases, axis=0)
    target = np.concatenate(targets, axis=0)
    weight = np.concatenate(weights, axis=0)
    if len(base) < 16:
        return None
    return {
        "base_points": base,
        "target_points": target,
        "weights": weight,
        "source_sampled": int(len(source_indices)),
        "target_sampled": int(len(target_indices)),
        "complete_pairs": int(complete_keep.sum()),
        "partial_pairs": int(partial_keep.sum()),
        "complete_threshold": float(complete_threshold),
        "partial_threshold": float(partial_threshold),
    }


def optimize_partial_affine_delta(
    initial_transform,
    source_points,
    target_points,
    *,
    pre_icp_iterations,
    max_pairs,
    complete_trim_quantile,
    partial_trim_quantile,
    partial_weight,
    max_step_translation,
    min_step_scale,
    max_step_scale,
    iterations,
    lr,
    distance_weight,
    transform_reg_weight,
    axis_scale_reg_weight,
    seed,
    device,
):
    if int(iterations) <= 0:
        return np.asarray(initial_transform, dtype=np.float64), {"enabled": False}

    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    initial_transform = np.asarray(initial_transform, dtype=np.float64)
    initial_stats = partial_alignment_stats(
        source_points,
        target_points,
        initial_transform,
        max_points=max_pairs,
        seed=seed,
    )
    if int(pre_icp_iterations) > 0:
        warmup_transform, warmup_info = refine_symmetric_partial_icp(
            initial_transform,
            source_points,
            target_points,
            iterations=pre_icp_iterations,
            max_pairs=max_pairs,
            complete_trim_quantile=complete_trim_quantile,
            partial_trim_quantile=partial_trim_quantile,
            partial_weight=partial_weight,
            max_step_translation=max_step_translation,
            min_step_scale=min_step_scale,
            max_step_scale=max_step_scale,
            seed=seed,
        )
    else:
        warmup_transform = initial_transform
        warmup_info = {"enabled": False}

    pairs = symmetric_partial_correspondence_pairs(
        warmup_transform,
        source_points,
        target_points,
        max_pairs=max_pairs,
        complete_trim_quantile=complete_trim_quantile,
        partial_trim_quantile=partial_trim_quantile,
        partial_weight=partial_weight,
        seed=seed,
    )
    if pairs is None:
        return initial_transform.copy(), {
            "enabled": True,
            "accepted": False,
            "reason": "not_enough_continuous_affine_pairs",
        }

    center = apply_sim3(source_points, warmup_transform).mean(axis=0)
    axes = pca_axes_for_transform(source_points, warmup_transform)
    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    base = torch.as_tensor(pairs["base_points"], dtype=dtype, device=torch_device)
    target = torch.as_tensor(pairs["target_points"], dtype=dtype, device=torch_device)
    weights = torch.as_tensor(pairs["weights"], dtype=dtype, device=torch_device)
    weights = weights / torch.clamp(weights.mean(), min=1e-6)
    center_t = torch.as_tensor(center.astype(np.float32), dtype=dtype, device=torch_device)
    axes_t = torch.as_tensor(axes.astype(np.float32), dtype=dtype, device=torch_device)

    log_axis_scale = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    rotvec = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    translation = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    optimizer = torch.optim.Adam([log_axis_scale, rotvec, translation], lr=float(lr))
    best = {
        "loss": float("inf"),
        "log_axis_scale": [0.0, 0.0, 0.0],
        "rotvec": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
    }
    for iteration in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        axis_scale = torch.exp(log_axis_scale)
        rotation = torch_rodrigues(rotvec)
        local = (base - center_t) @ axes_t
        scaled = (local * axis_scale) @ axes_t.T
        moved = scaled @ rotation.T + center_t + translation
        residual = moved - target
        per_point = torch.nn.functional.smooth_l1_loss(moved, target, beta=0.03, reduction="none").sum(dim=1)
        distance_loss = torch.mean(per_point * weights)
        transform_reg = 0.25 * rotvec.square().sum() + 0.25 * translation.square().sum()
        axis_reg = log_axis_scale.square().sum()
        loss = (
            float(distance_weight) * distance_loss
            + float(transform_reg_weight) * transform_reg
            + float(axis_scale_reg_weight) * axis_reg
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best["loss"]:
            distances = torch.linalg.norm(residual.detach(), dim=1)
            best = {
                "loss": loss_value,
                "iteration": int(iteration),
                "log_axis_scale": log_axis_scale.detach().cpu().numpy().astype(float).tolist(),
                "rotvec": rotvec.detach().cpu().numpy().astype(float).tolist(),
                "translation": translation.detach().cpu().numpy().astype(float).tolist(),
                "distance_loss": float(distance_loss.detach().cpu()),
                "distance_mean": float(distances.mean().cpu()),
                "distance_p95": float(torch.quantile(distances, 0.95).cpu()),
                "axis_scale_reg": float(axis_reg.detach().cpu()),
            }

    with torch.no_grad():
        best_scales = np.exp(np.asarray(best["log_axis_scale"], dtype=np.float64))
        best_rotation = torch_rodrigues(torch.as_tensor(best["rotvec"], dtype=dtype, device=torch_device)).detach().cpu().numpy()
    axis_linear = axes @ np.diag(best_scales) @ axes.T
    linear = best_rotation @ axis_linear
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = linear
    delta[:3, 3] = np.asarray(center, dtype=np.float64) + np.asarray(best["translation"], dtype=np.float64) - linear @ np.asarray(center, dtype=np.float64)
    optimized = delta @ warmup_transform
    candidate_stats = partial_alignment_stats(
        source_points,
        target_points,
        optimized,
        max_points=max_pairs,
        seed=seed,
    )
    info = {
        "enabled": True,
        "accepted": None,
        "method": "continuous_pca_affine_partial",
        "pre_icp": warmup_info,
        "pairs": {key: pairs[key] for key in (
            "source_sampled",
            "target_sampled",
            "complete_pairs",
            "partial_pairs",
            "complete_threshold",
            "partial_threshold",
        )},
        "iterations": int(iterations),
        "lr": float(lr),
        "initial_alignment": initial_stats,
        "candidate_alignment": candidate_stats,
        "initial_distance": initial_stats["partial_to_complete"],
        "candidate_distance": candidate_stats["partial_to_complete"],
        "best": best,
        "delta": (optimized @ np.linalg.inv(initial_transform)).tolist(),
    }
    return optimized, info


def nearest_trimmed_correspondences(
    transform,
    source_points,
    target_points,
    *,
    max_pairs,
    trim_quantile,
    seed,
):
    rng = np.random.default_rng(int(seed))
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_indices = np.arange(len(source_points))
    if len(source_indices) > int(max_pairs):
        source_indices = rng.choice(source_indices, size=int(max_pairs), replace=False)
    moved = apply_sim3(source_points[source_indices], transform)
    distances, nearest = cKDTree(target_points).query(moved, k=1)
    threshold = np.quantile(distances, float(trim_quantile))
    keep = distances <= threshold
    if int(keep.sum()) < 16:
        return None
    return {
        "source_points": source_points[source_indices][keep],
        "base_points": moved[keep],
        "target_points": target_points[nearest[keep]],
        "initial_distances": distances[keep],
        "sampled_points": int(len(source_indices)),
        "kept_pairs": int(keep.sum()),
        "trim_threshold": float(threshold),
    }


def optimize_partial_delta_sim3(
    initial_transform,
    source_points,
    target_points,
    *,
    max_pairs,
    trim_quantile,
    iterations,
    lr,
    distance_loss,
    distance_weight,
    transform_reg_weight,
    seed,
    device,
):
    if int(iterations) <= 0:
        return np.asarray(initial_transform, dtype=np.float64), {"enabled": False}

    correspondences = nearest_trimmed_correspondences(
        initial_transform,
        source_points,
        target_points,
        max_pairs=max_pairs,
        trim_quantile=trim_quantile,
        seed=seed,
    )
    if correspondences is None:
        return np.asarray(initial_transform, dtype=np.float64), {
            "enabled": True,
            "accepted": False,
            "reason": "not_enough_partial_correspondences",
        }

    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    base = torch.as_tensor(correspondences["base_points"], dtype=dtype, device=torch_device)
    target = torch.as_tensor(correspondences["target_points"], dtype=dtype, device=torch_device)
    log_scale = torch.nn.Parameter(torch.zeros((), dtype=dtype, device=torch_device))
    rotvec = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    translation = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    optimizer = torch.optim.Adam([log_scale, rotvec, translation], lr=float(lr))
    best = {
        "loss": float("inf"),
        "log_scale": 0.0,
        "rotvec": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
    }
    for iteration in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale)
        rotation = torch_rodrigues(rotvec)
        moved = scale * (base @ rotation.T) + translation
        residual = moved - target
        distances = torch.linalg.norm(residual, dim=1)
        if distance_loss == "l2":
            dist_loss = torch.mean(torch.sum(residual * residual, dim=1))
        else:
            dist_loss = torch.nn.functional.smooth_l1_loss(moved, target, beta=0.03)
        transform_reg = log_scale.square() + 0.25 * rotvec.square().sum() + 0.25 * translation.square().sum()
        loss = float(distance_weight) * dist_loss + float(transform_reg_weight) * transform_reg
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best["loss"]:
            best = {
                "loss": loss_value,
                "iteration": int(iteration),
                "log_scale": float(log_scale.detach().cpu()),
                "rotvec": rotvec.detach().cpu().numpy().astype(float).tolist(),
                "translation": translation.detach().cpu().numpy().astype(float).tolist(),
                "distance_loss": float(dist_loss.detach().cpu()),
                "distance_mean": float(distances.detach().mean().cpu()),
                "distance_p95": float(torch.quantile(distances.detach(), 0.95).cpu()),
            }

    with torch.no_grad():
        best_rotvec = torch.as_tensor(best["rotvec"], dtype=dtype, device=torch_device)
        best_rotation = torch_rodrigues(best_rotvec).detach().cpu().numpy()
    delta = make_sim3(
        scale=float(math.exp(best["log_scale"])),
        rotation=best_rotation,
        translation=best["translation"],
    )
    optimized = delta @ np.asarray(initial_transform, dtype=np.float64)
    moved_optimized = apply_sim3(correspondences["source_points"], optimized)
    initial_stats = visible_distance_stats(correspondences["base_points"], correspondences["target_points"])
    optimized_stats = visible_distance_stats(moved_optimized, correspondences["target_points"])
    info = {
        "enabled": True,
        "accepted": None,
        "sampled_points": correspondences["sampled_points"],
        "kept_pairs": correspondences["kept_pairs"],
        "trim_threshold": correspondences["trim_threshold"],
        "iterations": int(iterations),
        "lr": float(lr),
        "distance_loss": distance_loss,
        "initial_distance": initial_stats,
        "candidate_distance": optimized_stats,
        "best": best,
        "delta": delta.tolist(),
    }
    return optimized, info


def optimize_bridge_anchor_delta_sim3(
    initial_transform,
    source_points,
    anchor_points,
    intrinsic_px,
    image_shape,
    target_mask,
    *,
    max_pairs,
    trim_quantile,
    iterations,
    lr,
    distance_loss,
    distance_weight,
    silhouette_weight,
    silhouette_render_size,
    silhouette_points,
    silhouette_splat_radius,
    silhouette_sigma,
    silhouette_opacity,
    leakage_weight,
    miss_weight,
    transform_reg_weight,
    seed,
    device,
):
    if int(iterations) <= 0:
        return np.asarray(initial_transform, dtype=np.float64), {"enabled": False}

    correspondences = nearest_trimmed_correspondences(
        initial_transform,
        source_points,
        anchor_points,
        max_pairs=max_pairs,
        trim_quantile=trim_quantile,
        seed=seed,
    )
    if correspondences is None:
        return np.asarray(initial_transform, dtype=np.float64), {
            "enabled": True,
            "accepted": False,
            "reason": "not_enough_bridge_anchor_correspondences",
        }

    rng = np.random.default_rng(int(seed) + 29)
    silhouette_source = np.asarray(source_points, dtype=np.float64)
    if len(silhouette_source) > int(silhouette_points):
        silhouette_source = silhouette_source[
            rng.choice(len(silhouette_source), size=int(silhouette_points), replace=False)
        ]
    silhouette_base = apply_sim3(silhouette_source, initial_transform)
    target_np = resize_target_mask(target_mask, silhouette_render_size)
    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    base = torch.as_tensor(correspondences["base_points"], dtype=dtype, device=torch_device)
    target = torch.as_tensor(correspondences["target_points"], dtype=dtype, device=torch_device)
    silhouette_base_t = torch.as_tensor(silhouette_base, dtype=dtype, device=torch_device)
    intrinsic = torch.as_tensor(np.asarray(intrinsic_px, dtype=np.float32), dtype=dtype, device=torch_device)
    target_mask_t = torch.as_tensor(target_np, dtype=dtype, device=torch_device)
    target_sum = torch.clamp(target_mask_t.sum(), min=1.0)

    log_scale = torch.nn.Parameter(torch.zeros((), dtype=dtype, device=torch_device))
    rotvec = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    translation = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    optimizer = torch.optim.Adam([log_scale, rotvec, translation], lr=float(lr))
    best = {
        "loss": float("inf"),
        "log_scale": 0.0,
        "rotvec": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
    }
    eps = 1e-6
    for iteration in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale)
        rotation = torch_rodrigues(rotvec)
        moved = scale * (base @ rotation.T) + translation
        residual = moved - target
        distances = torch.linalg.norm(residual, dim=1)
        if distance_loss == "l2":
            dist_loss = torch.mean(torch.sum(residual * residual, dim=1))
        else:
            dist_loss = torch.nn.functional.smooth_l1_loss(moved, target, beta=0.03)

        silhouette_points_moved = scale * (silhouette_base_t @ rotation.T) + translation
        silhouette = render_soft_silhouette_torch(
            silhouette_points_moved,
            intrinsic,
            image_shape,
            render_size=silhouette_render_size,
            splat_radius=silhouette_splat_radius,
            sigma=silhouette_sigma,
            opacity=silhouette_opacity,
        )
        pred_sum = torch.clamp(silhouette.sum(), min=eps)
        intersection = (silhouette * target_mask_t).sum()
        dice_loss = 1.0 - (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        leakage = (silhouette * (1.0 - target_mask_t)).sum() / pred_sum
        miss = (target_mask_t * (1.0 - silhouette)).sum() / target_sum
        transform_reg = log_scale.square() + 0.25 * rotvec.square().sum() + 0.25 * translation.square().sum()
        loss = (
            float(distance_weight) * dist_loss
            + float(silhouette_weight) * dice_loss
            + float(leakage_weight) * leakage
            + float(miss_weight) * miss
            + float(transform_reg_weight) * transform_reg
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best["loss"]:
            best = {
                "loss": loss_value,
                "iteration": int(iteration),
                "log_scale": float(log_scale.detach().cpu()),
                "rotvec": rotvec.detach().cpu().numpy().astype(float).tolist(),
                "translation": translation.detach().cpu().numpy().astype(float).tolist(),
                "distance_loss": float(dist_loss.detach().cpu()),
                "distance_mean": float(distances.detach().mean().cpu()),
                "distance_p95": float(torch.quantile(distances.detach(), 0.95).cpu()),
                "dice_loss": float(dice_loss.detach().cpu()),
                "leakage": float(leakage.detach().cpu()),
                "miss": float(miss.detach().cpu()),
            }

    with torch.no_grad():
        best_rotvec = torch.as_tensor(best["rotvec"], dtype=dtype, device=torch_device)
        best_rotation = torch_rodrigues(best_rotvec).detach().cpu().numpy()
    delta = make_sim3(
        scale=float(math.exp(best["log_scale"])),
        rotation=best_rotation,
        translation=best["translation"],
    )
    optimized = delta @ np.asarray(initial_transform, dtype=np.float64)
    initial_anchor = anchor_alignment_stats(
        source_points,
        anchor_points,
        initial_transform,
        max_points=max_pairs,
        seed=seed,
    )
    candidate_anchor = anchor_alignment_stats(
        source_points,
        anchor_points,
        optimized,
        max_points=max_pairs,
        seed=seed,
    )
    info = {
        "enabled": True,
        "accepted": None,
        "sampled_points": correspondences["sampled_points"],
        "kept_pairs": correspondences["kept_pairs"],
        "trim_threshold": correspondences["trim_threshold"],
        "iterations": int(iterations),
        "lr": float(lr),
        "distance_loss": distance_loss,
        "initial_distance": initial_anchor["anchor_to_complete"],
        "candidate_distance": candidate_anchor["anchor_to_complete"],
        "initial_anchor": initial_anchor,
        "candidate_anchor": candidate_anchor,
        "best": best,
        "delta": delta.tolist(),
    }
    return optimized, info


def choose_bridge_anchor_refinement(
    baseline_transform,
    baseline_score,
    candidate_transform,
    candidate_score,
    optimization_info,
    *,
    min_2d_objective_gain,
    max_2d_objective_drop,
    min_anchor_improvement,
):
    info = dict(optimization_info)
    initial_anchor = info.get("initial_anchor") or {}
    candidate_anchor = info.get("candidate_anchor") or {}
    initial_mean = float((initial_anchor.get("anchor_to_complete") or {}).get("mean", float("inf")))
    candidate_mean = float((candidate_anchor.get("anchor_to_complete") or {}).get("mean", float("inf")))
    min_anchor_improvement = float(min_anchor_improvement)
    required_anchor = initial_mean * (1.0 - min_anchor_improvement)
    baseline_objective = score_2d_gate_objective(baseline_score)
    candidate_objective = score_2d_gate_objective(candidate_score)
    min_2d_objective_gain = float(min_2d_objective_gain)
    max_2d_objective_drop = float(max_2d_objective_drop)
    objective_gain = candidate_objective - baseline_objective
    two_d_ok = (
        objective_gain >= min_2d_objective_gain
        or candidate_objective >= baseline_objective - max_2d_objective_drop
    )
    anchor_improved = candidate_mean <= required_anchor
    info.update(
        {
            "baseline_2d_objective": float(baseline_objective),
            "candidate_2d_objective": float(candidate_objective),
            "2d_objective_gain": float(objective_gain),
            "min_2d_objective_gain": min_2d_objective_gain,
            "max_2d_objective_drop": max_2d_objective_drop,
            "min_anchor_improvement": min_anchor_improvement,
            "required_anchor_mean": float(required_anchor),
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
        }
    )
    if two_d_ok and anchor_improved:
        info.update({"accepted": True, "reason": "bridge_anchor_improved_with_2d_guard"})
        return np.asarray(candidate_transform, dtype=np.float64), candidate_score, info
    reason = "2d_objective_drop" if not two_d_ok else "bridge_anchor_distance_not_improved"
    info.update({"accepted": False, "reason": reason})
    return np.asarray(baseline_transform, dtype=np.float64), baseline_score, info


def choose_retry_continuous_variant(
    coordinate_transform,
    coordinate_score,
    coordinate_anchor,
    continuous_transform,
    continuous_score,
    continuous_info,
    *,
    anchor_scale,
    anchor_weight,
    max_2d_objective_drop,
    min_anchor_improvement,
):
    info = dict(continuous_info or {})
    coordinate_anchor = coordinate_anchor or {}
    candidate_anchor = info.get("candidate_anchor") or {}
    coordinate_mean = float((coordinate_anchor.get("anchor_to_complete") or {}).get("mean", float("inf")))
    candidate_mean = float((candidate_anchor.get("anchor_to_complete") or {}).get("mean", float("inf")))
    required_anchor = coordinate_mean * (1.0 - float(min_anchor_improvement))
    coordinate_2d = score_2d_gate_objective(coordinate_score)
    candidate_2d = score_2d_gate_objective(continuous_score)
    coordinate_joint = score_2d_anchor_objective(
        coordinate_score,
        coordinate_anchor,
        anchor_scale=anchor_scale,
        anchor_weight=anchor_weight,
    )
    candidate_joint = score_2d_anchor_objective(
        continuous_score,
        candidate_anchor,
        anchor_scale=anchor_scale,
        anchor_weight=anchor_weight,
    )
    two_d_ok = candidate_2d >= coordinate_2d - float(max_2d_objective_drop)
    anchor_improved = candidate_mean <= required_anchor
    joint_improved = candidate_joint > coordinate_joint + 1e-8
    info.update(
        {
            "coordinate_2d_objective": float(coordinate_2d),
            "candidate_2d_objective": float(candidate_2d),
            "coordinate_joint_objective": float(coordinate_joint),
            "candidate_joint_objective": float(candidate_joint),
            "max_2d_objective_drop": float(max_2d_objective_drop),
            "min_anchor_improvement": float(min_anchor_improvement),
            "required_anchor_mean": float(required_anchor),
        }
    )
    if two_d_ok and anchor_improved and joint_improved:
        info.update({"accepted": True, "reason": "retry_continuous_joint_objective_improved"})
        return (
            np.asarray(continuous_transform, dtype=np.float64),
            continuous_score,
            candidate_anchor,
            info,
        )
    if not two_d_ok:
        reason = "2d_objective_drop"
    elif not anchor_improved:
        reason = "bridge_anchor_distance_not_improved"
    else:
        reason = "joint_objective_not_improved"
    info.update({"accepted": False, "reason": reason})
    return (
        np.asarray(coordinate_transform, dtype=np.float64),
        coordinate_score,
        coordinate_anchor,
        info,
    )


def choose_partial_refinement(
    baseline_transform,
    candidate_transform,
    optimization_info,
    *,
    min_distance_improvement,
    max_delta_rotation_deg,
    max_delta_translation,
    min_delta_scale,
    max_delta_scale,
):
    info = dict(optimization_info)
    initial_distance = info.get("initial_distance") or {}
    candidate_distance = info.get("candidate_distance") or {}
    initial_mean = float(initial_distance.get("mean", float("inf")))
    candidate_mean = float(candidate_distance.get("mean", float("inf")))
    min_distance_improvement = float(min_distance_improvement)
    required_distance = initial_mean * (1.0 - min_distance_improvement)
    delta = np.asarray(info.get("delta", np.eye(4)), dtype=np.float64)
    delta_scale, delta_rotation, delta_translation = decompose_sim3(delta)
    delta_rotation_deg = float(np.degrees(Rotation.from_matrix(delta_rotation).magnitude()))
    delta_translation_norm = float(np.linalg.norm(delta_translation))
    distance_improved = candidate_mean <= required_distance
    scale_ok = float(min_delta_scale) <= float(delta_scale) <= float(max_delta_scale)
    rotation_ok = delta_rotation_deg <= float(max_delta_rotation_deg)
    translation_ok = delta_translation_norm <= float(max_delta_translation)
    info.update(
        {
            "min_distance_improvement": min_distance_improvement,
            "required_distance_mean": required_distance,
            "delta_decomposed": {
                "scale": float(delta_scale),
                "rotation_degrees": delta_rotation_deg,
                "translation_norm": delta_translation_norm,
                "translation": delta_translation.tolist(),
            },
            "limits": {
                "max_delta_rotation_deg": float(max_delta_rotation_deg),
                "max_delta_translation": float(max_delta_translation),
                "min_delta_scale": float(min_delta_scale),
                "max_delta_scale": float(max_delta_scale),
            },
        }
    )
    if distance_improved and scale_ok and rotation_ok and translation_ok:
        info.update({"accepted": True, "reason": "partial_distance_improved_with_small_delta"})
        return np.asarray(candidate_transform, dtype=np.float64), info
    if not distance_improved:
        reason = "partial_distance_not_improved"
    elif not scale_ok:
        reason = "delta_scale_out_of_bounds"
    elif not rotation_ok:
        reason = "delta_rotation_out_of_bounds"
    else:
        reason = "delta_translation_out_of_bounds"
    info.update({"accepted": False, "reason": reason})
    return np.asarray(baseline_transform, dtype=np.float64), info


def choose_anisotropic_partial_refinement(
    baseline_transform,
    candidate_transform,
    optimization_info,
    *,
    min_distance_improvement,
    min_delta_axis_scale,
    max_delta_axis_scale,
    max_delta_translation,
):
    info = dict(optimization_info)
    initial_distance = info.get("initial_distance") or {}
    candidate_distance = info.get("candidate_distance") or {}
    initial_mean = float(initial_distance.get("mean", float("inf")))
    candidate_mean = float(candidate_distance.get("mean", float("inf")))
    required_distance = initial_mean * (1.0 - float(min_distance_improvement))
    delta = np.asarray(info.get("delta", np.eye(4)), dtype=np.float64)
    singular_values = np.linalg.svd(delta[:3, :3], compute_uv=False)
    translation_norm = float(np.linalg.norm(delta[:3, 3]))
    distance_improved = candidate_mean <= required_distance
    axis_scale_ok = (
        float(singular_values.min()) >= float(min_delta_axis_scale)
        and float(singular_values.max()) <= float(max_delta_axis_scale)
    )
    translation_ok = translation_norm <= float(max_delta_translation)
    info.update(
        {
            "min_distance_improvement": float(min_distance_improvement),
            "required_distance_mean": float(required_distance),
            "delta_decomposed": {
                "axis_scales": singular_values.astype(float).tolist(),
                "translation_norm": translation_norm,
                "translation": delta[:3, 3].astype(float).tolist(),
            },
            "limits": {
                "min_delta_axis_scale": float(min_delta_axis_scale),
                "max_delta_axis_scale": float(max_delta_axis_scale),
                "max_delta_translation": float(max_delta_translation),
            },
        }
    )
    if distance_improved and axis_scale_ok and translation_ok:
        info.update({"accepted": True, "reason": "anisotropic_partial_distance_improved"})
        return np.asarray(candidate_transform, dtype=np.float64), info
    if not distance_improved:
        reason = "partial_distance_not_improved"
    elif not axis_scale_ok:
        reason = "delta_axis_scale_out_of_bounds"
    else:
        reason = "delta_translation_out_of_bounds"
    info.update({"accepted": False, "reason": reason})
    return np.asarray(baseline_transform, dtype=np.float64), info


def optimize_visible_3d_delta_sim3(
    initial_transform,
    source_points,
    target_points,
    intrinsic_px,
    image_shape,
    target_mask,
    *,
    max_depth_delta,
    max_pairs,
    trim_quantile,
    iterations,
    lr,
    distance_loss,
    distance_weight,
    silhouette_weight,
    silhouette_render_size,
    silhouette_points,
    silhouette_splat_radius,
    silhouette_sigma,
    silhouette_opacity,
    leakage_weight,
    miss_weight,
    transform_reg_weight,
    seed,
    device,
):
    if int(iterations) <= 0:
        return np.asarray(initial_transform, dtype=np.float64), {"enabled": False}

    correspondences = visible_3d_correspondences(
        initial_transform,
        source_points,
        target_points,
        intrinsic_px,
        image_shape,
        target_mask,
        max_depth_delta=max_depth_delta,
        max_pairs=max_pairs,
        trim_quantile=trim_quantile,
        seed=seed,
    )
    if correspondences is None:
        return np.asarray(initial_transform, dtype=np.float64), {
            "enabled": True,
            "accepted": False,
            "reason": "not_enough_visible_correspondences",
        }

    rng = np.random.default_rng(int(seed) + 17)
    silhouette_source = np.asarray(source_points, dtype=np.float64)
    if len(silhouette_source) > int(silhouette_points):
        silhouette_source = silhouette_source[
            rng.choice(len(silhouette_source), size=int(silhouette_points), replace=False)
        ]
    silhouette_base = apply_sim3(silhouette_source, initial_transform)
    target_np = resize_target_mask(target_mask, silhouette_render_size)
    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    base = torch.as_tensor(correspondences["base_points"], dtype=dtype, device=torch_device)
    target = torch.as_tensor(correspondences["target_points"], dtype=dtype, device=torch_device)
    silhouette_base_t = torch.as_tensor(silhouette_base, dtype=dtype, device=torch_device)
    intrinsic = torch.as_tensor(np.asarray(intrinsic_px, dtype=np.float32), dtype=dtype, device=torch_device)
    target_mask_t = torch.as_tensor(target_np, dtype=dtype, device=torch_device)
    target_sum = torch.clamp(target_mask_t.sum(), min=1.0)

    log_scale = torch.nn.Parameter(torch.zeros((), dtype=dtype, device=torch_device))
    rotvec = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    translation = torch.nn.Parameter(torch.zeros(3, dtype=dtype, device=torch_device))
    optimizer = torch.optim.Adam([log_scale, rotvec, translation], lr=float(lr))
    best = {
        "loss": float("inf"),
        "log_scale": 0.0,
        "rotvec": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
    }
    eps = 1e-6
    for iteration in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale)
        rotation = torch_rodrigues(rotvec)
        moved = scale * (base @ rotation.T) + translation
        residual = moved - target
        distances = torch.linalg.norm(residual, dim=1)
        if distance_loss == "l2":
            dist_loss = torch.mean(torch.sum(residual * residual, dim=1))
        else:
            dist_loss = torch.nn.functional.smooth_l1_loss(moved, target, beta=0.03)

        silhouette_points_moved = scale * (silhouette_base_t @ rotation.T) + translation
        silhouette = render_soft_silhouette_torch(
            silhouette_points_moved,
            intrinsic,
            image_shape,
            render_size=silhouette_render_size,
            splat_radius=silhouette_splat_radius,
            sigma=silhouette_sigma,
            opacity=silhouette_opacity,
        )
        pred_sum = torch.clamp(silhouette.sum(), min=eps)
        intersection = (silhouette * target_mask_t).sum()
        dice_loss = 1.0 - (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        leakage = (silhouette * (1.0 - target_mask_t)).sum() / pred_sum
        miss = (target_mask_t * (1.0 - silhouette)).sum() / target_sum
        transform_reg = log_scale.square() + 0.25 * rotvec.square().sum() + 0.25 * translation.square().sum()
        loss = (
            float(distance_weight) * dist_loss
            + float(silhouette_weight) * dice_loss
            + float(leakage_weight) * leakage
            + float(miss_weight) * miss
            + float(transform_reg_weight) * transform_reg
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best["loss"]:
            best = {
                "loss": loss_value,
                "iteration": int(iteration),
                "log_scale": float(log_scale.detach().cpu()),
                "rotvec": rotvec.detach().cpu().numpy().astype(float).tolist(),
                "translation": translation.detach().cpu().numpy().astype(float).tolist(),
                "distance_loss": float(dist_loss.detach().cpu()),
                "distance_mean": float(distances.detach().mean().cpu()),
                "distance_p95": float(torch.quantile(distances.detach(), 0.95).cpu()),
                "dice_loss": float(dice_loss.detach().cpu()),
                "leakage": float(leakage.detach().cpu()),
                "miss": float(miss.detach().cpu()),
            }

    with torch.no_grad():
        best_rotvec = torch.as_tensor(best["rotvec"], dtype=dtype, device=torch_device)
        best_rotation = torch_rodrigues(best_rotvec).detach().cpu().numpy()
    delta = make_sim3(
        scale=float(math.exp(best["log_scale"])),
        rotation=best_rotation,
        translation=best["translation"],
    )
    optimized = delta @ np.asarray(initial_transform, dtype=np.float64)
    moved_optimized = apply_sim3(correspondences["source_points"], optimized)
    initial_stats = visible_distance_stats(correspondences["base_points"], correspondences["target_points"])
    optimized_stats = visible_distance_stats(moved_optimized, correspondences["target_points"])
    info = {
        "enabled": True,
        "accepted": None,
        "visible_points": correspondences["visible_points"],
        "kept_pairs": correspondences["kept_pairs"],
        "trim_threshold": correspondences["trim_threshold"],
        "iterations": int(iterations),
        "lr": float(lr),
        "distance_loss": distance_loss,
        "initial_distance": initial_stats,
        "candidate_distance": optimized_stats,
        "best": best,
        "delta": delta.tolist(),
    }
    return optimized, info


def draw_overlay(path, image_path, target_mask, rendered_mask):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    overlay = image.copy()
    target = np.asarray(target_mask, dtype=bool)
    rendered = np.asarray(rendered_mask, dtype=bool)
    overlay[target] = (0.55 * overlay[target] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[rendered] = (0.55 * overlay[rendered] + 0.45 * np.array([0, 80, 255])).astype(np.uint8)
    overlap = target & rendered
    overlay[overlap] = (0.35 * overlay[overlap] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def paint_merge(first, first_color, second, second_color):
    a = deepcopy(first)
    b = deepcopy(second)
    a.paint_uniform_color(first_color)
    b.paint_uniform_color(second_color)
    return a + b


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def run(args):
    paths = infer_paths(
        flag=args.flag,
        sample_root=args.sample_root,
        sample_dir=args.sample_dir,
        out_root=args.out_root,
        out_dir=args.out_dir,
        image_name=args.image_name,
        complete_name=args.complete_name,
        object_mask_name=args.object_mask_name,
        moge_to_partial_name=args.moge_to_partial_name,
    )
    sample_dir = paths.sample_dir
    out_dir = paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = paths.image_path
    complete_path = paths.complete_path
    mask_path = paths.object_mask_path
    moge_to_partial_path = paths.moge_to_partial_path
    partial_path = Path(args.partial_path) if getattr(args, "partial_path", None) else PROJECT_ROOT / "data" / f"{args.flag}.ply"

    complete_pcd, complete_points = load_point_cloud(complete_path)
    partial_pcd, partial_points = load_point_cloud(partial_path)
    full_moge_points, full_moge_colors, full_moge_pixels, moge_info = run_moge_with_pixels(
        image_path=image_path,
        pretrained=args.moge_model,
        device=args.device,
        fp16=bool(args.fp16),
    )
    image_shape = tuple(int(v) for v in moge_info["image_hw"])
    object_mask = load_alpha_mask(mask_path)
    object_moge = filter_moge_points_by_object_mask(
        points=full_moge_points,
        colors=full_moge_colors,
        pixel_xy=full_moge_pixels,
        object_mask=object_mask,
        alpha_threshold=args.object_alpha_threshold,
        erode_pixels=args.object_mask_erode_pixels,
    )
    target_depth, target_mask = target_depth_from_moge(
        object_moge.points,
        object_moge.pixel_xy,
        image_shape=image_shape,
    )
    intrinsic_px = normalized_intrinsic_to_pixel(
        moge_info["output_keys"]["intrinsics"],
        image_shape,
    )

    eval_points = maybe_subsample(complete_points, args.eval_points, args.seed)
    rotations = axis_aligned_rotations()
    candidates = initial_candidates(
        source_points=eval_points,
        target_points=object_moge.points,
        rotations=rotations,
        scale_multipliers=parse_float_list(args.scale_multipliers),
    )
    best_initial = select_best_candidate(
        candidates,
        source_points=eval_points,
        intrinsic_px=intrinsic_px,
        target_depth=target_depth,
        target_mask=target_mask,
        splat_radius=args.splat_radius,
    )

    refined, refined_score, refine_history = refine_transform_coordinate_search(
        best_initial["transform"],
        eval_points,
        intrinsic_px,
        target_depth,
        target_mask,
        splat_radius=args.splat_radius,
        translation_steps=parse_float_list(args.translation_steps),
        rotation_steps_deg=parse_float_list(args.rotation_steps_deg),
        scale_steps=parse_float_list(args.scale_steps),
        rounds=args.refine_rounds,
    )
    retry_2d_search = {
        "enabled": False,
        "accepted": False,
        "reason": "not_run",
    }
    bridge_anchor_optimization = {
        "enabled": False,
        "accepted": False,
        "reason": "not_run",
    }
    post_retry_silhouette_optimization = {
        "enabled": False,
        "accepted": False,
        "reason": "not_run",
    }
    refined, refined_score, silhouette_optimization = apply_silhouette_refinement(
        refined,
        refined_score,
        complete_points,
        eval_points,
        intrinsic_px,
        image_shape,
        target_mask,
        target_depth,
        args=args,
    )
    visible_3d_optimized, visible_3d_optimization = optimize_visible_3d_delta_sim3(
        refined,
        complete_points,
        object_moge.points,
        intrinsic_px,
        image_shape,
        target_mask,
        max_depth_delta=args.visible_3d_opt_max_depth_delta,
        max_pairs=args.visible_3d_opt_max_pairs,
        trim_quantile=args.visible_3d_opt_trim_quantile,
        iterations=args.visible_3d_opt_iterations,
        lr=args.visible_3d_opt_lr,
        distance_loss=args.visible_3d_opt_distance_loss,
        distance_weight=args.visible_3d_opt_distance_weight,
        silhouette_weight=args.visible_3d_opt_silhouette_weight,
        silhouette_render_size=args.visible_3d_opt_silhouette_render_size,
        silhouette_points=args.visible_3d_opt_silhouette_points,
        silhouette_splat_radius=args.visible_3d_opt_silhouette_splat_radius,
        silhouette_sigma=args.visible_3d_opt_silhouette_sigma,
        silhouette_opacity=args.visible_3d_opt_silhouette_opacity,
        leakage_weight=args.visible_3d_opt_leakage_weight,
        miss_weight=args.visible_3d_opt_miss_weight,
        transform_reg_weight=args.visible_3d_opt_transform_reg_weight,
        seed=args.seed,
        device=args.device,
    )
    if visible_3d_optimization.get("enabled") and visible_3d_optimization.get("candidate_distance"):
        visible_3d_score = evaluate_transform(
            eval_points,
            visible_3d_optimized,
            intrinsic_px,
            target_depth,
            target_mask,
            splat_radius=args.splat_radius,
        )
        refined, refined_score, visible_3d_optimization = choose_visible_3d_refinement(
            refined,
            refined_score,
            visible_3d_optimized,
            visible_3d_score,
            visible_3d_optimization,
            max_score_drop=args.visible_3d_opt_max_score_drop,
            min_distance_improvement=args.visible_3d_opt_min_distance_improvement,
        )
    coordinate_refined = np.asarray(refined, dtype=np.float64).copy()
    if args.visible_icp_iterations > 0:
        icp_refined, icp_history = refine_visible_icp(
            coordinate_refined,
            complete_points,
            object_moge.points,
            intrinsic_px,
            image_shape,
            target_mask,
            iterations=args.visible_icp_iterations,
            trim_quantile=args.icp_trim_quantile,
            max_pairs=args.icp_max_pairs,
            seed=args.seed,
        )
        icp_score = evaluate_transform(
            eval_points,
            icp_refined,
            intrinsic_px,
            target_depth,
            target_mask,
            splat_radius=args.splat_radius,
        )
        refined, final_score, icp_acceptance = choose_icp_refinement(
            coordinate_refined,
            refined_score,
            icp_refined,
            icp_score,
            rollback_on_score_drop=getattr(args, "icp_rollback_on_score_drop", True),
            min_score_gain=getattr(args, "icp_min_score_gain", 0.0),
        )
    else:
        icp_history = []
        final_score = refined_score
        icp_acceptance = {
            "accepted": False,
            "reason": "disabled",
            "min_score_gain": float(getattr(args, "icp_min_score_gain", 0.0)),
            "baseline_score": refined_score,
            "candidate_score": None,
        }

    complete_to_moge = refined
    moge_to_partial = np.load(moge_to_partial_path)
    complete_to_partial = compose_complete_to_partial(moge_to_partial, complete_to_moge)
    bridge_anchor_points = np.empty((0, 3), dtype=np.float64)
    bridge_anchor_info = {
        "enabled": False,
        "reason": "disabled",
    }
    bridge_anchor_scale = 1.0
    if bool(getattr(args, "bridge_anchor_opt_enabled", True)):
        bridge_anchor_points, bridge_anchor_info = load_bridge_anchor_points(
            partial_points,
            sample_dir=sample_dir,
            flag=args.flag,
            moge_to_partial=moge_to_partial,
            max_points=getattr(args, "bridge_anchor_opt_max_anchors", 20000),
            seed=args.seed,
            index_name=getattr(args, "partial_to_moge_index_name", None),
        )
        if len(bridge_anchor_points) > 0:
            bridge_anchor_scale = max(float(np.linalg.norm(bbox_extent(bridge_anchor_points))), 1e-6)
    registration_2d_acceptance = evaluate_2d_acceptance(
        final_score,
        enabled=getattr(args, "require_2d_acceptance", True),
        min_iou=getattr(args, "min_2d_iou", 0.78),
        min_coverage=getattr(args, "min_2d_coverage", 0.80),
        max_leakage=getattr(args, "max_2d_leakage", 0.12),
        min_edge_iou=getattr(args, "min_2d_edge_iou", 0.02),
        max_edge_chamfer_px=getattr(args, "max_2d_edge_chamfer_px", 18.0),
    )
    if (
        bool(getattr(args, "retry_2d_on_gate_failure", True))
        and not registration_2d_acceptance["accepted"]
        and int(getattr(args, "retry_2d_top_k", 0)) > 0
    ):
        retry_candidates = initial_candidates(
            source_points=eval_points,
            target_points=object_moge.points,
            rotations=rotations,
            scale_multipliers=parse_float_list(args.retry_2d_scale_multipliers),
        )
        ranked = []
        for index, transform in enumerate(retry_candidates):
            score = evaluate_transform(
                eval_points,
                transform,
                intrinsic_px,
                target_depth,
                target_mask,
                splat_radius=args.splat_radius,
            )
            ranked.append(
                {
                    "index": int(index),
                    "transform": transform,
                    "initial_score": score,
                    "initial_2d_objective": score_2d_gate_objective(score),
                }
            )
        ranked.sort(key=lambda item: item["initial_2d_objective"], reverse=True)
        top_k = ranked[: int(args.retry_2d_top_k)]
        retry_items = []
        best_retry = {
            "transform": complete_to_moge,
            "score": final_score,
            "objective": score_2d_gate_objective(final_score),
            "joint_objective": score_2d_gate_objective(final_score),
            "acceptance": registration_2d_acceptance,
            "source": "baseline",
        }
        if bridge_anchor_info.get("enabled") and len(bridge_anchor_points) >= 16:
            baseline_anchor = anchor_alignment_stats(
                complete_points,
                bridge_anchor_points,
                complete_to_moge,
                max_points=args.bridge_anchor_opt_max_pairs,
                seed=args.seed,
            )
            best_retry["anchor_alignment"] = baseline_anchor
            best_retry["joint_objective"] = score_2d_anchor_objective(
                final_score,
                baseline_anchor,
                anchor_scale=bridge_anchor_scale,
                anchor_weight=args.bridge_anchor_opt_retry_anchor_weight,
            )
        for item in top_k:
            candidate_transform, candidate_score, candidate_history = refine_transform_coordinate_search(
                item["transform"],
                eval_points,
                intrinsic_px,
                target_depth,
                target_mask,
                splat_radius=args.splat_radius,
                translation_steps=parse_float_list(args.retry_2d_translation_steps),
                rotation_steps_deg=parse_float_list(args.retry_2d_rotation_steps_deg),
                scale_steps=parse_float_list(args.retry_2d_scale_steps),
                rounds=args.retry_2d_refine_rounds,
                selection_objective="2d_gate",
            )
            candidate_acceptance = evaluate_2d_acceptance(
                candidate_score,
                enabled=getattr(args, "require_2d_acceptance", True),
                min_iou=getattr(args, "min_2d_iou", 0.78),
                min_coverage=getattr(args, "min_2d_coverage", 0.80),
                max_leakage=getattr(args, "max_2d_leakage", 0.12),
                min_edge_iou=getattr(args, "min_2d_edge_iou", 0.02),
                max_edge_chamfer_px=getattr(args, "max_2d_edge_chamfer_px", 18.0),
            )
            objective = score_2d_gate_objective(candidate_score)
            candidate_anchor = None
            joint_objective = objective
            if bridge_anchor_info.get("enabled") and len(bridge_anchor_points) >= 16:
                candidate_anchor = anchor_alignment_stats(
                    complete_points,
                    bridge_anchor_points,
                    candidate_transform,
                    max_points=args.bridge_anchor_opt_max_pairs,
                    seed=args.seed,
                )
                joint_objective = score_2d_anchor_objective(
                    candidate_score,
                    candidate_anchor,
                    anchor_scale=bridge_anchor_scale,
                    anchor_weight=args.bridge_anchor_opt_retry_anchor_weight,
                )
            selected_transform = candidate_transform
            selected_score = candidate_score
            selected_anchor = candidate_anchor
            selected_acceptance = candidate_acceptance
            selected_variant = "coordinate"
            retry_continuous_optimization = {
                "enabled": False,
                "accepted": False,
                "reason": "disabled",
            }
            if (
                bool(getattr(args, "retry_continuous_opt_enabled", True))
                and bridge_anchor_info.get("enabled")
                and len(bridge_anchor_points) >= 16
                and candidate_anchor is not None
                and int(getattr(args, "retry_continuous_opt_iterations", 0)) > 0
            ):
                continuous_transform, retry_continuous_optimization = optimize_bridge_anchor_delta_sim3(
                    candidate_transform,
                    complete_points,
                    bridge_anchor_points,
                    intrinsic_px,
                    image_shape,
                    target_mask,
                    max_pairs=args.retry_continuous_opt_max_pairs,
                    trim_quantile=args.retry_continuous_opt_trim_quantile,
                    iterations=args.retry_continuous_opt_iterations,
                    lr=args.retry_continuous_opt_lr,
                    distance_loss=args.retry_continuous_opt_distance_loss,
                    distance_weight=args.retry_continuous_opt_distance_weight,
                    silhouette_weight=args.retry_continuous_opt_silhouette_weight,
                    silhouette_render_size=args.retry_continuous_opt_silhouette_render_size,
                    silhouette_points=args.retry_continuous_opt_silhouette_points,
                    silhouette_splat_radius=args.retry_continuous_opt_silhouette_splat_radius,
                    silhouette_sigma=args.retry_continuous_opt_silhouette_sigma,
                    silhouette_opacity=args.retry_continuous_opt_silhouette_opacity,
                    leakage_weight=args.retry_continuous_opt_leakage_weight,
                    miss_weight=args.retry_continuous_opt_miss_weight,
                    transform_reg_weight=args.retry_continuous_opt_transform_reg_weight,
                    seed=args.seed + int(item["index"]),
                    device=args.device,
                )
                if (
                    retry_continuous_optimization.get("enabled")
                    and retry_continuous_optimization.get("candidate_anchor")
                ):
                    continuous_score = evaluate_transform(
                        eval_points,
                        continuous_transform,
                        intrinsic_px,
                        target_depth,
                        target_mask,
                        splat_radius=args.splat_radius,
                    )
                    (
                        selected_transform,
                        selected_score,
                        selected_anchor,
                        retry_continuous_optimization,
                    ) = choose_retry_continuous_variant(
                        candidate_transform,
                        candidate_score,
                        candidate_anchor,
                        continuous_transform,
                        continuous_score,
                        retry_continuous_optimization,
                        anchor_scale=bridge_anchor_scale,
                        anchor_weight=args.bridge_anchor_opt_retry_anchor_weight,
                        max_2d_objective_drop=args.retry_continuous_opt_max_2d_objective_drop,
                        min_anchor_improvement=args.retry_continuous_opt_min_anchor_improvement,
                    )
                    retry_continuous_optimization.update(
                        {
                            "coordinate_score": candidate_score,
                            "continuous_score": continuous_score,
                        }
                    )
                    if retry_continuous_optimization.get("accepted"):
                        selected_variant = "continuous"
                    selected_acceptance = evaluate_2d_acceptance(
                        selected_score,
                        enabled=getattr(args, "require_2d_acceptance", True),
                        min_iou=getattr(args, "min_2d_iou", 0.78),
                        min_coverage=getattr(args, "min_2d_coverage", 0.80),
                        max_leakage=getattr(args, "max_2d_leakage", 0.12),
                        min_edge_iou=getattr(args, "min_2d_edge_iou", 0.02),
                        max_edge_chamfer_px=getattr(args, "max_2d_edge_chamfer_px", 18.0),
                    )
            objective = score_2d_gate_objective(selected_score)
            joint_objective = objective
            if selected_anchor is not None:
                joint_objective = score_2d_anchor_objective(
                    selected_score,
                    selected_anchor,
                    anchor_scale=bridge_anchor_scale,
                    anchor_weight=args.bridge_anchor_opt_retry_anchor_weight,
                )
            retry_item = {
                "index": item["index"],
                "initial_score": item["initial_score"],
                "initial_2d_objective": item["initial_2d_objective"],
                "selected_variant": selected_variant,
                "coordinate_refined_score": candidate_score,
                "coordinate_2d_objective": score_2d_gate_objective(candidate_score),
                "coordinate_anchor_alignment": candidate_anchor,
                "retry_continuous_optimization": retry_continuous_optimization,
                "refined_score": selected_score,
                "refined_2d_objective": objective,
                "refined_joint_objective": joint_objective,
                "anchor_alignment": selected_anchor,
                "registration_2d_acceptance": selected_acceptance,
                "refine_history": candidate_history,
            }
            retry_items.append(retry_item)
            candidate_better = joint_objective > best_retry["joint_objective"] + 1e-8
            candidate_passes = selected_acceptance["accepted"] and not best_retry["acceptance"]["accepted"]
            if candidate_passes or candidate_better:
                source = f"retry_candidate_{item['index']}"
                if selected_variant == "continuous":
                    source = f"{source}_continuous"
                best_retry = {
                    "transform": selected_transform,
                    "score": selected_score,
                    "objective": objective,
                    "joint_objective": joint_objective,
                    "acceptance": selected_acceptance,
                    "anchor_alignment": selected_anchor,
                    "source": source,
                }
        retry_2d_search = {
            "enabled": True,
            "accepted": best_retry["source"] != "baseline",
            "reason": "replaced_with_retry_candidate" if best_retry["source"] != "baseline" else "no_retry_candidate_improved",
            "top_k": int(args.retry_2d_top_k),
            "source": best_retry["source"],
            "baseline_2d_objective": score_2d_gate_objective(final_score),
            "best_2d_objective": best_retry["objective"],
            "best_joint_objective": best_retry["joint_objective"],
            "anchor_load": bridge_anchor_info,
            "anchor_scale": float(bridge_anchor_scale),
            "retry_anchor_weight": float(args.bridge_anchor_opt_retry_anchor_weight),
            "candidates": retry_items,
        }
        if best_retry["source"] != "baseline":
            complete_to_moge = best_retry["transform"]
            final_score = best_retry["score"]
            registration_2d_acceptance = best_retry["acceptance"]
            complete_to_moge, final_score, post_retry_silhouette_optimization = apply_silhouette_refinement(
                complete_to_moge,
                final_score,
                complete_points,
                eval_points,
                intrinsic_px,
                image_shape,
                target_mask,
                target_depth,
                args=args,
                reason_prefix="post_retry",
            )
            registration_2d_acceptance = evaluate_2d_acceptance(
                final_score,
                enabled=getattr(args, "require_2d_acceptance", True),
                min_iou=getattr(args, "min_2d_iou", 0.78),
                min_coverage=getattr(args, "min_2d_coverage", 0.80),
                max_leakage=getattr(args, "max_2d_leakage", 0.12),
                min_edge_iou=getattr(args, "min_2d_edge_iou", 0.02),
                max_edge_chamfer_px=getattr(args, "max_2d_edge_chamfer_px", 18.0),
            )
            complete_to_partial = compose_complete_to_partial(moge_to_partial, complete_to_moge)
    if bool(getattr(args, "bridge_anchor_opt_enabled", True)):
        bridge_anchor_optimization = {"anchor_load": bridge_anchor_info}
        if bridge_anchor_info.get("enabled") and len(bridge_anchor_points) >= 16:
            anchor_optimized, bridge_anchor_optimization = optimize_bridge_anchor_delta_sim3(
                complete_to_moge,
                complete_points,
                bridge_anchor_points,
                intrinsic_px,
                image_shape,
                target_mask,
                max_pairs=args.bridge_anchor_opt_max_pairs,
                trim_quantile=args.bridge_anchor_opt_trim_quantile,
                iterations=args.bridge_anchor_opt_iterations,
                lr=args.bridge_anchor_opt_lr,
                distance_loss=args.bridge_anchor_opt_distance_loss,
                distance_weight=args.bridge_anchor_opt_distance_weight,
                silhouette_weight=args.bridge_anchor_opt_silhouette_weight,
                silhouette_render_size=args.bridge_anchor_opt_silhouette_render_size,
                silhouette_points=args.bridge_anchor_opt_silhouette_points,
                silhouette_splat_radius=args.bridge_anchor_opt_silhouette_splat_radius,
                silhouette_sigma=args.bridge_anchor_opt_silhouette_sigma,
                silhouette_opacity=args.bridge_anchor_opt_silhouette_opacity,
                leakage_weight=args.bridge_anchor_opt_leakage_weight,
                miss_weight=args.bridge_anchor_opt_miss_weight,
                transform_reg_weight=args.bridge_anchor_opt_transform_reg_weight,
                seed=args.seed,
                device=args.device,
            )
            bridge_anchor_optimization["anchor_load"] = bridge_anchor_info
            if bridge_anchor_optimization.get("enabled") and bridge_anchor_optimization.get("candidate_anchor"):
                anchor_score = evaluate_transform(
                    eval_points,
                    anchor_optimized,
                    intrinsic_px,
                    target_depth,
                    target_mask,
                    splat_radius=args.splat_radius,
                )
                complete_to_moge, final_score, bridge_anchor_optimization = choose_bridge_anchor_refinement(
                    complete_to_moge,
                    final_score,
                    anchor_optimized,
                    anchor_score,
                    bridge_anchor_optimization,
                    min_2d_objective_gain=args.bridge_anchor_opt_min_2d_objective_gain,
                    max_2d_objective_drop=args.bridge_anchor_opt_max_2d_objective_drop,
                    min_anchor_improvement=args.bridge_anchor_opt_min_anchor_improvement,
                )
                bridge_anchor_optimization["anchor_load"] = bridge_anchor_info
                complete_to_partial = compose_complete_to_partial(moge_to_partial, complete_to_moge)
                registration_2d_acceptance = evaluate_2d_acceptance(
                    final_score,
                    enabled=getattr(args, "require_2d_acceptance", True),
                    min_iou=getattr(args, "min_2d_iou", 0.78),
                    min_coverage=getattr(args, "min_2d_coverage", 0.80),
                    max_leakage=getattr(args, "max_2d_leakage", 0.12),
                    min_edge_iou=getattr(args, "min_2d_edge_iou", 0.02),
                    max_edge_chamfer_px=getattr(args, "max_2d_edge_chamfer_px", 18.0),
                )
        elif bridge_anchor_info.get("enabled"):
            bridge_anchor_optimization.update(
                {
                    "enabled": True,
                    "accepted": False,
                    "reason": "not_enough_bridge_anchors_for_optimization",
                }
            )
    else:
        bridge_anchor_optimization = {
            "enabled": False,
            "accepted": False,
            "reason": "disabled",
        }
    if not registration_2d_acceptance["accepted"]:
        partial_refinement = {
            "enabled": False,
            "accepted": False,
            "reason": "registration_2d_threshold_not_met",
            "registration_2d_acceptance": registration_2d_acceptance,
        }
    elif args.partial_refine_mode == "pca_anisotropic":
        partial_optimized, partial_refinement = refine_pca_anisotropic_partial(
            complete_to_partial,
            complete_points,
            partial_points,
            scale_triplets=parse_scale_triplets(args.partial_refine_anisotropic_scale_triplets),
            pre_icp_iterations=args.partial_refine_anisotropic_pre_icp_iterations,
            icp_iterations=args.partial_refine_anisotropic_icp_iterations,
            max_pairs=args.partial_refine_max_pairs,
            complete_trim_quantile=args.partial_refine_complete_trim_quantile,
            partial_trim_quantile=args.partial_refine_partial_trim_quantile,
            partial_weight=args.partial_refine_partial_weight,
            max_step_translation=args.partial_refine_max_step_translation,
            min_step_scale=args.partial_refine_min_step_scale,
            max_step_scale=args.partial_refine_max_step_scale,
            objective_pc_p95_weight=args.partial_refine_objective_pc_p95_weight,
            objective_cp70_weight=args.partial_refine_objective_cp70_weight,
            objective_scale_reg_weight=args.partial_refine_objective_scale_reg_weight,
            seed=args.seed,
        )
        if partial_refinement.get("enabled") and partial_refinement.get("candidate_distance"):
            complete_to_partial, partial_refinement = choose_anisotropic_partial_refinement(
                complete_to_partial,
                partial_optimized,
                partial_refinement,
                min_distance_improvement=args.partial_refine_min_distance_improvement,
                min_delta_axis_scale=args.partial_refine_min_delta_axis_scale,
                max_delta_axis_scale=args.partial_refine_max_delta_axis_scale,
                max_delta_translation=args.partial_refine_max_delta_translation,
            )
    elif args.partial_refine_mode == "continuous_affine":
        partial_optimized, partial_refinement = optimize_partial_affine_delta(
            complete_to_partial,
            complete_points,
            partial_points,
            pre_icp_iterations=args.partial_refine_anisotropic_pre_icp_iterations,
            max_pairs=args.partial_refine_max_pairs,
            complete_trim_quantile=args.partial_refine_complete_trim_quantile,
            partial_trim_quantile=args.partial_refine_partial_trim_quantile,
            partial_weight=args.partial_refine_partial_weight,
            max_step_translation=args.partial_refine_max_step_translation,
            min_step_scale=args.partial_refine_min_step_scale,
            max_step_scale=args.partial_refine_max_step_scale,
            iterations=args.partial_refine_iterations,
            lr=args.partial_refine_lr,
            distance_weight=args.partial_refine_distance_weight,
            transform_reg_weight=args.partial_refine_transform_reg_weight,
            axis_scale_reg_weight=args.partial_refine_objective_scale_reg_weight,
            seed=args.seed,
            device=args.device,
        )
        if partial_refinement.get("enabled") and partial_refinement.get("candidate_distance"):
            complete_to_partial, partial_refinement = choose_anisotropic_partial_refinement(
                complete_to_partial,
                partial_optimized,
                partial_refinement,
                min_distance_improvement=args.partial_refine_min_distance_improvement,
                min_delta_axis_scale=args.partial_refine_min_delta_axis_scale,
                max_delta_axis_scale=args.partial_refine_max_delta_axis_scale,
                max_delta_translation=args.partial_refine_max_delta_translation,
            )
    else:
        partial_optimized, partial_refinement = refine_symmetric_partial_icp(
            complete_to_partial,
            complete_points,
            partial_points,
            iterations=args.partial_refine_iterations,
            max_pairs=args.partial_refine_max_pairs,
            complete_trim_quantile=args.partial_refine_complete_trim_quantile,
            partial_trim_quantile=args.partial_refine_partial_trim_quantile,
            partial_weight=args.partial_refine_partial_weight,
            max_step_translation=args.partial_refine_max_step_translation,
            min_step_scale=args.partial_refine_min_step_scale,
            max_step_scale=args.partial_refine_max_step_scale,
            seed=args.seed,
        )
        if partial_refinement.get("enabled") and partial_refinement.get("candidate_distance"):
            complete_to_partial, partial_refinement = choose_partial_refinement(
                complete_to_partial,
                partial_optimized,
                partial_refinement,
                min_distance_improvement=args.partial_refine_min_distance_improvement,
                max_delta_rotation_deg=args.partial_refine_max_delta_rotation_deg,
                max_delta_translation=args.partial_refine_max_delta_translation,
                min_delta_scale=args.partial_refine_min_delta_scale,
                max_delta_scale=args.partial_refine_max_delta_scale,
            )
    complete_in_moge = deepcopy(complete_pcd)
    complete_in_moge.points = o3d.utility.Vector3dVector(apply_sim3(complete_points, complete_to_moge))
    complete_in_partial = deepcopy(complete_pcd)
    complete_in_partial.points = o3d.utility.Vector3dVector(
        apply_sim3(complete_points, complete_to_partial)
    )
    moge_pcd = o3d.geometry.PointCloud()
    moge_pcd.points = o3d.utility.Vector3dVector(object_moge.points)
    moge_pcd.colors = o3d.utility.Vector3dVector(object_moge.colors)

    uv, valid = project_points(np.asarray(complete_in_moge.points), intrinsic_px, image_shape)
    rendered_depth, rendered_mask = zbuffer_depth(
        uv[valid],
        np.asarray(complete_in_moge.points)[valid, 2],
        image_shape=image_shape,
        splat_radius=args.splat_radius,
    )

    prefix = out_dir / args.flag
    paths = {
        "moge_object": str(prefix.with_name(prefix.name + "_moge_object_only.ply")),
        "complete_to_moge": str(prefix.with_name(prefix.name + "_complete_registered_to_moge.ply")),
        "moge_fused": str(prefix.with_name(prefix.name + "_moge_gray_complete_blue_fused.ply")),
        "complete_to_partial": str(prefix.with_name(prefix.name + "_complete_aligned_to_raw_partial.ply")),
        "partial_fused": str(prefix.with_name(prefix.name + "_raw_partial_gray_complete_blue_aligned.ply")),
        "complete_to_moge_transform": str(prefix.with_name(prefix.name + "_complete_to_moge_transform.npy")),
        "complete_to_partial_transform": str(prefix.with_name(prefix.name + "_complete_to_partial_transform.npy")),
        "overlay": str(prefix.with_name(prefix.name + "_render_to_moge_overlay.png")),
        "target_depth": str(prefix.with_name(prefix.name + "_moge_target_depth.npy")),
        "rendered_depth": str(prefix.with_name(prefix.name + "_complete_rendered_depth.npy")),
        "info": str(prefix.with_name(prefix.name + "_render_to_moge_sim3_info.json")),
    }
    final_name = getattr(args, "final_name", None)
    if final_name:
        paths["final_prediction"] = str(out_dir / final_name)

    o3d.io.write_point_cloud(paths["moge_object"], moge_pcd)
    o3d.io.write_point_cloud(paths["complete_to_moge"], complete_in_moge)
    o3d.io.write_point_cloud(
        paths["moge_fused"],
        paint_merge(moge_pcd, [0.55, 0.55, 0.55], complete_in_moge, [0.0, 0.2, 1.0]),
    )
    o3d.io.write_point_cloud(paths["complete_to_partial"], complete_in_partial)
    if final_name:
        o3d.io.write_point_cloud(paths["final_prediction"], complete_in_partial)
    o3d.io.write_point_cloud(
        paths["partial_fused"],
        paint_merge(partial_pcd, [0.55, 0.55, 0.55], complete_in_partial, [0.0, 0.2, 1.0]),
    )
    np.save(paths["complete_to_moge_transform"], complete_to_moge)
    np.save(paths["complete_to_partial_transform"], complete_to_partial)
    np.save(paths["target_depth"], target_depth)
    np.save(paths["rendered_depth"], rendered_depth)
    draw_overlay(paths["overlay"], image_path, target_mask, rendered_mask)

    scale, rotation, translation = decompose_sim3(complete_to_moge)
    info = {
        "method": "render_to_moge_sim3_point_zbuffer_no_freereg",
        "flag": args.flag,
        "sample_dir": str(sample_dir),
        "out_dir": str(out_dir),
        "inputs": {
            "image": str(image_path),
            "complete": str(complete_path),
            "object_mask": str(mask_path),
            "moge_to_partial": str(moge_to_partial_path),
            "partial": str(partial_path),
        },
        "moge": moge_info,
        "moge_object_points": int(len(object_moge.points)),
        "complete_points": int(len(complete_points)),
        "eval_points": int(len(eval_points)),
        "initial": {
            "candidate_count": int(len(candidates)),
            "best_index": int(best_initial["index"]),
            "best_score": best_initial["score"],
        },
        "refined_score": refined_score,
        "final_score": final_score,
        "refine_history": refine_history,
        "silhouette_optimization": silhouette_optimization,
        "visible_3d_optimization": visible_3d_optimization,
        "visible_icp_history": icp_history,
        "visible_icp_acceptance": icp_acceptance,
        "retry_2d_search": retry_2d_search,
        "bridge_anchor_optimization": bridge_anchor_optimization,
        "post_retry_silhouette_optimization": post_retry_silhouette_optimization,
        "complete_to_moge": complete_to_moge.tolist(),
        "complete_to_moge_decomposed": {
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "moge_to_partial": moge_to_partial.tolist(),
        "registration_2d_acceptance": registration_2d_acceptance,
        "partial_refinement": partial_refinement,
        "complete_to_partial": complete_to_partial.tolist(),
        "outputs": paths,
    }
    save_json(paths["info"], info)
    print(json.dumps(info, indent=2))
    return info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", default="06145")
    parser.add_argument("--sample_root", default=str(DEFAULT_SAMPLE_ROOT))
    parser.add_argument("--sample_dir", default=None)
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--image_name", default=None)
    parser.add_argument("--complete_name", default=None)
    parser.add_argument("--object_mask_name", default=None)
    parser.add_argument("--moge_to_partial_name", default=None)
    parser.add_argument("--partial_path", default=None)
    parser.add_argument("--moge_model", default=str(DEFAULT_MOGE_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object_alpha_threshold", type=int, default=128)
    parser.add_argument("--object_mask_erode_pixels", type=int, default=0)
    parser.add_argument("--eval_points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--splat_radius", type=int, default=1)
    parser.add_argument("--scale_multipliers", default="0.55,0.7,0.85,1.0,1.15,1.3")
    parser.add_argument("--translation_steps", default="0.12,0.06,0.03,0.015")
    parser.add_argument("--rotation_steps_deg", default="12,6,3")
    parser.add_argument("--scale_steps", default="1.12,1.06,1.03")
    parser.add_argument("--refine_rounds", type=int, default=1)
    parser.add_argument("--silhouette_opt_iterations", type=int, default=80)
    parser.add_argument("--silhouette_opt_render_size", type=int, default=128)
    parser.add_argument("--silhouette_opt_max_points", type=int, default=12000)
    parser.add_argument("--silhouette_opt_lr", type=float, default=0.02)
    parser.add_argument("--silhouette_opt_splat_radius", type=int, default=1)
    parser.add_argument("--silhouette_opt_sigma", type=float, default=0.75)
    parser.add_argument("--silhouette_opt_opacity", type=float, default=0.08)
    parser.add_argument("--silhouette_opt_leakage_weight", type=float, default=1.25)
    parser.add_argument("--silhouette_opt_miss_weight", type=float, default=0.65)
    parser.add_argument("--silhouette_opt_outside_distance_weight", type=float, default=1.0)
    parser.add_argument("--silhouette_opt_depth_weight", type=float, default=0.0)
    parser.add_argument("--silhouette_opt_boundary_weight", type=float, default=0.0)
    parser.add_argument("--silhouette_opt_area_weight", type=float, default=0.15)
    parser.add_argument("--silhouette_opt_center_weight", type=float, default=2.0)
    parser.add_argument("--silhouette_opt_transform_reg_weight", type=float, default=0.02)
    parser.add_argument("--silhouette_opt_min_score_gain", type=float, default=0.0)
    parser.add_argument("--visible_3d_opt_iterations", type=int, default=80)
    parser.add_argument("--visible_3d_opt_max_pairs", type=int, default=12000)
    parser.add_argument("--visible_3d_opt_trim_quantile", type=float, default=0.7)
    parser.add_argument("--visible_3d_opt_max_depth_delta", type=float, default=0.015)
    parser.add_argument("--visible_3d_opt_lr", type=float, default=0.01)
    parser.add_argument("--visible_3d_opt_distance_loss", choices=("smooth_l1", "l2"), default="smooth_l1")
    parser.add_argument("--visible_3d_opt_distance_weight", type=float, default=1.0)
    parser.add_argument("--visible_3d_opt_silhouette_weight", type=float, default=0.35)
    parser.add_argument("--visible_3d_opt_silhouette_render_size", type=int, default=128)
    parser.add_argument("--visible_3d_opt_silhouette_points", type=int, default=12000)
    parser.add_argument("--visible_3d_opt_silhouette_splat_radius", type=int, default=1)
    parser.add_argument("--visible_3d_opt_silhouette_sigma", type=float, default=0.75)
    parser.add_argument("--visible_3d_opt_silhouette_opacity", type=float, default=0.08)
    parser.add_argument("--visible_3d_opt_leakage_weight", type=float, default=0.5)
    parser.add_argument("--visible_3d_opt_miss_weight", type=float, default=0.25)
    parser.add_argument("--visible_3d_opt_transform_reg_weight", type=float, default=0.02)
    parser.add_argument("--visible_3d_opt_max_score_drop", type=float, default=0.10)
    parser.add_argument("--visible_3d_opt_min_distance_improvement", type=float, default=0.02)
    parser.add_argument("--visible_icp_iterations", type=int, default=3)
    parser.add_argument("--icp_trim_quantile", type=float, default=0.7)
    parser.add_argument("--icp_max_pairs", type=int, default=20000)
    parser.add_argument("--icp_rollback_on_score_drop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--icp_min_score_gain", type=float, default=0.0)
    parser.add_argument("--require_2d_acceptance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_2d_iou", type=float, default=0.78)
    parser.add_argument("--min_2d_coverage", type=float, default=0.80)
    parser.add_argument("--max_2d_leakage", type=float, default=0.12)
    parser.add_argument("--min_2d_edge_iou", type=float, default=0.02)
    parser.add_argument("--max_2d_edge_chamfer_px", type=float, default=18.0)
    parser.add_argument("--retry_2d_on_gate_failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry_2d_top_k", type=int, default=12)
    parser.add_argument("--retry_2d_scale_multipliers", default="0.45,0.55,0.65,0.75,0.85,1.0,1.15,1.3,1.45,1.6")
    parser.add_argument("--retry_2d_translation_steps", default="0.12,0.06,0.03,0.015")
    parser.add_argument("--retry_2d_rotation_steps_deg", default="12,6,3")
    parser.add_argument("--retry_2d_scale_steps", default="1.12,1.06,1.03")
    parser.add_argument("--retry_2d_refine_rounds", type=int, default=1)
    parser.add_argument("--bridge_anchor_opt_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partial_to_moge_index_name", default=None)
    parser.add_argument("--bridge_anchor_opt_max_anchors", type=int, default=20000)
    parser.add_argument("--bridge_anchor_opt_iterations", type=int, default=80)
    parser.add_argument("--bridge_anchor_opt_max_pairs", type=int, default=12000)
    parser.add_argument("--bridge_anchor_opt_trim_quantile", type=float, default=0.45)
    parser.add_argument("--bridge_anchor_opt_lr", type=float, default=0.01)
    parser.add_argument("--bridge_anchor_opt_distance_loss", choices=("smooth_l1", "l2"), default="smooth_l1")
    parser.add_argument("--bridge_anchor_opt_distance_weight", type=float, default=0.20)
    parser.add_argument("--bridge_anchor_opt_silhouette_weight", type=float, default=1.0)
    parser.add_argument("--bridge_anchor_opt_silhouette_render_size", type=int, default=128)
    parser.add_argument("--bridge_anchor_opt_silhouette_points", type=int, default=12000)
    parser.add_argument("--bridge_anchor_opt_silhouette_splat_radius", type=int, default=1)
    parser.add_argument("--bridge_anchor_opt_silhouette_sigma", type=float, default=0.75)
    parser.add_argument("--bridge_anchor_opt_silhouette_opacity", type=float, default=0.08)
    parser.add_argument("--bridge_anchor_opt_leakage_weight", type=float, default=0.10)
    parser.add_argument("--bridge_anchor_opt_miss_weight", type=float, default=2.0)
    parser.add_argument("--bridge_anchor_opt_transform_reg_weight", type=float, default=0.10)
    parser.add_argument("--bridge_anchor_opt_min_2d_objective_gain", type=float, default=0.0)
    parser.add_argument("--bridge_anchor_opt_max_2d_objective_drop", type=float, default=0.02)
    parser.add_argument("--bridge_anchor_opt_min_anchor_improvement", type=float, default=0.01)
    parser.add_argument("--bridge_anchor_opt_retry_anchor_weight", type=float, default=0.35)
    parser.add_argument("--retry_continuous_opt_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry_continuous_opt_iterations", type=int, default=24)
    parser.add_argument("--retry_continuous_opt_max_pairs", type=int, default=6000)
    parser.add_argument("--retry_continuous_opt_trim_quantile", type=float, default=0.45)
    parser.add_argument("--retry_continuous_opt_lr", type=float, default=0.008)
    parser.add_argument("--retry_continuous_opt_distance_loss", choices=("smooth_l1", "l2"), default="smooth_l1")
    parser.add_argument("--retry_continuous_opt_distance_weight", type=float, default=0.20)
    parser.add_argument("--retry_continuous_opt_silhouette_weight", type=float, default=1.0)
    parser.add_argument("--retry_continuous_opt_silhouette_render_size", type=int, default=96)
    parser.add_argument("--retry_continuous_opt_silhouette_points", type=int, default=6000)
    parser.add_argument("--retry_continuous_opt_silhouette_splat_radius", type=int, default=1)
    parser.add_argument("--retry_continuous_opt_silhouette_sigma", type=float, default=0.75)
    parser.add_argument("--retry_continuous_opt_silhouette_opacity", type=float, default=0.08)
    parser.add_argument("--retry_continuous_opt_leakage_weight", type=float, default=0.10)
    parser.add_argument("--retry_continuous_opt_miss_weight", type=float, default=2.0)
    parser.add_argument("--retry_continuous_opt_transform_reg_weight", type=float, default=0.10)
    parser.add_argument("--retry_continuous_opt_max_2d_objective_drop", type=float, default=0.02)
    parser.add_argument("--retry_continuous_opt_min_anchor_improvement", type=float, default=0.005)
    parser.add_argument(
        "--partial_refine_mode",
        choices=("pca_anisotropic", "symmetric_icp", "continuous_affine"),
        default="pca_anisotropic",
    )
    parser.add_argument("--partial_refine_iterations", type=int, default=12)
    parser.add_argument("--partial_refine_max_pairs", type=int, default=6000)
    parser.add_argument("--partial_refine_complete_trim_quantile", type=float, default=0.05)
    parser.add_argument("--partial_refine_partial_trim_quantile", type=float, default=1.0)
    parser.add_argument("--partial_refine_partial_weight", type=float, default=8.0)
    parser.add_argument("--partial_refine_anisotropic_pre_icp_iterations", type=int, default=12)
    parser.add_argument("--partial_refine_anisotropic_icp_iterations", type=int, default=6)
    parser.add_argument(
        "--partial_refine_anisotropic_scale_triplets",
        default="1.0,0.85,1.0;1.08,0.85,1.08;1.0,0.85,1.08;1.08,0.95,1.08;0.95,0.85,1.0;1.0,0.85,0.95;0.95,0.85,0.95;1.08,0.85,1.0",
    )
    parser.add_argument("--partial_refine_objective_pc_p95_weight", type=float, default=0.30)
    parser.add_argument("--partial_refine_objective_cp70_weight", type=float, default=0.20)
    parser.add_argument("--partial_refine_objective_scale_reg_weight", type=float, default=0.004)
    parser.add_argument("--partial_refine_max_step_translation", type=float, default=0.08)
    parser.add_argument("--partial_refine_min_step_scale", type=float, default=0.85)
    parser.add_argument("--partial_refine_max_step_scale", type=float, default=1.15)
    parser.add_argument("--partial_refine_trim_quantile", type=float, default=0.35)
    parser.add_argument("--partial_refine_lr", type=float, default=0.01)
    parser.add_argument("--partial_refine_distance_loss", choices=("smooth_l1", "l2"), default="smooth_l1")
    parser.add_argument("--partial_refine_distance_weight", type=float, default=1.0)
    parser.add_argument("--partial_refine_transform_reg_weight", type=float, default=0.02)
    parser.add_argument("--partial_refine_min_distance_improvement", type=float, default=0.01)
    parser.add_argument("--partial_refine_max_delta_rotation_deg", type=float, default=12.0)
    parser.add_argument("--partial_refine_max_delta_translation", type=float, default=0.12)
    parser.add_argument("--partial_refine_min_delta_scale", type=float, default=0.9)
    parser.add_argument("--partial_refine_max_delta_scale", type=float, default=1.25)
    parser.add_argument("--partial_refine_min_delta_axis_scale", type=float, default=0.60)
    parser.add_argument("--partial_refine_max_delta_axis_scale", type=float, default=2.00)
    parser.add_argument("--final_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
