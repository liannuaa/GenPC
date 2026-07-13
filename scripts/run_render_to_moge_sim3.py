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
):
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
            for proposal in proposals:
                score = evaluate_transform(
                    source_points,
                    proposal,
                    intrinsic_px,
                    target_depth,
                    target_mask,
                    splat_radius=splat_radius,
                )
                if score["score"] > local_best_score["score"]:
                    local_best_score = score
                    local_best_transform = proposal
            if local_best_score["score"] > best_score["score"] + 1e-8:
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


def render_soft_silhouette_torch(points, intrinsic_px, image_shape, render_size, splat_radius, sigma, opacity):
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
    if uv.numel() == 0:
        return torch.zeros((render_size, render_size), dtype=points.dtype, device=points.device)

    base = torch.floor(uv).long()
    flat_size = render_size * render_size
    density = torch.zeros(flat_size, dtype=points.dtype, device=points.device)
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
    density = density.reshape(render_size, render_size)
    return 1.0 - torch.exp(-float(opacity) * density)


def optimize_silhouette_delta_sim3(
    initial_transform,
    source_points,
    intrinsic_px,
    image_shape,
    target_mask,
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

    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    base = torch.as_tensor(base_points, dtype=dtype, device=torch_device)
    intrinsic = torch.as_tensor(np.asarray(intrinsic_px, dtype=np.float32), dtype=dtype, device=torch_device)
    target = torch.as_tensor(target_np, dtype=dtype, device=torch_device)
    outside = torch.as_tensor(outside_distance.astype(np.float32), dtype=dtype, device=torch_device)
    target_sum = torch.clamp(target.sum(), min=1.0)
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
        silhouette = render_soft_silhouette_torch(
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
        "best": best,
        "delta": delta.tolist(),
    }
    return optimized, info


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
    silhouette_optimized, silhouette_optimization = optimize_silhouette_delta_sim3(
        refined,
        complete_points,
        intrinsic_px,
        image_shape,
        target_mask,
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
        area_weight=args.silhouette_opt_area_weight,
        center_weight=args.silhouette_opt_center_weight,
        transform_reg_weight=args.silhouette_opt_transform_reg_weight,
        seed=args.seed,
        device=args.device,
    )
    if silhouette_optimization.get("enabled"):
        silhouette_score = evaluate_transform(
            eval_points,
            silhouette_optimized,
            intrinsic_px,
            target_depth,
            target_mask,
            splat_radius=args.splat_radius,
        )
        min_gain = float(args.silhouette_opt_min_score_gain)
        if silhouette_score["score"] > refined_score["score"] + min_gain:
            refined = silhouette_optimized
            silhouette_optimization.update(
                {
                    "accepted": True,
                    "reason": "render_score_improved",
                    "min_score_gain": min_gain,
                    "baseline_score": refined_score,
                    "candidate_score": silhouette_score,
                }
            )
            refined_score = silhouette_score
        else:
            silhouette_optimization.update(
                {
                    "accepted": False,
                    "reason": "render_score_not_improved",
                    "min_score_gain": min_gain,
                    "baseline_score": refined_score,
                    "candidate_score": silhouette_score,
                }
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
    partial_optimized, partial_refinement = optimize_partial_delta_sim3(
        complete_to_partial,
        complete_points,
        partial_points,
        max_pairs=args.partial_refine_max_pairs,
        trim_quantile=args.partial_refine_trim_quantile,
        iterations=args.partial_refine_iterations,
        lr=args.partial_refine_lr,
        distance_loss=args.partial_refine_distance_loss,
        distance_weight=args.partial_refine_distance_weight,
        transform_reg_weight=args.partial_refine_transform_reg_weight,
        seed=args.seed,
        device=args.device,
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
        "complete_to_moge": complete_to_moge.tolist(),
        "complete_to_moge_decomposed": {
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "moge_to_partial": moge_to_partial.tolist(),
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
    parser.add_argument("--partial_refine_iterations", type=int, default=80)
    parser.add_argument("--partial_refine_max_pairs", type=int, default=12000)
    parser.add_argument("--partial_refine_trim_quantile", type=float, default=0.35)
    parser.add_argument("--partial_refine_lr", type=float, default=0.01)
    parser.add_argument("--partial_refine_distance_loss", choices=("smooth_l1", "l2"), default="smooth_l1")
    parser.add_argument("--partial_refine_distance_weight", type=float, default=1.0)
    parser.add_argument("--partial_refine_transform_reg_weight", type=float, default=0.02)
    parser.add_argument("--partial_refine_min_distance_improvement", type=float, default=0.01)
    parser.add_argument("--partial_refine_max_delta_rotation_deg", type=float, default=12.0)
    parser.add_argument("--partial_refine_max_delta_translation", type=float, default=0.12)
    parser.add_argument("--partial_refine_min_delta_scale", type=float, default=0.9)
    parser.add_argument("--partial_refine_max_delta_scale", type=float, default=1.1)
    parser.add_argument("--final_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
