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
    partial_pcd, _ = load_point_cloud(partial_path)
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
        "visible_icp_history": icp_history,
        "visible_icp_acceptance": icp_acceptance,
        "complete_to_moge": complete_to_moge.tolist(),
        "complete_to_moge_decomposed": {
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "moge_to_partial": moge_to_partial.tolist(),
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
    parser.add_argument("--visible_icp_iterations", type=int, default=3)
    parser.add_argument("--icp_trim_quantile", type=float, default=0.7)
    parser.add_argument("--icp_max_pairs", type=int, default=20000)
    parser.add_argument("--icp_rollback_on_score_drop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--icp_min_score_gain", type=float, default=0.0)
    parser.add_argument("--final_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
