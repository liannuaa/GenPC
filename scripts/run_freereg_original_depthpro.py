import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEREG_ROOT = PROJECT_ROOT / "third_party" / "FreeReg"
LEGACY_FREEREG_ROOT = PROJECT_ROOT.parent / "FreeReg"
DEFAULT_FALLBACK_IR_3D = (0.10, 0.20)
DEFAULT_MAX_COMPLETE_TO_IMAGE_TRANSLATION = 50.0


def add_freereg_to_path(freereg_root):
    freereg_root = Path(freereg_root).resolve()
    if not freereg_root.exists() and LEGACY_FREEREG_ROOT.exists():
        freereg_root = LEGACY_FREEREG_ROOT.resolve()

    depthpro_src = freereg_root / "tools" / "DepthPro" / "src"
    for path in (str(depthpro_src), str(freereg_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return freereg_root


def load_object_mask(mask_path, image_shape, threshold, erode_pixels):
    mask = Image.open(mask_path).convert("L")
    width, height = image_shape[1], image_shape[0]
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    mask_np = np.asarray(mask) >= int(threshold)
    if erode_pixels > 0:
        kernel_size = int(erode_pixels) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_np = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask_np


def image_to_masked_depthpro_points(pipe, image, object_mask, max_points, seed):
    from Utils.utils import edge_filter

    height, width = image.shape[:2]
    pipe.H = height
    pipe.W = width
    depth, _, _, intrinsic = pipe.depthpro(image)
    sky = depth > 199.0
    xyz = pipe.projector.proj_depth(depth, intrinsic, depth_unit=1.0)
    edge = edge_filter(depth, sky, times=0.05).reshape(-1)
    valid = (~edge) & (~sky.reshape(-1)) & object_mask.reshape(-1)
    xyz = xyz[valid]
    if len(xyz) == 0:
        raise RuntimeError("Object mask removed all DepthPro points.")
    if len(xyz) > max_points:
        rng = np.random.default_rng(int(seed))
        xyz = xyz[rng.permutation(len(xyz))[: int(max_points)]]
    return xyz, intrinsic, {"height": int(height), "width": int(width), "valid_object_points": int(len(xyz))}


def write_colored_pcd(path, points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.paint_uniform_color(color)
    o3d.io.write_point_cloud(str(path), pcd)
    return pcd


def sim3_matrix(scale, transform):
    matrix = np.asarray(transform, dtype=np.float64).copy()
    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[:3, :3] *= float(scale)
    return matrix @ scale_matrix


def apply_matrix(points, transform):
    points = np.asarray(points, dtype=np.float64)
    hom = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ hom.T).T[:, :3]


def project_points(points, intrinsic, image_size):
    points = np.asarray(points, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    width, height = image_size
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    camera = (intrinsic @ points.T).T
    projected[valid] = camera[valid, :2] / camera[valid, 2:3]
    valid &= (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    return projected, valid


def projected_silhouette(points, intrinsic, image_size, dilate_pixels=2):
    width, height = image_size
    uv, valid = project_points(points, intrinsic, image_size)
    mask = np.zeros((height, width), dtype=np.uint8)
    if valid.any():
        xy = np.rint(uv[valid]).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
        mask[xy[:, 1], xy[:, 0]] = 1
    if dilate_pixels > 0 and mask.any():
        kernel_size = int(dilate_pixels) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def mask_boundary(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not mask_u8.any():
        return mask_u8.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (mask_u8 > 0) & (eroded == 0)


def boundary_chamfer_pixels(source_mask, target_mask):
    source_edge = mask_boundary(source_mask)
    target_edge = mask_boundary(target_mask)
    if not source_edge.any() or not target_edge.any():
        return float("inf")
    distance = cv2.distanceTransform((~target_edge).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[source_edge].mean())


def score_projected_silhouette(projected_mask, target_mask, edge_weight=0.15, leakage_weight=0.25):
    projected_mask = np.asarray(projected_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    intersection = projected_mask & target_mask
    union = projected_mask | target_mask
    projected_count = int(projected_mask.sum())
    target_count = int(target_mask.sum())
    iou = float(intersection.sum() / max(union.sum(), 1))
    coverage = float(intersection.sum() / max(target_count, 1))
    leakage = float((projected_mask & ~target_mask).sum() / max(projected_count, 1))
    forward_edge = boundary_chamfer_pixels(projected_mask, target_mask)
    backward_edge = boundary_chamfer_pixels(target_mask, projected_mask)
    if np.isfinite(forward_edge) and np.isfinite(backward_edge):
        edge_chamfer = 0.5 * (forward_edge + backward_edge)
        diagonal = max(np.linalg.norm(target_mask.shape), 1.0)
        edge_chamfer_norm = float(edge_chamfer / diagonal)
    else:
        edge_chamfer = float("inf")
        edge_chamfer_norm = 1.0
    score = float(iou + 0.25 * coverage - leakage_weight * leakage - edge_weight * edge_chamfer_norm)
    return {
        "score": score,
        "iou": iou,
        "coverage": coverage,
        "leakage": leakage,
        "edge_chamfer_px": edge_chamfer,
        "edge_chamfer_norm": edge_chamfer_norm,
        "projected_pixels": projected_count,
        "target_pixels": target_count,
        "intersection_pixels": int(intersection.sum()),
    }


def draw_silhouette_overlay(path, image, target_mask, projected_mask):
    image_uint8 = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    overlay = image_uint8.copy()
    target = np.asarray(target_mask, dtype=bool)
    projected = np.asarray(projected_mask, dtype=bool)
    overlay[target] = (0.55 * overlay[target] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[projected] = (0.55 * overlay[projected] + 0.45 * np.array([0, 80, 255])).astype(np.uint8)
    overlap = target & projected
    overlay[overlap] = (0.35 * overlay[overlap] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)
    Image.fromarray(overlay).save(path)
    return str(path)


def render_projected_points(points, intrinsic, image_size, max_points=60000, seed=1184):
    width, height = image_size
    points = np.asarray(points, dtype=np.float64)
    if len(points) > max_points:
        rng = np.random.default_rng(int(seed))
        points = points[rng.choice(len(points), int(max_points), replace=False)]
    uv, valid = project_points(points, intrinsic, image_size)
    uv = np.rint(uv[valid]).astype(np.int32)
    uv[:, 0] = np.clip(uv[:, 0], 0, width - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height - 1)
    z = points[valid, 2]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if len(uv) == 0:
        return Image.fromarray(canvas)

    order = np.argsort(z)[::-1]
    uv = uv[order]
    z = z[order]
    z_min, z_max = np.percentile(z, [2, 98])
    z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    colors = np.stack(
        [
            np.full_like(z_norm, 45),
            90 + (1.0 - z_norm) * 80,
            170 + (1.0 - z_norm) * 70,
        ],
        axis=1,
    ).astype(np.uint8)
    canvas[uv[:, 1], uv[:, 0]] = colors
    return Image.fromarray(canvas)


def sample_feature_grid_nearest(feature_grid, uv, image_size):
    feature_grid = np.asarray(feature_grid, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float64)
    width, height = image_size
    grid_h, grid_w = feature_grid.shape[:2]
    valid = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    xi = np.floor(uv[:, 0] / max(float(width), 1.0) * grid_w).astype(np.int64)
    yi = np.floor(uv[:, 1] / max(float(height), 1.0) * grid_h).astype(np.int64)
    xi = np.clip(xi, 0, grid_w - 1)
    yi = np.clip(yi, 0, grid_h - 1)
    sampled = feature_grid[yi, xi]
    sampled[~valid] = 0.0
    norms = np.linalg.norm(sampled, axis=1, keepdims=True)
    sampled = sampled / np.maximum(norms, 1e-12)
    return sampled, valid


def rank_matches_by_semantic_features(feature_grid, image_uv, complete_uv, image_size, min_similarity, max_matches):
    image_features, image_valid = sample_feature_grid_nearest(feature_grid, image_uv, image_size)
    complete_features, complete_valid = sample_feature_grid_nearest(feature_grid, complete_uv, image_size)
    scores = np.sum(image_features * complete_features, axis=1).astype(np.float32)
    valid = image_valid & complete_valid & np.isfinite(scores) & (scores >= float(min_similarity))
    indices = np.flatnonzero(valid)
    if len(indices) > 0:
        order = np.argsort(-scores[indices])
        indices = indices[order][: int(max_matches)]
    return {
        "indices": indices.astype(np.int64),
        "scores": scores[indices].astype(np.float32),
        "valid_matches": int(valid.sum()),
        "total_matches": int(len(scores)),
        "min_similarity": float(min_similarity),
        "max_matches": int(max_matches),
    }


def load_semantic_feature_grid(path):
    data = np.load(path, allow_pickle=False)
    feature_grid = np.asarray(data["features"], dtype=np.float32)
    image_size = tuple(int(v) for v in np.asarray(data["image_size"]).tolist())
    metadata = {}
    for key in data.files:
        if key not in {"features", "image_size"}:
            value = data[key]
            metadata[key] = value.tolist() if hasattr(value, "tolist") else str(value)
    return feature_grid, image_size, metadata


def draw_freereg_match_figure(
    path,
    image,
    registered_complete_points,
    image_kpt_uvs,
    complete_kpts,
    matches,
    complete_to_image,
    intrinsic,
    max_lines,
    seed,
    selected_match_indices=None,
):
    image_uint8 = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    height, width = image_uint8.shape[:2]
    image_size = (width, height)

    left = Image.fromarray(image_uint8).convert("RGB")
    right = render_projected_points(
        registered_complete_points,
        intrinsic,
        image_size,
        seed=seed,
    ).convert("RGB")
    combined = Image.new("RGB", (width * 2, height), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(right, (width, 0))
    draw = ImageDraw.Draw(combined, "RGBA")

    if selected_match_indices is None:
        match_indices = np.arange(len(matches), dtype=np.int64)
    else:
        match_indices = np.asarray(selected_match_indices, dtype=np.int64)
    match_indices = match_indices[(match_indices >= 0) & (match_indices < len(matches))]

    selected_matches = matches[match_indices]
    matched_image_uv = np.asarray(image_kpt_uvs, dtype=np.float64)[selected_matches[:, 0]]
    matched_complete = np.asarray(complete_kpts, dtype=np.float64)[selected_matches[:, 1]]
    matched_complete_image = apply_matrix(matched_complete, complete_to_image)
    matched_complete_uv, complete_valid = project_points(matched_complete_image, intrinsic, image_size)
    image_valid = (
        np.isfinite(matched_image_uv).all(axis=1)
        & (matched_image_uv[:, 0] >= 0)
        & (matched_image_uv[:, 0] < width)
        & (matched_image_uv[:, 1] >= 0)
        & (matched_image_uv[:, 1] < height)
    )
    valid = image_valid & complete_valid
    valid_indices = np.flatnonzero(valid)
    if selected_match_indices is None and len(valid_indices) > 0:
        reproj_distance = np.linalg.norm(
            matched_image_uv[valid_indices] - matched_complete_uv[valid_indices],
            axis=1,
        )
        valid_indices = valid_indices[np.argsort(reproj_distance)]
        valid_indices = valid_indices[: int(max_lines)]
    elif len(valid_indices) > 0:
        valid_indices = valid_indices[: int(max_lines)]

    rng = np.random.default_rng(int(seed))
    colors = rng.integers(40, 235, size=(max(len(valid_indices), 1), 3), dtype=np.uint8)
    radius = 3
    for draw_index, match_index in enumerate(valid_indices):
        color = tuple(int(v) for v in colors[draw_index]) + (185,)
        left_xy = matched_image_uv[match_index]
        right_xy = matched_complete_uv[match_index] + np.array([width, 0], dtype=np.float64)
        left_xy = tuple(float(v) for v in left_xy)
        right_xy = tuple(float(v) for v in right_xy)
        draw.line([left_xy, right_xy], fill=color, width=1)
        for xy in (left_xy, right_xy):
            x, y = xy
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline=color,
                fill=color[:3] + (210,),
            )

    path = Path(path)
    combined.save(path)
    return {
        "path": str(path),
        "valid_projected_matches": int(len(valid_indices)),
        "input_matches": int(len(match_indices)),
        "total_matches": int(len(matches)),
        "max_lines": int(max_lines),
    }


def parse_float_list(value):
    if value is None:
        return []
    return [float(item) for item in str(value).split(",") if item.strip()]


def build_ir_3d_candidates(auto_ir_3d, explicit_ir_3d, fallback_ir_3d):
    if explicit_ir_3d is not None:
        return [{"label": "explicit", "ir_3d": float(explicit_ir_3d)}]

    candidates = [{"label": "auto", "ir_3d": float(auto_ir_3d)}]
    for value in fallback_ir_3d:
        value = float(value)
        if not any(np.isclose(value, item["ir_3d"]) for item in candidates):
            candidates.append({"label": f"fallback_{value:g}", "ir_3d": value})
    return candidates


def transform_translation_norm(transform):
    transform = np.asarray(transform, dtype=np.float64)
    return float(np.linalg.norm(transform[:3, 3]))


def estimate_freereg_sim3(
    pipe,
    image_kpt_uvs,
    image_kpts,
    complete_kpts,
    matches,
    ir_3d,
    min_hypotheses,
    max_complete_to_image_translation,
):
    solver = pipe.solver
    solver.ird_3d = float(ir_3d)
    solver.ird_2d = max(10, (pipe.H + pipe.W) / 200.0) if pipe.ir_2d is None else pipe.ir_2d

    matched_image_uvs = image_kpt_uvs[matches[:, 0]]
    matched_image_kpts = image_kpts[matches[:, 0]]
    matched_complete_kpts = complete_kpts[matches[:, 1]]
    scales, hypotheses = solver.gen_hypos(
        matched_image_kpts,
        matched_complete_kpts,
        solver.iters,
        solver.ird_3d,
        np_per_hypo=solver.np_per_hypo,
    )
    hypothesis_count = int(len(scales))
    diagnostics = {
        "ir_3d": float(solver.ird_3d),
        "ir_2d": float(solver.ird_2d),
        "hypotheses": hypothesis_count,
        "valid": False,
    }
    if hypothesis_count < int(min_hypotheses):
        diagnostics["reject_reason"] = "not_enough_hypotheses"
        return diagnostics

    scale, image_to_complete_rigid = solver.ransac(
        matched_image_uvs,
        matched_image_kpts,
        matched_complete_kpts,
        scales,
        hypotheses,
        thres2d=solver.ird_2d,
        thres3d=solver.ird_3d,
    )
    image_to_complete = sim3_matrix(scale, image_to_complete_rigid)
    complete_to_image = np.linalg.inv(image_to_complete)
    complete_to_image_translation = transform_translation_norm(complete_to_image)
    diagnostics.update(
        {
            "valid": bool(np.isfinite(image_to_complete).all()),
            "freereg_scale": float(scale),
            "image_to_complete_translation_norm": transform_translation_norm(image_to_complete),
            "complete_to_image_translation_norm": complete_to_image_translation,
            "image_to_complete_rigid": image_to_complete_rigid,
            "image_to_complete": image_to_complete,
            "complete_to_image": complete_to_image,
        }
    )
    if not diagnostics["valid"]:
        diagnostics["reject_reason"] = "nonfinite_transform"
    elif complete_to_image_translation > float(max_complete_to_image_translation):
        diagnostics["valid"] = False
        diagnostics["reject_reason"] = "complete_to_image_translation_too_large"
    return diagnostics


def json_ready_candidate(candidate):
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"image_to_complete_rigid", "image_to_complete", "complete_to_image"}
    }


def run(args):
    freereg_root = add_freereg_to_path(args.freereg_root)
    from demo import Pipe

    sample_dir = Path(args.sample_dir)
    image_path = sample_dir / args.image_name
    complete_path = sample_dir / args.complete_name
    mask_path = sample_dir / args.object_mask_name
    out_prefix = sample_dir / args.output_prefix

    np.random.seed(int(args.seed))
    pipe = Pipe(args.nkpts, args.vs, args.w_2d, args.ir_2d, args.ir_3d)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    object_mask = load_object_mask(
        mask_path,
        image.shape,
        threshold=args.mask_threshold,
        erode_pixels=args.extra_erode_pixels,
    )
    image_pc, intrinsic, depthpro_info = image_to_masked_depthpro_points(
        pipe,
        image,
        object_mask,
        max_points=args.max_depthpro_points,
        seed=args.seed,
    )

    complete_pcd = o3d.io.read_point_cloud(str(complete_path))
    complete_points = np.asarray(complete_pcd.points, dtype=np.float64)
    if len(complete_points) == 0:
        raise RuntimeError(f"Empty complete point cloud: {complete_path}")

    image_pc, complete_for_reg = pipe._coarse_align(image_pc, complete_points.copy())
    pipe._determine_vs(image_pc, complete_for_reg)
    print(
        f"[FreeReg masked] image_object_points={len(image_pc)} "
        f"complete={len(complete_for_reg)} vs={pipe.vs:.8f}"
    )

    image_kpts, image_feats = pipe._extract_yoho(image_pc, pipe.nkpts)
    image_kpt_uvs, _ = pipe.projector.proj_3to2(image_kpts, intrinsic, np.eye(4))
    complete_kpts, complete_feats = pipe._extract_yoho(complete_for_reg, pipe.nkpts)
    matches = pipe._match(image_feats, complete_feats).astype(np.int16)
    print(f"[FreeReg masked] matches={len(matches)}")
    if len(matches) < 4:
        raise RuntimeError("Not enough FreeReg descriptor matches.")

    pipe.solver.set_intrinsic(intrinsic)
    auto_ir_3d = pipe.vs * 5
    candidates = build_ir_3d_candidates(
        auto_ir_3d=auto_ir_3d,
        explicit_ir_3d=args.ir_3d,
        fallback_ir_3d=parse_float_list(args.fallback_ir_3d),
    )
    candidate_rng_state = np.random.get_state()
    candidate_results = []
    for candidate in candidates:
        np.random.set_state(candidate_rng_state)
        result = estimate_freereg_sim3(
            pipe,
            image_kpt_uvs,
            image_kpts,
            complete_kpts,
            matches,
            ir_3d=candidate["ir_3d"],
            min_hypotheses=args.min_hypotheses,
            max_complete_to_image_translation=args.max_complete_to_image_translation,
        )
        result["label"] = candidate["label"]
        if result["valid"] and args.candidate_selection == "silhouette":
            candidate_registered = apply_matrix(complete_for_reg, result["complete_to_image"])
            candidate_mask = projected_silhouette(
                candidate_registered,
                intrinsic,
                (image.shape[1], image.shape[0]),
                dilate_pixels=args.silhouette_dilate_pixels,
            )
            result["silhouette_score"] = score_projected_silhouette(
                candidate_mask,
                object_mask,
                edge_weight=args.silhouette_edge_weight,
                leakage_weight=args.silhouette_leakage_weight,
            )
        candidate_results.append(result)
        if result["valid"] and args.candidate_selection == "first_valid":
            break
    valid_results = [item for item in candidate_results if item["valid"]]
    if args.candidate_selection == "silhouette" and valid_results:
        selected = max(valid_results, key=lambda item: item.get("silhouette_score", {}).get("score", -float("inf")))
    else:
        selected = valid_results[0] if valid_results else None
    if selected is None:
        raise RuntimeError(
            "FreeReg failed to produce a valid non-random transform. "
            f"Candidates: {[json_ready_candidate(item) for item in candidate_results]}"
        )

    freereg_scale = selected["freereg_scale"]
    image_to_complete_rigid = selected["image_to_complete_rigid"]
    image_to_complete = selected["image_to_complete"]
    complete_to_image = selected["complete_to_image"]
    registered_complete_points = apply_matrix(complete_for_reg, complete_to_image)

    image_points_path = Path(str(out_prefix) + "_object_depthpro_points.ply")
    registered_path = Path(str(out_prefix) + "_complete_registered_to_object_depthpro.ply")
    fused_path = Path(str(out_prefix) + "_gray_object_depthpro_blue_complete_fused.ply")
    match_figure_path = Path(str(out_prefix) + "_image_pointcloud_match_lines.png")
    semantic_match_figure_path = Path(str(out_prefix) + "_dino_semantic_match_lines.png")
    silhouette_overlay_path = Path(str(out_prefix) + "_silhouette_overlay.png")
    info_path = Path(str(out_prefix) + "_info.json")

    image_pcd = write_colored_pcd(image_points_path, image_pc, [0.55, 0.55, 0.55])
    registered_pcd = write_colored_pcd(registered_path, registered_complete_points, [0.0, 0.25, 1.0])
    o3d.io.write_point_cloud(str(fused_path), image_pcd + registered_pcd)
    match_figure_info = draw_freereg_match_figure(
        match_figure_path,
        image=image,
        registered_complete_points=registered_complete_points,
        image_kpt_uvs=image_kpt_uvs,
        complete_kpts=complete_kpts,
        matches=matches,
        complete_to_image=complete_to_image,
        intrinsic=intrinsic,
        max_lines=args.max_match_lines,
        seed=args.seed,
    )
    semantic_match_info = None
    if args.semantic_feature_name:
        feature_path = sample_dir / args.semantic_feature_name
        feature_grid, feature_image_size, feature_metadata = load_semantic_feature_grid(feature_path)
        if tuple(feature_image_size) != (image.shape[1], image.shape[0]):
            raise RuntimeError(
                f"Semantic feature image size {feature_image_size} does not match input image "
                f"{(image.shape[1], image.shape[0])}."
            )
        matched_image_uv = np.asarray(image_kpt_uvs, dtype=np.float64)[matches[:, 0]]
        matched_complete = np.asarray(complete_kpts, dtype=np.float64)[matches[:, 1]]
        matched_complete_image = apply_matrix(matched_complete, complete_to_image)
        matched_complete_uv, _ = project_points(matched_complete_image, intrinsic, feature_image_size)
        semantic_ranking = rank_matches_by_semantic_features(
            feature_grid=feature_grid,
            image_uv=matched_image_uv,
            complete_uv=matched_complete_uv,
            image_size=feature_image_size,
            min_similarity=args.semantic_min_similarity,
            max_matches=args.semantic_max_matches,
        )
        semantic_figure = draw_freereg_match_figure(
            semantic_match_figure_path,
            image=image,
            registered_complete_points=registered_complete_points,
            image_kpt_uvs=image_kpt_uvs,
            complete_kpts=complete_kpts,
            matches=matches,
            complete_to_image=complete_to_image,
            intrinsic=intrinsic,
            max_lines=args.max_match_lines,
            seed=args.seed,
            selected_match_indices=semantic_ranking["indices"],
        )
        semantic_match_info = {
            "feature_path": str(feature_path),
            "feature_metadata": feature_metadata,
            "ranking": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in semantic_ranking.items()
            },
            "figure": semantic_figure,
        }
    silhouette_overlay_info = None
    if args.candidate_selection == "silhouette":
        selected_mask = projected_silhouette(
            registered_complete_points,
            intrinsic,
            (image.shape[1], image.shape[0]),
            dilate_pixels=args.silhouette_dilate_pixels,
        )
        overlay_path = draw_silhouette_overlay(
            silhouette_overlay_path,
            image=image,
            target_mask=object_mask,
            projected_mask=selected_mask,
        )
        silhouette_overlay_info = {
            "path": overlay_path,
            "selected_score": selected.get("silhouette_score"),
            "dilate_pixels": int(args.silhouette_dilate_pixels),
        }

    info = {
        "method": "original_F-FreeReg_DepthPro_YOHO_Kabsch_object_masked_adaptive_ir3d",
        "freereg_root": str(freereg_root),
        "image": str(image_path),
        "complete_point_cloud": str(complete_path),
        "object_mask": str(mask_path),
        "mask_threshold": int(args.mask_threshold),
        "extra_erode_pixels": int(args.extra_erode_pixels),
        "max_depthpro_points": int(args.max_depthpro_points),
        "nkpts": int(pipe.nkpts),
        "w_2d": float(pipe.w_2d),
        "vs": float(pipe.vs),
        "auto_ir_3d": float(auto_ir_3d),
        "selected_ir_3d": float(selected["ir_3d"]),
        "selected_ir_3d_label": selected["label"],
        "candidate_selection": args.candidate_selection,
        "silhouette_dilate_pixels": int(args.silhouette_dilate_pixels),
        "silhouette_edge_weight": float(args.silhouette_edge_weight),
        "silhouette_leakage_weight": float(args.silhouette_leakage_weight),
        "min_hypotheses": int(args.min_hypotheses),
        "max_complete_to_image_translation": float(args.max_complete_to_image_translation),
        "freereg_candidates": [json_ready_candidate(item) for item in candidate_results],
        "depthpro": depthpro_info,
        "complete_points": int(len(complete_for_reg)),
        "image_keypoints": int(len(image_kpts)),
        "complete_keypoints": int(len(complete_kpts)),
        "matches": int(len(matches)),
        "fixed_uv": True,
        "freereg_scale": float(freereg_scale),
        "intrinsic": intrinsic.tolist(),
        "image_to_complete_rigid": image_to_complete_rigid.tolist(),
        "image_to_complete": image_to_complete.tolist(),
        "complete_to_image": complete_to_image.tolist(),
        "outputs": {
            "object_depthpro_points": str(image_points_path),
            "registered_complete": str(registered_path),
            "fused": str(fused_path),
            "match_figure": str(match_figure_path),
            "semantic_match_figure": None if semantic_match_info is None else str(semantic_match_figure_path),
            "silhouette_overlay": None if silhouette_overlay_info is None else str(silhouette_overlay_path),
        },
        "match_figure": match_figure_info,
        "semantic_match_figure": semantic_match_info,
        "silhouette_overlay": silhouette_overlay_info,
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--image-name", default="img.png")
    parser.add_argument("--complete-name", default=None)
    parser.add_argument("--object-mask-name", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--freereg-root", default=str(DEFAULT_FREEREG_ROOT))
    parser.add_argument("--nkpts", type=int, default=5000)
    parser.add_argument("--vs", type=float, default=None)
    parser.add_argument("--w_2d", type=float, default=0.5)
    parser.add_argument("--ir_2d", type=int, default=None)
    parser.add_argument("--ir_3d", type=float, default=None)
    parser.add_argument(
        "--fallback-ir-3d",
        default=",".join(str(value) for value in DEFAULT_FALLBACK_IR_3D),
        help="Comma-separated fallback 3D inlier thresholds used when auto ir_3d has too few hypotheses.",
    )
    parser.add_argument("--min-hypotheses", type=int, default=2)
    parser.add_argument(
        "--max-complete-to-image-translation",
        type=float,
        default=DEFAULT_MAX_COMPLETE_TO_IMAGE_TRANSLATION,
    )
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--extra-erode-pixels", type=int, default=0)
    parser.add_argument("--max-depthpro-points", type=int, default=50000)
    parser.add_argument("--max-match-lines", type=int, default=200)
    parser.add_argument("--candidate-selection", choices=["first_valid", "silhouette"], default="first_valid")
    parser.add_argument("--silhouette-dilate-pixels", type=int, default=2)
    parser.add_argument("--silhouette-edge-weight", type=float, default=0.15)
    parser.add_argument("--silhouette-leakage-weight", type=float, default=0.25)
    parser.add_argument("--semantic-feature-name", default=None)
    parser.add_argument("--semantic-min-similarity", type=float, default=0.55)
    parser.add_argument("--semantic-max-matches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1184)
    args = parser.parse_args()
    sample_dir = Path(args.sample_dir)
    if args.complete_name is None:
        args.complete_name = f"{sample_dir.name}_hunyuan2.1.ply"
    if args.output_prefix is None:
        args.output_prefix = f"{sample_dir.name}_freereg_original_depthpro_fixeduv_sim3"
    return args


if __name__ == "__main__":
    run(parse_args())
