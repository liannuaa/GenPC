import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_image_feature_registration_probe import (
    _moge_points_with_pixels,
    _partial_pixel_lookup,
)
from scripts.run_partial2partial_registration_probe import (
    nearest_neighbor_stats,
    voxel_downsample_to_count,
)


DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "workspace" / "scansalon_car_partial2partial_inputs" / "car__394"
DEFAULT_MOGE_MODEL = PROJECT_ROOT / "models" / "moge-2-vitl"


def _read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _source_mask_from_uv(point_uv, image_size, radius=5):
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    xy = np.rint(np.asarray(point_uv) * (image_size - 1)).astype(np.int32)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < image_size) & (xy[:, 1] >= 0) & (xy[:, 1] < image_size)
    for x, y in xy[valid]:
        cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
    return mask.astype(bool)


def _foreground_mask(image, white_threshold=248):
    return ~((image >= int(white_threshold)).all(axis=-1))


def _load_dinov2(model_name, device):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    patch_size = int(getattr(model.config, "patch_size", 14))
    return model, patch_size


def _extract_patch_features(model, image_rgb, image_size, device):
    image = Image.fromarray(image_rgb)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensor = transform(image)[None].to(device)
    with torch.no_grad():
        output = model(pixel_values=tensor)
    tokens = output.last_hidden_state[:, 1:, :]
    dim = tokens.shape[-1]
    grid = int(round(tokens.shape[1] ** 0.5))
    features = tokens.reshape(1, grid, grid, dim).permute(0, 3, 1, 2)
    features = F.normalize(features, p=2, dim=1)[0].permute(1, 2, 0).cpu().numpy()
    return features


def _patch_centers(grid, image_size):
    step = image_size / float(grid)
    xs = (np.arange(grid) + 0.5) * step
    ys = (np.arange(grid) + 0.5) * step
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def _sample_feature_grid(features, centers, mask, min_mask_coverage=0.2):
    grid = features.shape[0]
    image_size = mask.shape[0]
    step = image_size / float(grid)
    selected_features = []
    selected_centers = []
    half = max(1, int(step * 0.5))
    for feat, (x, y) in zip(features.reshape(-1, features.shape[-1]), centers):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - half), min(image_size, xi + half + 1)
        y0, y1 = max(0, yi - half), min(image_size, yi + half + 1)
        if mask[y0:y1, x0:x1].mean() >= float(min_mask_coverage):
            selected_features.append(feat)
            selected_centers.append([x, y])
    if not selected_features:
        return np.zeros((0, features.shape[-1]), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.asarray(selected_features, dtype=np.float32), np.asarray(selected_centers, dtype=np.float32)


def _mutual_nn_matches(src_features, ref_features, min_similarity, max_matches):
    sims = src_features @ ref_features.T
    src_to_ref = sims.argmax(axis=1)
    ref_to_src = sims.argmax(axis=0)
    src_ids = np.arange(len(src_features))
    mutual = ref_to_src[src_to_ref] == src_ids
    scores = sims[src_ids, src_to_ref]
    keep = mutual & (scores >= float(min_similarity))
    pairs = np.stack([src_ids[keep], src_to_ref[keep]], axis=1)
    scores = scores[keep]
    order = np.argsort(-scores)[: int(max_matches)]
    return pairs[order], scores[order]


def _similarity_transform(source, target, with_scale=True):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    cov = target_centered.T @ source_centered / max(len(source), 1)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    if with_scale:
        var = np.mean(np.sum(source_centered * source_centered, axis=1))
        scale = float(np.trace(np.diag(singular) @ sign) / max(var, 1e-12))
    else:
        scale = 1.0
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_mean - scale * rotation @ source_mean
    return transform


def _ransac_similarity(source, target, iterations, threshold, seed, with_scale=True):
    rng = np.random.default_rng(int(seed))
    best = None
    count = len(source)
    if count < 4:
        raise RuntimeError(f"Need at least 4 correspondences, got {count}")
    for _ in range(int(iterations)):
        ids = rng.choice(count, size=4, replace=False)
        transform = _similarity_transform(source[ids], target[ids], with_scale=with_scale)
        aligned = (transform @ np.c_[source, np.ones(count)].T).T[:, :3]
        distances = np.linalg.norm(aligned - target, axis=1)
        inliers = distances < float(threshold)
        score = int(inliers.sum())
        error = float(np.median(distances[inliers])) if score else float("inf")
        if best is None or (score, -error) > (best["score"], -best["error"]):
            best = {"transform": transform, "inliers": inliers, "score": score, "error": error}
    if best["score"] >= 4:
        best["transform"] = _similarity_transform(source[best["inliers"]], target[best["inliers"]], with_scale=with_scale)
    return best


def _fixed_scale_from_rotation(source_points, raw_transform, fixed_scale):
    center = source_points.mean(axis=0)
    predicted_center = (raw_transform @ np.r_[center, 1.0])[:3]
    u, _, vt = np.linalg.svd(raw_transform[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    transform = np.eye(4)
    transform[:3, :3] = float(fixed_scale) * rotation
    transform[:3, 3] = predicted_center - float(fixed_scale) * rotation @ center
    return transform


def _write_result(sample_dir, name, partial, moge_target, transform):
    aligned = deepcopy(partial)
    aligned.transform(transform)
    red = deepcopy(moge_target)
    red.paint_uniform_color([1.0, 0.0, 0.0])
    blue = deepcopy(aligned)
    blue.paint_uniform_color([0.0, 0.25, 1.0])
    aligned_path = sample_dir / f"car__394_partial_to_mogev2_2x_{name}.ply"
    compare_path = sample_dir / f"car__394_mogev2_red_{name}_blue_compare.ply"
    transform_path = sample_dir / f"car__394_partial_to_mogev2_2x_{name}_transform.npy"
    o3d.io.write_point_cloud(str(aligned_path), aligned)
    o3d.io.write_point_cloud(str(compare_path), red + blue)
    np.save(transform_path, transform)
    return {
        "aligned": str(aligned_path),
        "compare": str(compare_path),
        "transform": str(transform_path),
        "stats": nearest_neighbor_stats(aligned, moge_target),
        "singular_values": [float(v) for v in np.linalg.svd(transform[:3, :3], compute_uv=False)],
    }


def run(args):
    sample_dir = Path(args.sample_dir)
    partial = o3d.io.read_point_cloud(str(sample_dir / "car__394_hunyuan3d_omni_point_control_processed.ply"))
    partial_points = np.asarray(partial.points, dtype=np.float64)
    point_uv = np.load(sample_dir / "point_uv.npy")
    target_rgb = _read_rgb(sample_dir / "img_sam.png")
    partial_rgb = _read_rgb(sample_dir / args.partial_image)
    moge_full = o3d.io.read_point_cloud(str(sample_dir / "car__394_img_sam_cropped_mogev2_points.ply"))
    moge_target = voxel_downsample_to_count(moge_full, len(partial.points) * 2)

    model, patch_size = _load_dinov2(args.model_name, args.device)
    src_features_grid = _extract_patch_features(model, partial_rgb, args.image_size, args.device)
    ref_features_grid = _extract_patch_features(model, target_rgb, args.image_size, args.device)
    grid = src_features_grid.shape[0]
    centers = _patch_centers(grid, args.image_size)

    source_mask = _source_mask_from_uv(point_uv, args.image_size, radius=args.source_mask_radius)
    target_mask = _foreground_mask(cv2.resize(target_rgb, (args.image_size, args.image_size)), args.white_threshold)
    src_features, src_centers = _sample_feature_grid(src_features_grid, centers, source_mask)
    ref_features, ref_centers = _sample_feature_grid(ref_features_grid, centers, target_mask)
    pairs, scores = _mutual_nn_matches(src_features, ref_features, args.min_similarity, args.max_matches)

    partial_tree, partial_point_indices, partial_points_all = _partial_pixel_lookup(partial_points, point_uv, args.image_size)
    moge_tree, moge_points, crop_box, moge_hw = _moge_points_with_pixels(
        sample_dir / "img_sam.png",
        args.moge_model,
        args.device,
        args.fp16,
        args.white_threshold,
        args.alpha_threshold,
        args.crop_padding,
    )

    src_xy = src_centers[pairs[:, 0]]
    ref_xy = ref_centers[pairs[:, 1]]
    partial_dist, partial_nn = partial_tree.query(src_xy, k=1)
    moge_dist, moge_nn = moge_tree.query(ref_xy, k=1)
    keep = (partial_dist <= args.partial_pixel_radius) & (moge_dist <= args.moge_pixel_radius)
    source_3d = partial_points_all[partial_point_indices[partial_nn[keep]]]
    target_3d = moge_points[moge_nn[keep]]
    kept_scores = scores[keep]
    if len(source_3d) > args.max_correspondences:
        order = np.argsort(-kept_scores)[: args.max_correspondences]
        source_3d = source_3d[order]
        target_3d = target_3d[order]
        kept_scores = kept_scores[order]

    raw = _ransac_similarity(
        source_3d,
        target_3d,
        args.ransac_iterations,
        args.ransac_threshold,
        args.seed,
        with_scale=True,
    )
    raw_transform = raw["transform"]
    baseline_path = sample_dir / "car__394_partial_to_mogev2_2x_trim1p0_scale_clamped_85_transform.npy"
    fixed_scale = float(np.linalg.svd(np.load(baseline_path)[:3, :3], compute_uv=False)[0])
    fixed_transform = _fixed_scale_from_rotation(partial_points, raw_transform, fixed_scale)

    image_tag = Path(args.partial_image).stem.replace(".", "_")
    raw_result = _write_result(sample_dir, f"dinov2_{image_tag}_similarity", partial, moge_target, raw_transform)
    fixed_result = _write_result(sample_dir, f"dinov2_{image_tag}_fixedscale", partial, moge_target, fixed_transform)
    info = {
        "sample": "car__394",
        "method": "dinov2_dense_features_mutual_nn_3d_ransac",
        "model": args.model_name,
        "partial_image": args.partial_image,
        "patch_size": patch_size,
        "feature_grid": [int(grid), int(grid)],
        "source_patches": int(len(src_features)),
        "target_patches": int(len(ref_features)),
        "mutual_matches": int(len(pairs)),
        "pixel_filtered_correspondences": int(len(source_3d)),
        "score_mean": None if len(kept_scores) == 0 else float(np.mean(kept_scores)),
        "score_min": None if len(kept_scores) == 0 else float(np.min(kept_scores)),
        "ransac_inliers": int(raw["score"]),
        "ransac_inlier_median_error": float(raw["error"]),
        "crop_box_xyxy": list(crop_box),
        "moge_inference_hw": list(moge_hw),
        "raw_result": raw_result,
        "fixed_scale": fixed_scale,
        "fixed_result": fixed_result,
    }
    info_path = sample_dir / f"car__394_dinov2_{image_tag}_image_feature_probe_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--moge_model", default=str(DEFAULT_MOGE_MODEL))
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--partial_image", default="depth.png")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--source_mask_radius", type=int, default=8)
    parser.add_argument("--white_threshold", type=int, default=248)
    parser.add_argument("--alpha_threshold", type=int, default=8)
    parser.add_argument("--crop_padding", type=int, default=8)
    parser.add_argument("--min_similarity", type=float, default=0.55)
    parser.add_argument("--max_matches", type=int, default=500)
    parser.add_argument("--max_correspondences", type=int, default=300)
    parser.add_argument("--partial_pixel_radius", type=float, default=14.0)
    parser.add_argument("--moge_pixel_radius", type=float, default=14.0)
    parser.add_argument("--ransac_iterations", type=int, default=3000)
    parser.add_argument("--ransac_threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7351)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
