import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mogev2_registration_probe import (
    build_foreground_mask,
    crop_to_foreground,
    image_foreground_mask,
    image_to_rgb_for_moge,
)
from scripts.run_partial2partial_registration_probe import (
    nearest_neighbor_stats,
    voxel_downsample_to_count,
)


DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "workspace" / "scansalon_car_partial2partial_inputs" / "car__394"
DEFAULT_MOGE_MODEL = PROJECT_ROOT / "models" / "moge-2-vitl"


def _load_rgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def _paint_points(points, uv_norm, size):
    image = np.full((size, size), 255, dtype=np.uint8)
    uv = np.asarray(uv_norm, dtype=np.float64)
    xy = np.rint(uv * (size - 1)).astype(np.int32)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
    xy = xy[valid]
    for x, y in xy:
        cv2.circle(image, (int(x), int(y)), 2, 0, -1, lineType=cv2.LINE_AA)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def _partial_pixel_lookup(points, uv_norm, size):
    xy = np.rint(np.asarray(uv_norm, dtype=np.float64) * (size - 1)).astype(np.int32)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
    pixel_xy = xy[valid]
    point_indices = np.where(valid)[0]
    tree = cKDTree(pixel_xy.astype(np.float64))
    return tree, point_indices, np.asarray(points, dtype=np.float64)


def _moge_points_with_pixels(image_path, pretrained, device, fp16, white_threshold, alpha_threshold, crop_padding):
    from moge.model.v2 import MoGeModel

    image_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        raise FileNotFoundError(image_path)
    if image_raw.ndim == 2:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
    elif image_raw.shape[-1] == 4:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGRA2RGBA)
    else:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    foreground = image_foreground_mask(image_raw, white_threshold=white_threshold, alpha_threshold=alpha_threshold)
    cropped_raw, cropped_foreground, crop_box = crop_to_foreground(
        image_raw, foreground, padding=crop_padding
    )
    image_rgb = image_to_rgb_for_moge(cropped_raw)
    image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    pretrained = Path(pretrained)
    if pretrained.is_dir():
        pretrained = pretrained / "model.pt"
    model = MoGeModel.from_pretrained(str(pretrained)).to(device).eval()
    with torch.no_grad():
        if fp16 and str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.infer(image_tensor)
        else:
            output = model.infer(image_tensor)

    points = output["points"].detach().float().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()
    valid = build_foreground_mask(
        points,
        mask,
        image_rgb,
        white_threshold=white_threshold,
        image_mask=cropped_foreground,
    )
    ys, xs = np.where(valid)
    crop_x0, crop_y0, _, _ = crop_box
    original_xy = np.stack([xs + crop_x0, ys + crop_y0], axis=1).astype(np.float64)
    valid_points = points[valid].astype(np.float64)
    tree = cKDTree(original_xy)
    return tree, valid_points, crop_box, image_rgb.shape[:2]


def _loftr_matches(partial_gray, target_rgb, device, max_dim=512):
    from kornia.feature import LoFTR

    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    partial_gray = cv2.resize(partial_gray, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
    if target_gray.shape != (max_dim, max_dim):
        target_gray = cv2.resize(target_gray, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)

    image0 = torch.from_numpy(partial_gray / 255.0)[None, None].float().to(device)
    image1 = torch.from_numpy(target_gray / 255.0)[None, None].float().to(device)
    matcher = LoFTR(pretrained="outdoor").to(device).eval()
    with torch.no_grad():
        correspondences = matcher({"image0": image0, "image1": image1})
    keypoints0 = correspondences["keypoints0"].detach().cpu().numpy()
    keypoints1 = correspondences["keypoints1"].detach().cpu().numpy()
    confidence = correspondences["confidence"].detach().cpu().numpy()
    return keypoints0, keypoints1, confidence


def _rigid_transform(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    cov = target_centered.T @ source_centered / max(len(source), 1)
    u, _, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _ransac_3d(source, target, iterations, threshold, seed):
    rng = np.random.default_rng(seed)
    best = None
    count = len(source)
    if count < 4:
        raise RuntimeError(f"Need at least 4 correspondences, got {count}")
    for _ in range(iterations):
        idx = rng.choice(count, size=4, replace=False)
        transform = _rigid_transform(source[idx], target[idx])
        aligned = (transform @ np.c_[source, np.ones(count)].T).T[:, :3]
        distances = np.linalg.norm(aligned - target, axis=1)
        inliers = distances < threshold
        score = int(inliers.sum())
        error = float(np.median(distances[inliers])) if score else float("inf")
        if best is None or (score, -error) > (best["score"], -best["error"]):
            best = {"transform": transform, "inliers": inliers, "score": score, "error": error}
    if best["score"] >= 4:
        best["transform"] = _rigid_transform(source[best["inliers"]], target[best["inliers"]])
    return best


def _write_compare(path, moge, aligned_partial):
    red = deepcopy(moge)
    red.paint_uniform_color([1.0, 0.0, 0.0])
    blue = deepcopy(aligned_partial)
    blue.paint_uniform_color([0.0, 0.25, 1.0])
    o3d.io.write_point_cloud(str(path), red + blue)


def run(args):
    sample_dir = Path(args.sample_dir)
    partial = o3d.io.read_point_cloud(str(sample_dir / "car__394_hunyuan3d_omni_point_control_processed.ply"))
    partial_points = np.asarray(partial.points, dtype=np.float64)
    point_uv = np.load(sample_dir / "point_uv.npy")
    target_rgb = _load_rgb(sample_dir / "img_sam.png")
    moge_full = o3d.io.read_point_cloud(str(sample_dir / "car__394_img_sam_cropped_mogev2_points.ply"))
    moge_target = voxel_downsample_to_count(moge_full, len(partial.points) * 2)

    partial_image = _paint_points(partial_points, point_uv, args.image_size)
    cv2.imwrite(str(sample_dir / "car__394_partial_projection_for_loftr.png"), partial_image)

    partial_tree, partial_point_indices, partial_points_all = _partial_pixel_lookup(
        partial_points, point_uv, args.image_size
    )
    moge_tree, moge_points, crop_box, moge_hw = _moge_points_with_pixels(
        sample_dir / "img_sam.png",
        args.moge_model,
        args.device,
        args.fp16,
        args.white_threshold,
        args.alpha_threshold,
        args.crop_padding,
    )

    kp0, kp1, conf = _loftr_matches(partial_image, target_rgb, args.device, max_dim=args.image_size)
    if len(conf) == 0:
        raise RuntimeError("LoFTR produced no matches")
    order = np.argsort(-conf)
    order = order[: min(args.max_matches, len(order))]
    kp0 = kp0[order]
    kp1 = kp1[order]
    conf = conf[order]

    partial_dist, partial_nn = partial_tree.query(kp0, k=1)
    moge_dist, moge_nn = moge_tree.query(kp1, k=1)
    keep = (partial_dist <= args.partial_pixel_radius) & (moge_dist <= args.moge_pixel_radius)
    src = partial_points_all[partial_point_indices[partial_nn[keep]]]
    dst = moge_points[moge_nn[keep]]
    kept_conf = conf[keep]
    if len(src) > args.max_correspondences:
        order = np.argsort(-kept_conf)[: args.max_correspondences]
        src = src[order]
        dst = dst[order]
        kept_conf = kept_conf[order]

    ransac = _ransac_3d(src, dst, args.ransac_iterations, args.ransac_threshold, args.seed)
    transform = ransac["transform"]
    aligned = deepcopy(partial)
    aligned.transform(transform)

    aligned_path = sample_dir / "car__394_partial_to_mogev2_2x_image_feature_loftr.ply"
    compare_path = sample_dir / "car__394_mogev2_red_image_feature_loftr_blue_compare.ply"
    info_path = sample_dir / "car__394_image_feature_loftr_probe_info.json"
    transform_path = sample_dir / "car__394_partial_to_mogev2_2x_image_feature_loftr_transform.npy"
    o3d.io.write_point_cloud(str(aligned_path), aligned)
    _write_compare(compare_path, moge_target, aligned)
    np.save(transform_path, transform)

    info = {
        "sample": "car__394",
        "method": "partial_projection_to_img_sam_loftr_moge_backproject_ransac",
        "loftr_matches": int(len(conf)),
        "pixel_filtered_correspondences": int(len(src)),
        "ransac_inliers": int(ransac["score"]),
        "ransac_inlier_median_error": float(ransac["error"]),
        "stats": nearest_neighbor_stats(aligned, moge_target),
        "crop_box_xyxy": list(crop_box),
        "moge_inference_hw": list(moge_hw),
        "outputs": {
            "aligned": str(aligned_path),
            "compare": str(compare_path),
            "transform": str(transform_path),
            "partial_projection": str(sample_dir / "car__394_partial_projection_for_loftr.png"),
        },
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--moge_model", default=str(DEFAULT_MOGE_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--white_threshold", type=int, default=248)
    parser.add_argument("--alpha_threshold", type=int, default=8)
    parser.add_argument("--crop_padding", type=int, default=8)
    parser.add_argument("--max_matches", type=int, default=2000)
    parser.add_argument("--max_correspondences", type=int, default=500)
    parser.add_argument("--partial_pixel_radius", type=float, default=6.0)
    parser.add_argument("--moge_pixel_radius", type=float, default=6.0)
    parser.add_argument("--ransac_iterations", type=int, default=2000)
    parser.add_argument("--ransac_threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7351)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
