import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_moge_pixel_index_bridge import (
    filter_moge_points_by_object_mask,
    load_alpha_mask,
    run_moge_with_pixels,
)
from scripts.run_moge_to_partial_from_index import (
    apply_transform,
    ransac_similarity,
)


DEFAULT_MOGE_MODEL = PROJECT_ROOT / "models" / "moge-2-vitl"


def load_pcd_points(path):
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd, points


def project_depthpro_points_to_pixel_xy(points, intrinsic, image_size):
    points = np.asarray(points, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    cam = (intrinsic @ points.T).T
    z = cam[:, 2]
    uv = cam[:, :2] / (z[:, None] + 1e-8)
    valid = np.isfinite(uv).all(axis=1)
    valid &= z > 1e-8
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < image_size)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < image_size)
    return uv.astype(np.float64), valid


def build_depthpro_to_moge_correspondences(
    depthpro_points,
    depthpro_pixel_xy,
    moge_points,
    moge_pixel_xy,
    max_pixel_distance,
    max_correspondences,
    seed,
):
    depthpro_pixel_xy = np.asarray(depthpro_pixel_xy, dtype=np.float64)
    moge_pixel_xy = np.asarray(moge_pixel_xy, dtype=np.float64)
    distances, nearest = cKDTree(moge_pixel_xy).query(depthpro_pixel_xy, k=1)
    keep = distances <= float(max_pixel_distance)
    source = np.asarray(depthpro_points, dtype=np.float64)[keep]
    target = np.asarray(moge_points, dtype=np.float64)[nearest[keep]]
    pixel_distances = distances[keep]

    if len(source) > int(max_correspondences):
        rng = np.random.default_rng(int(seed))
        ids = rng.choice(len(source), size=int(max_correspondences), replace=False)
        source = source[ids]
        target = target[ids]
        pixel_distances = pixel_distances[ids]
    return source, target, pixel_distances


def paint_compare(partial_path, complete_aligned):
    partial = o3d.io.read_point_cloud(str(partial_path))
    partial_vis = deepcopy(partial)
    partial_vis.paint_uniform_color([0.55, 0.55, 0.55])
    complete_vis = deepcopy(complete_aligned)
    complete_vis.paint_uniform_color([0.05, 0.25, 1.0])
    return partial_vis + complete_vis


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))


def run(args):
    sample_dir = Path(args.sample_dir)
    flag = args.flag
    complete_name = args.complete_name or f"{flag}_hunyuan2.1.ply"
    partial_path = Path(args.partial_path) if args.partial_path else PROJECT_ROOT / "data" / f"{flag}.ply"
    gt_path = Path(args.gt_path) if args.gt_path else PROJECT_ROOT / "data" / "GT" / f"{flag}.ply"
    freereg_info_name = args.freereg_info_name or f"{flag}_freereg_original_depthpro_fixeduv_sim3_info.json"
    depthpro_points_name = (
        args.depthpro_points_name
        or f"{flag}_freereg_original_depthpro_fixeduv_sim3_object_depthpro_points.ply"
    )
    object_mask_name = args.object_mask_name or f"{flag}_moge_to_raw_partial_object_mask.png"
    moge_to_partial_name = (
        args.moge_to_partial_name
        or f"{flag}_moge_to_raw_partial_moge_to_raw_partial_transform.npy"
    )
    output_prefix_name = args.output_prefix or f"{flag}_complete_to_partial_depthpro_moge"

    freereg_info_path = sample_dir / freereg_info_name
    freereg_info = json.loads(freereg_info_path.read_text())
    complete_to_depthpro = np.asarray(freereg_info["complete_to_image"], dtype=np.float64)
    intrinsic = np.asarray(freereg_info["intrinsic"], dtype=np.float64)

    depthpro_pcd, depthpro_points = load_pcd_points(sample_dir / depthpro_points_name)
    depthpro_pixel_xy, valid_depthpro = project_depthpro_points_to_pixel_xy(
        depthpro_points,
        intrinsic,
        image_size=args.image_size,
    )
    depthpro_points = depthpro_points[valid_depthpro]
    depthpro_pixel_xy = depthpro_pixel_xy[valid_depthpro]

    full_moge_points, full_moge_colors, full_moge_pixels, moge_info = run_moge_with_pixels(
        image_path=sample_dir / args.image_name,
        pretrained=args.moge_model,
        device=args.device,
        fp16=bool(args.fp16),
    )
    object_mask = load_alpha_mask(sample_dir / object_mask_name)
    object_moge = filter_moge_points_by_object_mask(
        points=full_moge_points,
        colors=full_moge_colors,
        pixel_xy=full_moge_pixels,
        object_mask=object_mask,
        alpha_threshold=args.object_alpha_threshold,
        erode_pixels=0,
    )

    source, target, pixel_distances = build_depthpro_to_moge_correspondences(
        depthpro_points=depthpro_points,
        depthpro_pixel_xy=depthpro_pixel_xy,
        moge_points=object_moge.points,
        moge_pixel_xy=object_moge.pixel_xy,
        max_pixel_distance=args.max_pixel_distance,
        max_correspondences=args.max_correspondences,
        seed=args.seed,
    )
    if len(source) < 4:
        raise RuntimeError(f"Need at least 4 DepthPro-MoGe correspondences, got {len(source)}")

    depthpro_to_moge_result = ransac_similarity(
        source,
        target,
        iterations=args.ransac_iterations,
        threshold=args.ransac_threshold,
        seed=args.seed,
    )
    depthpro_to_moge = depthpro_to_moge_result["transform"]
    moge_to_partial = np.load(sample_dir / moge_to_partial_name)
    complete_to_partial = moge_to_partial @ depthpro_to_moge @ complete_to_depthpro

    complete_pcd, complete_points = load_pcd_points(sample_dir / complete_name)
    complete_aligned_points = apply_transform(complete_points, complete_to_partial)
    complete_aligned = deepcopy(complete_pcd)
    complete_aligned.points = o3d.utility.Vector3dVector(complete_aligned_points)

    output_prefix = sample_dir / output_prefix_name
    aligned_path = output_prefix.with_name(output_prefix.name + "_complete_aligned_to_raw_partial.ply")
    compare_path = output_prefix.with_name(output_prefix.name + "_raw_partial_gray_complete_blue_aligned.ply")
    transform_path = output_prefix.with_name(output_prefix.name + "_complete_to_raw_partial_transform.npy")
    info_path = output_prefix.with_name(output_prefix.name + "_info.json")

    o3d.io.write_point_cloud(str(aligned_path), complete_aligned)
    o3d.io.write_point_cloud(str(compare_path), paint_compare(partial_path, complete_aligned))
    np.save(transform_path, complete_to_partial)

    info = {
        "method": "complete_to_partial_via_fixeduv_depthpro_to_moge_pixel_bridge",
        "flag": str(flag),
        "sample_dir": str(sample_dir),
        "complete": str(sample_dir / complete_name),
        "partial": str(partial_path),
        "gt": str(gt_path),
        "freereg_info": str(freereg_info_path),
        "depthpro_points": str(sample_dir / depthpro_points_name),
        "moge_model": str(args.moge_model),
        "moge_to_partial_transform": str(sample_dir / moge_to_partial_name),
        "depthpro_valid_points": int(len(depthpro_points)),
        "moge_object_points": int(len(object_moge.points)),
        "depthpro_moge_correspondences": int(len(source)),
        "depthpro_moge_pixel_distance_mean": float(pixel_distances.mean()),
        "depthpro_moge_pixel_distance_p95": float(np.percentile(pixel_distances, 95)),
        "depthpro_to_moge": depthpro_to_moge.tolist(),
        "depthpro_to_moge_ransac": {
            "inliers": int(depthpro_to_moge_result["score"]),
            "inlier_ratio": float(depthpro_to_moge_result["score"] / max(len(source), 1)),
            "median_error": float(depthpro_to_moge_result["error"]),
            "mean_error": float(depthpro_to_moge_result["mean_error"]),
            "p95_error": float(depthpro_to_moge_result["p95_error"]),
            "threshold": float(args.ransac_threshold),
        },
        "complete_to_depthpro": complete_to_depthpro.tolist(),
        "moge_to_partial": moge_to_partial.tolist(),
        "complete_to_partial": complete_to_partial.tolist(),
        "moge": moge_info,
        "outputs": {
            "aligned_complete": str(aligned_path),
            "compare": str(compare_path),
            "transform": str(transform_path),
        },
    }
    save_json(info_path, info)
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", required=True)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--image_name", default="img.png")
    parser.add_argument("--complete_name", default=None)
    parser.add_argument("--partial_path", default=None)
    parser.add_argument("--gt_path", default=None)
    parser.add_argument("--freereg_info_name", default=None)
    parser.add_argument("--depthpro_points_name", default=None)
    parser.add_argument("--object_mask_name", default=None)
    parser.add_argument("--moge_to_partial_name", default=None)
    parser.add_argument("--moge_model", default=str(DEFAULT_MOGE_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--object_alpha_threshold", type=int, default=128)
    parser.add_argument("--max_pixel_distance", type=float, default=1.5)
    parser.add_argument("--max_correspondences", type=int, default=50000)
    parser.add_argument("--ransac_iterations", type=int, default=5000)
    parser.add_argument("--ransac_threshold", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1184)
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
