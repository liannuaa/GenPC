import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "workspace" / "scansalon_zup_side_512" / "car__132"


def pcd_points(path):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    return pcd, np.asarray(pcd.points, dtype=np.float64)


def estimate_similarity_transform(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source and target must both have shape [N, 3], got {source.shape}, {target.shape}")
    if len(source) < 3:
        raise ValueError("At least 3 point correspondences are required.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    source_variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ correction) / max(source_variance, 1e-12))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_mean - scale * rotation @ source_mean
    return transform


def apply_transform(points, transform):
    points = np.asarray(points, dtype=np.float64)
    hom = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ hom.T).T[:, :3]


def ransac_similarity(source, target, iterations, threshold, seed):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) != len(target):
        raise ValueError("source and target correspondence counts differ.")
    if len(source) < 4:
        raise ValueError(f"Need at least 4 correspondences, got {len(source)}")

    rng = np.random.default_rng(int(seed))
    best = None
    sample_size = min(4, len(source))
    for _ in range(int(iterations)):
        ids = rng.choice(len(source), size=sample_size, replace=False)
        try:
            transform = estimate_similarity_transform(source[ids], target[ids])
        except np.linalg.LinAlgError:
            continue
        aligned = apply_transform(source, transform)
        distances = np.linalg.norm(aligned - target, axis=1)
        inliers = distances <= float(threshold)
        score = int(inliers.sum())
        error = float(np.median(distances[inliers])) if score else float("inf")
        if best is None or (score, -error) > (best["score"], -best["error"]):
            best = {
                "transform": transform,
                "inliers": inliers,
                "score": score,
                "error": error,
            }
    if best is None or best["score"] < 4:
        raise RuntimeError("RANSAC failed to find at least 4 inlier correspondences.")
    best["transform"] = estimate_similarity_transform(source[best["inliers"]], target[best["inliers"]])
    aligned = apply_transform(source, best["transform"])
    distances = np.linalg.norm(aligned - target, axis=1)
    best["inliers"] = distances <= float(threshold)
    best["score"] = int(best["inliers"].sum())
    best["error"] = float(np.median(distances[best["inliers"]]))
    best["mean_error"] = float(distances[best["inliers"]].mean())
    best["p95_error"] = float(np.percentile(distances[best["inliers"]], 95))
    return best


def save_pcd(path, points, colors=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def make_compare_cloud(partial_pcd, aligned_moge_pcd):
    partial = deepcopy(partial_pcd)
    moge = deepcopy(aligned_moge_pcd)
    partial.paint_uniform_color([0.55, 0.55, 0.55])
    moge.paint_uniform_color([1.0, 0.0, 0.0])
    return partial + moge


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def run(args):
    import open3d as o3d

    sample_dir = Path(args.sample_dir)
    partial_pcd, partial_points = pcd_points(sample_dir / args.partial_name)
    moge_pcd, moge_points = pcd_points(sample_dir / args.moge_name)
    partial_to_moge = np.load(sample_dir / args.index_name)

    if len(partial_points) != len(partial_to_moge):
        raise ValueError(
            f"partial point count {len(partial_points)} does not match index length {len(partial_to_moge)}"
        )
    valid = partial_to_moge >= 0
    source = moge_points[partial_to_moge[valid]]
    target = partial_points[valid]
    if len(source) > int(args.max_correspondences):
        rng = np.random.default_rng(int(args.seed))
        ids = rng.choice(len(source), size=int(args.max_correspondences), replace=False)
        source = source[ids]
        target = target[ids]

    result = ransac_similarity(
        source,
        target,
        iterations=args.ransac_iterations,
        threshold=args.ransac_threshold,
        seed=args.seed,
    )
    transform = result["transform"]
    aligned_points = apply_transform(moge_points, transform)

    prefix = sample_dir / args.output_prefix
    transform_path = prefix.with_name(prefix.name + "_moge_to_partial_transform.npy")
    aligned_path = prefix.with_name(prefix.name + "_moge_aligned_to_partial.ply")
    compare_path = prefix.with_name(prefix.name + "_partial_gray_moge_red_aligned.ply")
    info_path = prefix.with_name(prefix.name + "_info.json")

    np.save(transform_path, transform)
    colors = np.asarray(moge_pcd.colors, dtype=np.float64) if moge_pcd.has_colors() else None
    save_pcd(aligned_path, aligned_points, colors)
    aligned_pcd = o3d.io.read_point_cloud(str(aligned_path))
    o3d.io.write_point_cloud(str(compare_path), make_compare_cloud(partial_pcd, aligned_pcd))

    info = {
        "partial": str(sample_dir / args.partial_name),
        "moge": str(sample_dir / args.moge_name),
        "index": str(sample_dir / args.index_name),
        "valid_correspondences": int(valid.sum()),
        "used_correspondences": int(len(source)),
        "ransac_inliers": int(result["score"]),
        "ransac_inlier_ratio": float(result["score"] / max(len(source), 1)),
        "ransac_median_error": float(result["error"]),
        "ransac_mean_error": float(result["mean_error"]),
        "ransac_p95_error": float(result["p95_error"]),
        "ransac_threshold": float(args.ransac_threshold),
        "transform": transform.tolist(),
        "outputs": {
            "transform": str(transform_path),
            "aligned_moge": str(aligned_path),
            "compare": str(compare_path),
        },
    }
    save_json(info_path, info)
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--partial_name", default="depth_view_point_cloud.ply")
    parser.add_argument("--moge_name", default="moge_object_only.ply")
    parser.add_argument("--index_name", default="car__132_moge_pixel_bridge_rmbg_partial_to_moge_index.npy")
    parser.add_argument("--output_prefix", default="car__132_moge_to_partial_from_index")
    parser.add_argument("--ransac_iterations", type=int, default=5000)
    parser.add_argument("--ransac_threshold", type=float, default=0.08)
    parser.add_argument("--max_correspondences", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7351)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
