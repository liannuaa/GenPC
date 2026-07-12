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

from optim_registration.predator_registration import estimate_predator_transform
from scripts.run_partial2partial_registration_probe import (
    nearest_neighbor_stats,
    voxel_downsample_to_count,
)


DEFAULT_SAMPLE_DIR = (
    PROJECT_ROOT / "workspace" / "scansalon_car_partial2partial_inputs" / "car__394"
)


def _rotation_angle_deg(rotation):
    value = (np.trace(rotation) - 1.0) * 0.5
    value = np.clip(value, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _rigid_transform(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    cov = target_centered.T @ source_centered / max(len(source), 1)
    u, _, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _fixed_scale_transform_from_rotation(source_points, raw_transform, fixed_scale):
    source_points = np.asarray(source_points, dtype=np.float64)
    source_center = source_points.mean(axis=0)
    predicted_center = (raw_transform @ np.r_[source_center, 1.0])[:3]
    u, _, vt = np.linalg.svd(raw_transform[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = float(fixed_scale) * rotation
    transform[:3, 3] = predicted_center - float(fixed_scale) * rotation @ source_center
    return transform


def _fixed_scale_trimmed_rigid_refine(
    source_pcd,
    target_pcd,
    init_transform,
    trims=(1.0, 0.85, 0.7, 0.55),
    iterations=25,
    max_step_angle=12.0,
):
    source_points = np.asarray(source_pcd.points, dtype=np.float64)
    target_points = np.asarray(target_pcd.points, dtype=np.float64)
    tree = cKDTree(target_points)
    init_scale = float(np.linalg.svd(init_transform[:3, :3], compute_uv=False)[0])
    best = None
    results = []

    for trim in trims:
        transform = np.array(init_transform, dtype=np.float64)
        logs = []
        for iteration in range(int(iterations)):
            current = (transform @ np.c_[source_points, np.ones(len(source_points))].T).T[:, :3]
            distances, indices = tree.query(current, k=1, workers=-1)
            keep_count = max(50, int(len(distances) * float(trim)))
            keep = np.argpartition(distances, keep_count - 1)[:keep_count]
            update = _rigid_transform(current[keep], target_points[indices[keep]])
            step_angle = _rotation_angle_deg(update[:3, :3])
            if step_angle > float(max_step_angle):
                damped = np.eye(4, dtype=np.float64)
                damped[:3, 3] = update[:3, 3]
                update = damped
                step_angle = 0.0
            transform = update @ transform
            if iteration in {0, 1, 2, 4, 9, 14, int(iterations) - 1}:
                aligned = deepcopy(source_pcd)
                aligned.transform(transform)
                stats = nearest_neighbor_stats(aligned, target_pcd)
                logs.append(
                    {
                        "iteration": int(iteration + 1),
                        "step_angle_deg": float(step_angle),
                        **stats,
                    }
                )

        aligned = deepcopy(source_pcd)
        aligned.transform(transform)
        stats = nearest_neighbor_stats(aligned, target_pcd)
        singular_values = np.linalg.svd(transform[:3, :3], compute_uv=False)
        score = stats["source_to_target_p95"] + 0.15 * stats["target_to_source_p95"]
        record = {
            "trim": float(trim),
            "score": float(score),
            "stats": stats,
            "singular_values": [float(v) for v in singular_values],
            "scale_change_ratio": float(singular_values[0] / max(init_scale, 1e-12)),
            "transform": transform.tolist(),
            "logs": logs,
        }
        results.append(record)
        if best is None or record["score"] < best["score"]:
            best = record
    return best, results


def _write_red_gray(path, source_pcd, target_pcd):
    red = deepcopy(source_pcd)
    red.paint_uniform_color([1.0, 0.0, 0.0])
    gray = deepcopy(target_pcd)
    gray.paint_uniform_color([0.55, 0.55, 0.55])
    o3d.io.write_point_cloud(str(path), red + gray)


def _save_result(sample_dir, name, source_pcd, target_pcd, transform):
    aligned = deepcopy(source_pcd)
    aligned.transform(transform)
    pcd_path = sample_dir / f"car__394_partial_to_mogev2_2x_gray_{name}.ply"
    compare_path = sample_dir / f"car__394_partial_red_to_mogev2_2x_gray_{name}.ply"
    transform_path = sample_dir / f"car__394_partial_to_mogev2_2x_{name}_transform.npy"
    o3d.io.write_point_cloud(str(pcd_path), aligned)
    _write_red_gray(compare_path, aligned, target_pcd)
    np.save(transform_path, transform)
    return {
        "aligned": str(pcd_path),
        "compare": str(compare_path),
        "transform": str(transform_path),
        "stats": nearest_neighbor_stats(aligned, target_pcd),
        "singular_values": [
            float(v) for v in np.linalg.svd(transform[:3, :3], compute_uv=False)
        ],
    }


def run(args):
    sample_dir = Path(args.sample_dir)
    partial_path = sample_dir / "car__394_hunyuan3d_omni_point_control_processed.ply"
    moge_path = sample_dir / "car__394_img_sam_cropped_mogev2_points.ply"
    baseline_transform_path = sample_dir / "car__394_partial_to_mogev2_2x_trim1p0_scale_clamped_85_transform.npy"

    partial = o3d.io.read_point_cloud(str(partial_path))
    moge_full = o3d.io.read_point_cloud(str(moge_path))
    if len(partial.points) == 0 or len(moge_full.points) == 0:
        raise RuntimeError("empty partial or MoGe point cloud")

    target = voxel_downsample_to_count(moge_full, len(partial.points) * int(args.target_ratio))
    target_path = sample_dir / "car__394_mogev2_2x_predator_target.ply"
    o3d.io.write_point_cloud(str(target_path), target)

    raw_transform, predator_info = estimate_predator_transform(
        target,
        partial,
        device=args.device,
        n_points=args.n_points,
        distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n,
        seed=args.seed,
    )
    raw_result = _save_result(sample_dir, "predator_raw", partial, target, raw_transform)

    raw_scale = float(np.linalg.svd(raw_transform[:3, :3], compute_uv=False)[0])
    if baseline_transform_path.exists():
        baseline_transform = np.load(baseline_transform_path)
        fixed_scale = float(np.linalg.svd(baseline_transform[:3, :3], compute_uv=False)[0])
    else:
        fixed_scale = raw_scale
    fixed_init = _fixed_scale_transform_from_rotation(
        np.asarray(partial.points), raw_transform, fixed_scale
    )
    refined_best, refined_results = _fixed_scale_trimmed_rigid_refine(
        partial,
        target,
        fixed_init,
        trims=tuple(float(v) for v in args.refine_trims.split(",")),
        iterations=args.refine_iterations,
    )
    fixed_result = _save_result(
        sample_dir,
        "predator_fixedscale_refine",
        partial,
        target,
        np.asarray(refined_best["transform"], dtype=np.float64),
    )

    info = {
        "sample": "car__394",
        "partial": str(partial_path),
        "moge_full": str(moge_path),
        "target": str(target_path),
        "target_points": len(target.points),
        "partial_points": len(partial.points),
        "predator": predator_info,
        "raw_result": raw_result,
        "raw_scale": raw_scale,
        "fixed_scale": fixed_scale,
        "fixed_init_singular_values": [
            float(v) for v in np.linalg.svd(fixed_init[:3, :3], compute_uv=False)
        ],
        "fixed_refine_best": refined_best,
        "fixed_refine_results": refined_results,
        "fixed_result": fixed_result,
    }
    info_path = sample_dir / "car__394_partial_to_mogev2_2x_predator_probe_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({
        "raw_stats": raw_result["stats"],
        "raw_singular_values": raw_result["singular_values"],
        "fixed_stats": fixed_result["stats"],
        "fixed_singular_values": fixed_result["singular_values"],
        "info": str(info_path),
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_ratio", type=int, default=2)
    parser.add_argument("--n_points", type=int, default=1000)
    parser.add_argument("--distance_threshold", type=float, default=0.05)
    parser.add_argument("--ransac_n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--refine_trims", default="1.0,0.85,0.7,0.55")
    parser.add_argument("--refine_iterations", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
