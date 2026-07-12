import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SAMPLE_DIR = (
    PROJECT_ROOT
    / "workspace"
    / "scansalon_hunyuan3d_omni_car__132_geotransformer_only"
    / "car__132"
)


METHOD_NOTES = """Partial-to-partial 2D+3D registration variant.

This is intentionally copied out of the legacy GenPC 2D+3D path instead of
modifying it. The legacy path estimates a 2D pose by rendering a complete GLB;
that fails in the MoGeV2 partial-to-partial case because the GLB proxy and the
MoGeV2 point cloud are different geometries.

This variant treats the MoGeV2 point cloud as another partial observation. It
uses the generated image/depth view convention (XZ image plane, +Y to -Y view)
to map MoGe camera axes into the processed partial frame, searches a small set
of sign/axis candidates with 2D-view bbox initialization, then refines the best
candidate with 3D ICP. The initialization uses one uniform scale only; it never
stretches the MoGeV2 point cloud independently per bbox axis. For car__132 the
stable candidate is source MoGe x,z,y -> target partial x,y,z with signs -,-,-.
"""


def robust_center_extent(points, percentile=1.0):
    points = np.asarray(points, dtype=np.float64)
    q = float(percentile)
    mins = np.percentile(points, q, axis=0)
    maxs = np.percentile(points, 100.0 - q, axis=0)
    return (mins + maxs) * 0.5, np.maximum(maxs - mins, 1e-8)


def image_view_bbox_transform(
    source_points,
    target_points,
    axis_map=(0, 2, 1),
    signs=(-1.0, -1.0, -1.0),
    percentile=1.0,
):
    """Map MoGe camera axes to the partial x/y/z frame using uniform scaling.

    For the current ScanSalon setup the generated image view is XZ from +Y.
    MoGe camera coordinates are image-x, image-y, camera-depth.  The default
    maps source x,z,y to partial x,y,z, then flips signs.  The transform is
    constrained to rotation/reflection + one uniform scale + translation.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    axis_map = tuple(int(v) for v in axis_map)
    signs = np.asarray(signs, dtype=np.float64)
    if sorted(axis_map) != [0, 1, 2]:
        raise ValueError(f"axis_map must be a permutation of 0,1,2, got {axis_map}")
    if signs.shape != (3,):
        raise ValueError(f"signs must have shape (3,), got {signs.shape}")

    axis_matrix = np.zeros((3, 3), dtype=np.float64)
    for target_axis, source_axis in enumerate(axis_map):
        axis_matrix[target_axis, source_axis] = signs[target_axis]

    mapped = source_points @ axis_matrix.T
    source_center, source_extent = robust_center_extent(mapped, percentile=percentile)
    target_center, target_extent = robust_center_extent(target_points, percentile=percentile)
    source_scale = float(source_extent.max())
    target_scale = float(target_extent.max())
    scale = target_scale / max(source_scale, 1e-8)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * axis_matrix
    transform[:3, 3] = target_center - scale * source_center
    return transform


def voxel_downsample_to_count(pcd, target_count, iterations=24):
    points = np.asarray(pcd.points)
    if len(points) <= target_count:
        return deepcopy(pcd)

    lo = 1e-4
    hi = float(np.max(np.ptp(points, axis=0)) * 0.2)
    best = None
    for _ in range(int(iterations)):
        mid = (lo + hi) * 0.5
        down = pcd.voxel_down_sample(mid)
        if best is None or abs(len(down.points) - target_count) < abs(
            len(best.points) - target_count
        ):
            best = down
        if len(down.points) > target_count:
            lo = mid
        else:
            hi = mid
    return best


def transform_cloud(pcd, transform):
    out = deepcopy(pcd)
    out.transform(transform)
    return out


def nearest_neighbor_stats(source, target):
    d1 = np.asarray(source.compute_point_cloud_distance(target))
    d2 = np.asarray(target.compute_point_cloud_distance(source))
    return {
        "source_to_target_mean": float(np.mean(d1)),
        "source_to_target_p95": float(np.percentile(d1, 95)),
        "target_to_source_mean": float(np.mean(d2)),
        "target_to_source_p95": float(np.percentile(d2, 95)),
    }


def score_alignment(source, target):
    stats = nearest_neighbor_stats(source, target)
    return (
        stats["source_to_target_p95"] + 0.25 * stats["target_to_source_p95"],
        stats,
    )


def run_icp(source, target, threshold, max_iteration=60):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(max_iteration)
    )
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(threshold),
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )


def find_image_view_init(source_search, target_search, axis_maps, thresholds):
    source_points = np.asarray(source_search.points)
    target_points = np.asarray(target_search.points)
    best = None
    candidates = []
    for axis_map in axis_maps:
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            init_transform = image_view_bbox_transform(
                source_points,
                target_points,
                axis_map=axis_map,
                signs=signs,
                percentile=1.0,
            )
            initialized = transform_cloud(source_search, init_transform)
            for threshold in thresholds:
                result = run_icp(initialized, target_search, threshold)
                final_transform = result.transformation @ init_transform
                aligned = transform_cloud(source_search, final_transform)
                score, stats = score_alignment(target_search, aligned)
                candidate = {
                    "axis_map": list(axis_map),
                    "signs": [float(v) for v in signs],
                    "threshold": float(threshold),
                    "score": float(score),
                    "fitness": float(result.fitness),
                    "rmse": float(result.inlier_rmse),
                    "stats": stats,
                    "transform": final_transform,
                }
                candidates.append(candidate)
                if best is None or candidate["score"] < best["score"]:
                    best = candidate
    return best, candidates


def refine_on_full_subset(source_subset, target, init_transform, thresholds):
    initialized = transform_cloud(source_subset, init_transform)
    best = None
    for threshold in thresholds:
        result = run_icp(initialized, target, threshold)
        final_transform = result.transformation @ init_transform
        aligned = transform_cloud(source_subset, final_transform)
        score, stats = score_alignment(target, aligned)
        candidate = {
            "threshold": float(threshold),
            "score": float(score),
            "fitness": float(result.fitness),
            "rmse": float(result.inlier_rmse),
            "stats": stats,
            "transform": final_transform,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    return best


def make_red_gray_visual(partial, moge):
    partial_vis = deepcopy(partial)
    partial_vis.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.55, 0.55, 0.55]], dtype=np.float64), (len(partial.points), 1))
    )
    moge_vis = deepcopy(moge)
    moge_vis.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(moge.points), 1))
    )
    return moge_vis + partial_vis


def output_prefix(sample_dir, flag):
    return sample_dir / f"{flag}_img_sam_cropped_mogev2_partial2partial_2d3d"


def run_one(sample_dir, flag, args):
    sample_dir = Path(sample_dir)
    partial_path = sample_dir / f"{flag}_hunyuan3d_omni_point_control_processed.ply"
    preferred_moge_path = sample_dir / f"{flag}_img_sam_cropped_mogev2_points.ply"
    fallback_moge_path = sample_dir / f"{flag}_mogev2_points.ply"
    moge_path = preferred_moge_path if preferred_moge_path.exists() else fallback_moge_path

    partial = o3d.io.read_point_cloud(str(partial_path))
    moge_full = o3d.io.read_point_cloud(str(moge_path))
    if len(partial.points) == 0 or len(moge_full.points) == 0:
        raise RuntimeError("empty partial or MoGeV2 point cloud")

    moge_subset = voxel_downsample_to_count(moge_full, len(partial.points))
    search_partial = partial.voxel_down_sample(float(args.search_partial_voxel))
    search_moge = voxel_downsample_to_count(moge_full, int(args.search_moge_points))
    print(
        f"[inputs] partial={len(partial.points)} moge_full={len(moge_full.points)} "
        f"moge_subset={len(moge_subset.points)} search_partial={len(search_partial.points)} "
        f"search_moge={len(search_moge.points)}"
    )

    search_thresholds = [float(v) for v in args.search_thresholds.split(",")]
    refine_thresholds = [float(v) for v in args.refine_thresholds.split(",")]
    axis_maps = [(0, 2, 1), (1, 2, 0)]
    init_best, init_candidates = find_image_view_init(
        search_moge, search_partial, axis_maps, search_thresholds
    )
    print(
        "[2d-init] "
        f"axis_map={init_best['axis_map']} signs={init_best['signs']} "
        f"threshold={init_best['threshold']:.4f} score={init_best['score']:.6f}"
    )

    refined = refine_on_full_subset(
        moge_subset, partial, init_best["transform"], refine_thresholds
    )
    final_transform = refined["transform"]
    aligned_full = transform_cloud(moge_full, final_transform)
    aligned_subset = transform_cloud(moge_subset, final_transform)
    full_score, full_stats = score_alignment(partial, aligned_full)
    subset_score, subset_stats = score_alignment(partial, aligned_subset)
    print(
        "[3d-refine] "
        f"threshold={refined['threshold']:.4f} score={refined['score']:.6f} "
        f"partial_to_full_p95={full_stats['source_to_target_p95']:.6f}"
    )

    prefix = output_prefix(sample_dir, flag)
    out_full = prefix.with_name(prefix.name + "_to_processed_partial.ply")
    out_subset = prefix.with_name(prefix.name + "_subset_to_processed_partial.ply")
    out_vis = prefix.with_name(prefix.name + "_red_vs_partial_gray.ply")
    out_info = prefix.with_name(prefix.name + "_info.json")
    out_transform = prefix.with_name(prefix.name + "_transform.npy")

    o3d.io.write_point_cloud(str(out_full), aligned_full)
    o3d.io.write_point_cloud(str(out_subset), aligned_subset)
    o3d.io.write_point_cloud(str(out_vis), make_red_gray_visual(partial, aligned_full))
    np.save(str(out_transform), final_transform)

    info = {
        "method": "partial_to_partial_image_view_2d_bbox_init_plus_3d_icp",
        "note": (
            "Copied partial-to-partial variant. It uses the generated image view "
            "to map MoGe camera x/y/depth axes into partial x/z/y axes, then "
            "refines with 3D ICP. The legacy GenPC 2D+3D method is untouched."
        ),
        "inputs": {
            "partial": str(partial_path),
            "moge": str(moge_path),
            "partial_points": len(partial.points),
            "moge_full_points": len(moge_full.points),
            "moge_subset_points": len(moge_subset.points),
        },
        "image_view_init": {
            k: v for k, v in init_best.items() if k != "transform"
        },
        "top_image_view_candidates": [
            {k: v for k, v in candidate.items() if k != "transform"}
            for candidate in sorted(init_candidates, key=lambda item: item["score"])[:8]
        ],
        "refine": {k: v for k, v in refined.items() if k != "transform"},
        "full_score": float(full_score),
        "subset_score": float(subset_score),
        "full_stats": full_stats,
        "subset_stats": subset_stats,
        "outputs": {
            "full": str(out_full),
            "subset": str(out_subset),
            "vis": str(out_vis),
            "transform": str(out_transform),
        },
    }
    out_info.write_text(json.dumps(info, indent=2))
    print("[done]")
    print(json.dumps(info, indent=2))
    return info


def run(args):
    if args.write_method_notes:
        notes_path = Path(args.write_method_notes)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(METHOD_NOTES)

    sample_dirs = []
    if args.sample_dirs:
        sample_dirs.extend(Path(path) for path in args.sample_dirs)
    else:
        sample_dirs.append(Path(args.sample_dir))

    infos = []
    for sample_dir in sample_dirs:
        flag = args.flag or sample_dir.name
        if args.flag and len(sample_dirs) > 1:
            raise ValueError("--flag can only be used with one sample directory")
        print(f"[sample] {flag} @ {sample_dir}")
        infos.append(run_one(sample_dir, flag, args))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(infos, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--sample_dirs", nargs="*")
    parser.add_argument("--flag", default=None)
    parser.add_argument("--search_partial_voxel", type=float, default=0.04)
    parser.add_argument("--search_moge_points", type=int, default=2000)
    parser.add_argument("--search_thresholds", default="0.06,0.1,0.15")
    parser.add_argument("--refine_thresholds", default="0.04,0.06,0.08,0.1,0.15")
    parser.add_argument("--write_method_notes")
    parser.add_argument("--summary_json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
