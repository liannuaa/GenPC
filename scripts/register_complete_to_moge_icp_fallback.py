import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_mogev2_registration_probe import nearest_neighbor_stats, run_similarity_icp


DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "workspace" / "scansalon_zup_side_512" / "car__132"


def paint_or_keep(pcd, color):
    out = deepcopy(pcd)
    if not out.has_colors():
        out.paint_uniform_color(color)
    return out


def build_registration_info(
    complete_path,
    moge_path,
    transform,
    icp_info,
    registered_path,
    merged_path,
    complete_to_moge_stats,
    moge_to_complete_stats,
):
    return {
        **icp_info,
        "method": "fallback_similarity_icp_complete_to_moge",
        "source_complete": str(complete_path),
        "target_moge": str(moge_path),
        "hunyuan_to_moge": np.asarray(transform, dtype=np.float64).tolist(),
        "complete_to_moge_nn": complete_to_moge_stats,
        "moge_to_complete_nn": moge_to_complete_stats,
        "outputs": {
            "registered_complete": str(registered_path),
            "merged": str(merged_path),
        },
    }


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def run(args):
    sample_dir = Path(args.sample_dir)
    complete_path = Path(args.complete)
    moge_path = Path(args.moge)
    out_dir = Path(args.out_dir) if args.out_dir else sample_dir / "fallback_hunyuan_to_moge_icp"
    out_dir.mkdir(parents=True, exist_ok=True)

    complete = o3d.io.read_point_cloud(str(complete_path))
    moge = o3d.io.read_point_cloud(str(moge_path))
    aligned, transform, icp_info = run_similarity_icp(
        complete,
        moge,
        max_correspondence_distance=args.max_correspondence_distance,
        voxel_size=args.voxel_size,
    )

    registered_path = out_dir / f"{args.prefix}_registered_to_moge_icp.ply"
    merged_path = out_dir / f"{args.prefix}_moge_plus_complete_icp.ply"
    json_path = out_dir / f"{args.prefix}_moge_icp_registration.json"
    o3d.io.write_point_cloud(str(registered_path), aligned)
    moge_vis = paint_or_keep(moge, [0.1, 0.55, 1.0])
    complete_vis = o3d.geometry.PointCloud(aligned)
    complete_vis.paint_uniform_color([1.0, 0.45, 0.1])
    o3d.io.write_point_cloud(str(merged_path), moge_vis + complete_vis)

    info = build_registration_info(
        complete_path=complete_path,
        moge_path=moge_path,
        transform=transform,
        icp_info=icp_info,
        registered_path=registered_path,
        merged_path=merged_path,
        complete_to_moge_stats=nearest_neighbor_stats(aligned, moge),
        moge_to_complete_stats=nearest_neighbor_stats(moge, aligned),
    )
    save_json(json_path, info)
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--complete", required=True)
    parser.add_argument("--moge", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--prefix", default="car__132_hunyuan")
    parser.add_argument("--max_correspondence_distance", type=float, default=0.08)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
