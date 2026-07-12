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
DEFAULT_PARTIAL = PROJECT_ROOT / "workspace" / "scansalon" / "_inputs_denoised" / "car" / "car__132.ply"


def compose_complete_to_partial(complete_to_moge, moge_to_partial):
    complete_to_moge = np.asarray(complete_to_moge, dtype=np.float64)
    moge_to_partial = np.asarray(moge_to_partial, dtype=np.float64)
    if complete_to_moge.shape != (4, 4) or moge_to_partial.shape != (4, 4):
        raise ValueError(
            f"Transforms must be 4x4, got {complete_to_moge.shape} and {moge_to_partial.shape}"
        )
    return moge_to_partial @ complete_to_moge


def apply_transform(points, transform):
    points = np.asarray(points, dtype=np.float64)
    hom = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ hom.T).T[:, :3]


def read_freereg_complete_to_moge(path):
    data = json.loads(Path(path).read_text())
    if "hunyuan_to_moge" in data:
        return np.asarray(data["hunyuan_to_moge"], dtype=np.float64), data
    outputs = data.get("outputs", {})
    if "hunyuan_to_moge" in outputs:
        return np.asarray(outputs["hunyuan_to_moge"], dtype=np.float64), data
    raise KeyError(f"FreeReg JSON does not contain hunyuan_to_moge: {path}")


def paint_or_keep(pcd, color):
    out = deepcopy(pcd)
    if not out.has_colors():
        out.paint_uniform_color(color)
    return out


def write_outputs(
    complete_path,
    partial_path,
    complete_to_partial,
    output_prefix,
):
    import open3d as o3d

    complete = o3d.io.read_point_cloud(str(complete_path))
    partial = o3d.io.read_point_cloud(str(partial_path))
    complete_points = np.asarray(complete.points, dtype=np.float64)
    aligned_points = apply_transform(complete_points, complete_to_partial)

    aligned = deepcopy(complete)
    aligned.points = o3d.utility.Vector3dVector(aligned_points)

    prefix = Path(output_prefix)
    aligned_path = prefix.with_name(prefix.name + "_complete_aligned_to_raw_partial.ply")
    compare_path = prefix.with_name(prefix.name + "_raw_partial_gray_complete_blue_aligned.ply")
    transform_path = prefix.with_name(prefix.name + "_complete_to_raw_partial_transform.npy")

    o3d.io.write_point_cloud(str(aligned_path), aligned)
    partial_vis = paint_or_keep(partial, [0.55, 0.55, 0.55])
    complete_vis = deepcopy(aligned)
    complete_vis.paint_uniform_color([0.05, 0.25, 1.0])
    o3d.io.write_point_cloud(str(compare_path), partial_vis + complete_vis)
    np.save(transform_path, complete_to_partial)
    return aligned_path, compare_path, transform_path


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def run(args):
    sample_dir = Path(args.sample_dir)
    complete_to_moge, freereg_info = read_freereg_complete_to_moge(args.freereg_json)
    moge_to_partial = np.load(args.moge_to_partial_transform)
    complete_to_partial = compose_complete_to_partial(
        complete_to_moge=complete_to_moge,
        moge_to_partial=moge_to_partial,
    )

    output_prefix = sample_dir / args.output_prefix
    aligned_path, compare_path, transform_path = write_outputs(
        complete_path=args.complete,
        partial_path=args.partial,
        complete_to_partial=complete_to_partial,
        output_prefix=output_prefix,
    )
    info_path = output_prefix.with_name(output_prefix.name + "_info.json")
    info = {
        "complete": str(args.complete),
        "partial": str(args.partial),
        "freereg_json": str(args.freereg_json),
        "moge_to_partial_transform": str(args.moge_to_partial_transform),
        "complete_to_moge": complete_to_moge.tolist(),
        "moge_to_partial": moge_to_partial.tolist(),
        "complete_to_partial": complete_to_partial.tolist(),
        "freereg_summary": {
            "backend": freereg_info.get("backend"),
            "matches": freereg_info.get("matches"),
            "inliers": freereg_info.get("inliers"),
            "rmse": freereg_info.get("rmse"),
        },
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
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--complete", required=True)
    parser.add_argument("--partial", default=str(DEFAULT_PARTIAL))
    parser.add_argument("--freereg_json", required=True)
    parser.add_argument(
        "--moge_to_partial_transform",
        default=str(DEFAULT_SAMPLE_DIR / "moge_to_raw_partial_transform.npy"),
    )
    parser.add_argument("--output_prefix", default="car__132_complete_to_raw_partial")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
