import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataUtils import SHAPENET_CATEGORY_BY_TAXONOMY


def pc_norm(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    return pc / scale


def random_sample_points(points, n_points, rng):
    choice = rng.permutation(points.shape[0])
    sampled = points[choice[:n_points]]
    if sampled.shape[0] < n_points:
        zeros = np.zeros((n_points - sampled.shape[0], 3), dtype=sampled.dtype)
        sampled = np.concatenate([sampled, zeros], axis=0)
    return sampled


def write_xyz_ply(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    o3d.io.write_point_cloud(str(path), pcd)


def load_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_base_config(args, sample_ids_file):
    with open(args.base_config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"] = "shapenet55"
    cfg["sample_ids"] = []
    cfg["sample_ids_file"] = str(sample_ids_file)
    cfg["max_samples"] = args.max_samples
    cfg["run_stage1"] = True
    cfg["run_stage2"] = True
    cfg["run_metric"] = True
    cfg["skip_existing"] = True
    cfg["normalize_input"] = bool(args.normalize)
    cfg["generate_res"] = 512
    cfg["qwen_depth_input_res"] = 512
    cfg["qwen_cpu_offload"] = False
    cfg["qwen_cpu_text_encoder"] = True

    cfg.setdefault("paths", {})
    cfg["paths"]["data_dir"] = str(args.output_dir / "partial")
    cfg["paths"]["gt_dir"] = str(args.output_dir / "complete")
    cfg["paths"]["output_dir"] = "workspace/ShapeNet55"

    cfg.setdefault("outputs", {})
    cfg["outputs"]["save_intermediates"] = bool(args.save_intermediates)
    cfg["outputs"]["keep_files"] = ["img.png", "depth.png", "{flag}_fused.ply"]
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Projected-ShapeNet55 test partial/complete pairs for GenPC."
    )
    parser.add_argument(
        "--split_file",
        default="/opt/data/private/cr/PoinTr/data/ShapeNet55-34/Projected_ShapeNet-55_noise/test.txt",
        help="PoinTr Projected-ShapeNet55 split file.",
    )
    parser.add_argument(
        "--complete_root",
        default="/opt/data/private/cr/ShapeNet55",
        help="Root containing shapenet_pc/*.npy complete point clouds.",
    )
    parser.add_argument(
        "--partial_root",
        default="/opt/data/private/cr/pcd",
        help="Root containing taxonomy/model/models/*.pcd projected partial point clouds.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/ShapeNet55"),
        help="Output directory for GenPC-ready partial/complete pairs.",
    )
    parser.add_argument(
        "--base_config",
        default="configs/config.yaml",
        help="Base GenPC config to copy runtime/model settings from.",
    )
    parser.add_argument(
        "--config_out",
        type=Path,
        default=Path("configs/config_shapenet55.yaml"),
        help="Generated ShapeNet55 config path.",
    )
    parser.add_argument(
        "--rendering",
        type=int,
        default=0,
        help="Projected partial rendering index. PoinTr uses 0 for test.",
    )
    parser.add_argument(
        "--partial_points",
        type=int,
        default=2048,
        help="Number of partial points after PoinTr-style RandomSamplePoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic partial sampling.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for preparing/debugging a prefix of the split.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable runtime normalization of partial inputs in the generated config.",
    )
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Keep Stage 1/2 intermediate outputs in the generated config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    partial_out = output_dir / "partial"
    complete_out = output_dir / "complete"
    sample_ids_file = output_dir / "test_ids.txt"
    metadata_file = output_dir / "metadata.csv"

    rows = load_split(args.split_file)
    if args.max_samples:
        rows = rows[: args.max_samples]

    sample_ids = []
    metadata = []
    complete_root = Path(args.complete_root)
    partial_root = Path(args.partial_root)

    for idx, line in enumerate(rows):
        filename = Path(line).name
        sample_id = Path(filename).stem
        taxonomy_id, model_id = sample_id.split("-", 1)
        category = SHAPENET_CATEGORY_BY_TAXONOMY.get(taxonomy_id, taxonomy_id)

        complete_path = complete_root / "shapenet_pc" / filename
        partial_path = partial_root / taxonomy_id / model_id / "models" / f"{args.rendering}.pcd"
        if not complete_path.exists():
            raise FileNotFoundError(f"Complete point cloud not found: {complete_path}")
        if not partial_path.exists():
            raise FileNotFoundError(f"Partial point cloud not found: {partial_path}")

        complete = np.load(complete_path).astype(np.float32)
        partial_pcd = o3d.io.read_point_cloud(str(partial_path))
        partial = np.asarray(partial_pcd.points, dtype=np.float32)
        rng = np.random.default_rng(args.seed + idx)
        partial = random_sample_points(partial, args.partial_points, rng)

        write_xyz_ply(partial_out / f"{sample_id}.ply", partial)
        write_xyz_ply(complete_out / f"{sample_id}.ply", complete)

        sample_ids.append(sample_id)
        metadata.append(
            {
                "sample_id": sample_id,
                "taxonomy_id": taxonomy_id,
                "category": category,
                "model_id": model_id,
                "partial_path": str(partial_path),
                "complete_path": str(complete_path),
            }
        )

        if (idx + 1) % 500 == 0:
            print(f"Prepared {idx + 1}/{len(rows)} samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_ids_file, "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")

    with open(metadata_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "taxonomy_id",
                "category",
                "model_id",
                "partial_path",
                "complete_path",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata)

    config = build_base_config(args, sample_ids_file)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.config_out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Prepared {len(sample_ids)} ShapeNet55 samples under {output_dir}")
    print(f"Wrote sample ids: {sample_ids_file}")
    print(f"Wrote metadata: {metadata_file}")
    print(f"Wrote config: {args.config_out}")


if __name__ == "__main__":
    main()
