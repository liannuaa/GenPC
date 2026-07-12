import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import yaml
from munch import Munch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main
from utils.runtime import normalize_runtime_config, resolve_path


DEFAULT_DATASET_ROOT = Path("/opt/data/private/cr/resources/ScanSalon")


def sample_id(category, relative_pcd_path):
    return f"{category}__{Path(relative_pcd_path).stem}"


def sample_mesh_to_ply(mesh_path, ply_path, num_points, seed):
    if ply_path.exists():
        return

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        points = mesh.sample(num_points)
    finally:
        np.random.set_state(state)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)


def deterministic_seed(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def load_scansalon_metadata(dataset_root):
    metadata_path = dataset_root / "metadata.csv"
    rows = []
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            pcd_rel = row["pcd_filename"]
            mesh_rel = row["mesh_filename"]
            sid = sample_id(category, pcd_rel)
            rows.append(
                {
                    "sample_id": sid,
                    "category": category,
                    "pcd_path": dataset_root / pcd_rel,
                    "mesh_path": dataset_root / mesh_rel,
                }
            )
    return rows


def count_points(pcd_path):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    return len(pcd.points)


def denoise_point_cloud(input_path, output_path, nb_neighbors, std_ratio, radius, min_neighbors):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.io.read_point_cloud(str(input_path))
    raw_count = len(pcd.points)
    if raw_count == 0:
        raise ValueError(f"Empty point cloud: {input_path}")
    if output_path.exists():
        existing_pcd = o3d.io.read_point_cloud(str(output_path))
        return raw_count, len(existing_pcd.points)

    sor_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    radius_pcd, _ = sor_pcd.remove_radius_outlier(
        nb_points=min_neighbors,
        radius=radius,
    )
    if len(radius_pcd.points) >= max(200, raw_count // 3):
        chosen = radius_pcd
    elif len(sor_pcd.points) >= max(200, raw_count // 3):
        chosen = sor_pcd
    else:
        chosen = pcd
    o3d.io.write_point_cloud(str(output_path), chosen, write_ascii=False)
    return raw_count, len(chosen.points)


def build_cfg(args):
    with open(args.config, "r") as f:
        cfg = Munch.fromDict(yaml.safe_load(f))

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    rows = load_scansalon_metadata(dataset_root)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        rows = [row for row in rows if row["sample_id"] in wanted]
        missing = sorted(wanted.difference(row["sample_id"] for row in rows))
        if missing:
            raise ValueError(f"Requested ScanSalon sample ids not found: {', '.join(missing)}")
    total_rows = len(rows)
    if args.min_partial_points > 0:
        kept_rows = []
        filtered_rows = []
        for row in rows:
            point_count = count_points(row["pcd_path"])
            row["partial_points"] = point_count
            if point_count >= args.min_partial_points:
                kept_rows.append(row)
            else:
                filtered_rows.append(row)
        rows = kept_rows
        print(
            f"Filtered ScanSalon partials: kept {len(rows)}/{total_rows}, "
            f"removed {len(filtered_rows)} with < {args.min_partial_points} points."
        )
    if args.max_samples:
        rows = rows[: args.max_samples]

    workspace = Path(args.workspace).expanduser()
    if not workspace.is_absolute():
        workspace = resolve_path(workspace)
    gt_dir = workspace / "_gt_mesh_samples"
    denoised_dir = workspace / "_inputs_denoised"

    sample_ids = []
    input_paths = {}
    gt_paths = {}
    prompt_overrides = {}
    metric_seed_overrides = {}
    denoised_count = 0
    filtered_actual_rows = []

    for row in rows:
        sid = row["sample_id"]
        input_path = row["pcd_path"]
        input_point_count = row.get("partial_points")
        if not bool(getattr(args, "no_denoise_partials", False)):
            denoised_path = denoised_dir / row["category"] / f"{sid}.ply"
            raw_points, clean_points = denoise_point_cloud(
                input_path,
                denoised_path,
                args.denoise_nb_neighbors,
                args.denoise_std_ratio,
                args.denoise_radius,
                args.denoise_min_neighbors,
            )
            input_path = denoised_path
            denoised_count += 1
            row["denoised_points"] = clean_points
            row["raw_points"] = raw_points
            input_point_count = clean_points
        elif input_point_count is None:
            input_point_count = count_points(input_path)

        if args.min_partial_points > 0 and input_point_count < args.min_partial_points:
            filtered_actual_rows.append(
                {
                    "sample_id": sid,
                    "category": row["category"],
                    "points": input_point_count,
                    "path": str(input_path),
                }
            )
            continue

        sample_ids.append(sid)
        input_paths[sid] = str(input_path)
        prompt_overrides[sid] = row["category"]

        gt_ply_path = gt_dir / row["category"] / f"{sid}.ply"
        gt_paths[sid] = str(gt_ply_path)
        metric_seed_overrides[sid] = deterministic_seed(sid)
        if not args.skip_gt_prepare:
            sample_mesh_to_ply(
                row["mesh_path"],
                gt_ply_path,
                args.gt_points,
                deterministic_seed(f"gt::{sid}"),
            )

    if filtered_actual_rows:
        print(
            "Filtered ScanSalon actual inputs after denoise/input selection: "
            f"removed {len(filtered_actual_rows)} with < {args.min_partial_points} points."
        )
        preview = ", ".join(
            f"{row['sample_id']}({row['points']})" for row in filtered_actual_rows[:10]
        )
        suffix = "..." if len(filtered_actual_rows) > 10 else ""
        print(f"Filtered actual-input preview: {preview}{suffix}")

    cfg.paths.output_dir = str(workspace)
    cfg.sample_ids = sample_ids
    cfg.input_paths = input_paths
    cfg.gt_paths = gt_paths
    cfg.prompt_overrides = prompt_overrides
    cfg.metric_seed_overrides = metric_seed_overrides
    cfg.dataset = "scansalon"
    cfg.run_stage1 = True
    cfg.run_stage2 = True
    cfg.run_metric = True
    cfg.skip_existing = bool(args.skip_existing)
    cfg.control_model = args.control_model
    if args.generative_model:
        cfg.generative_model = args.generative_model
    cfg.normalize_input = False
    cfg.pipeline_mode = args.pipeline_mode
    cfg.metric_num_points = args.metric_num_points
    cfg.metric_indices_dir = str(workspace / "metric_indices")
    cfg.depth_projection = args.depth_projection
    cfg.hunyuan_omni_point_normalize = args.hunyuan_omni_point_normalize
    cfg.qwen_cpu_offload = bool(args.qwen_cpu_offload)
    cfg.reg_fine_xyz = bool(args.reg_fine_xyz)
    cfg.reg_backend = args.reg_backend
    cfg.reg_pose_iters = int(args.reg_pose_iters)
    cfg.reg_pose_cam_bias_num = int(args.reg_pose_cam_bias_num)
    cfg.reg_coarse_scale_steps = int(args.reg_coarse_scale_steps)

    if not hasattr(cfg, "outputs") or cfg.outputs is None:
        cfg.outputs = Munch()
    cfg.outputs.save_intermediates = bool(args.save_intermediates)

    normalize_runtime_config(cfg)
    print(f"Prepared ScanSalon config with {len(sample_ids)} samples.")
    if denoised_count:
        print(f"Using denoised partials for {denoised_count} samples.")
    else:
        print("Using raw partials.")
    print(f"Control model: {cfg.control_model}")
    print(f"Generative model: {cfg.generative_model}")
    print(f"Depth projection: {cfg.depth_projection}")
    print(f"Hunyuan3D-Omni point normalize: {cfg.hunyuan_omni_point_normalize}")
    print(
        "Registration: "
        f"backend={cfg.reg_backend}, "
        f"pose_iters={cfg.reg_pose_iters}, "
        f"pose_cam_bias_num={cfg.reg_pose_cam_bias_num}, "
        f"coarse_scale_steps={cfg.reg_coarse_scale_steps}, "
        f"fine_xyz={cfg.reg_fine_xyz}"
    )
    print(f"Categories: {', '.join(sorted(set(prompt_overrides.values())))}")
    print(f"Workspace: {cfg.paths.output_dir}")
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--workspace", default="workspace/scansalon")
    parser.add_argument("--sample_ids", nargs="*", help="Optional ScanSalon sample ids, e.g. car__132.")
    parser.add_argument("--gt_points", type=int, default=100000)
    parser.add_argument("--metric_num_points", type=int, default=16384)
    parser.add_argument("--min_partial_points", type=int, default=1000)
    parser.add_argument("--control_model", default="depth_passthrough")
    parser.add_argument("--generative_model", default=None)
    parser.add_argument(
        "--depth_projection",
        default="canonical_view",
        choices=["reference_view", "canonical_view", "view_select", "xz_from_pos_y"],
    )
    parser.add_argument("--hunyuan_omni_point_normalize", default="bbox", choices=["bbox", "none"])
    parser.add_argument("--qwen_cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reg_fine_xyz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reg_backend", default="geotransformer", choices=["geotransformer", "legacy", "one_sided"])
    parser.add_argument("--reg_pose_iters", type=int, default=120)
    parser.add_argument("--reg_pose_cam_bias_num", type=int, default=4)
    parser.add_argument("--reg_coarse_scale_steps", type=int, default=9)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--pipeline_mode", choices=["per_sample", "phased"], default="per_sample")
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_gt_prepare", action="store_true")
    parser.add_argument("--no_denoise_partials", action="store_true")
    parser.add_argument("--denoise_nb_neighbors", type=int, default=20)
    parser.add_argument("--denoise_std_ratio", type=float, default=1.5)
    parser.add_argument("--denoise_radius", type=float, default=0.04)
    parser.add_argument("--denoise_min_neighbors", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main(build_cfg(parse_args()))
