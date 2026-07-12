import argparse
import sys
from argparse import Namespace
from pathlib import Path

import open3d as o3d
import yaml
from munch import Munch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main
from scripts.run_scansalon import build_cfg as build_scansalon_cfg
from utils.runtime import normalize_runtime_config, resolve_path


def denoise_point_cloud(input_path, output_path, nb_neighbors, std_ratio, radius, min_neighbors):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.io.read_point_cloud(str(input_path))
    raw_count = len(pcd.points)
    if raw_count == 0:
        raise ValueError(f"Empty point cloud: {input_path}")

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


def load_first_scansalon_sample(args):
    scan_args = Namespace(
        config=args.config,
        dataset_root=args.dataset_root,
        workspace=args.source_workspace,
        gt_points=100000,
        metric_num_points=16384,
        min_partial_points=args.min_partial_points,
        max_samples=1,
        pipeline_mode="per_sample",
        save_intermediates=True,
        skip_existing=False,
        skip_gt_prepare=True,
        no_denoise_partials=True,
        denoise_nb_neighbors=args.denoise_nb_neighbors,
        denoise_std_ratio=args.denoise_std_ratio,
        denoise_radius=args.denoise_radius,
        denoise_min_neighbors=args.denoise_min_neighbors,
    )
    cfg = build_scansalon_cfg(scan_args)
    sample_id = cfg.sample_ids[0]
    return sample_id, cfg.input_paths[sample_id], cfg.prompt_overrides[sample_id]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset_root", default="/opt/data/private/cr/resources/ScanSalon")
    parser.add_argument("--source_workspace", default="workspace/scansalon")
    parser.add_argument("--workspace", default="workspace/scansalon_ablation")
    parser.add_argument("--min_partial_points", type=int, default=1000)
    parser.add_argument("--denoise_nb_neighbors", type=int, default=20)
    parser.add_argument("--denoise_std_ratio", type=float, default=1.5)
    parser.add_argument("--denoise_radius", type=float, default=0.04)
    parser.add_argument("--denoise_min_neighbors", type=int, default=8)
    return parser.parse_args()


def main_ablation():
    args = parse_args()
    source_sample_id, raw_input_path, category = load_first_scansalon_sample(args)
    raw_input_path = Path(raw_input_path)

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = resolve_path(workspace)
    input_dir = workspace / "_inputs"
    denoised_input_path = input_dir / f"{source_sample_id}_denoised.ply"
    raw_count, denoised_count = denoise_point_cloud(
        raw_input_path,
        denoised_input_path,
        args.denoise_nb_neighbors,
        args.denoise_std_ratio,
        args.denoise_radius,
        args.denoise_min_neighbors,
    )
    print(
        f"Denoised {source_sample_id}: raw {raw_count} points -> "
        f"{denoised_count} points"
    )

    with open(args.config, "r") as f:
        cfg = Munch.fromDict(yaml.safe_load(f))
    cfg.paths.output_dir = str(workspace)
    cfg.sample_ids = []
    cfg.input_paths = {}
    cfg.prompt_overrides = {}

    for variant, input_path in (
        ("raw", raw_input_path),
        ("denoised", denoised_input_path),
    ):
        sample_id = f"{source_sample_id}_{variant}"
        cfg.sample_ids.append(sample_id)
        cfg.input_paths[sample_id] = str(input_path)
        cfg.prompt_overrides[sample_id] = category

    cfg.dataset = "scansalon_ablation"
    cfg.run_stage1 = True
    cfg.run_stage2 = False
    cfg.run_metric = False
    cfg.skip_existing = False
    cfg.normalize_input = False
    if not hasattr(cfg, "outputs") or cfg.outputs is None:
        cfg.outputs = Munch()
    cfg.outputs.save_intermediates = True
    normalize_runtime_config(cfg)
    print(f"Running Stage 1 ablation for {len(cfg.sample_ids)} variants.")
    main(cfg)


if __name__ == "__main__":
    main_ablation()
