import yaml
import torch
import gc
import argparse
import csv
from pathlib import Path
from munch import Munch
import numpy as np
import open3d as o3d
from collections import defaultdict
from utils.dataUtils import (
    getCategory,
    load_xyz,
    normalize_numpy,
    resolve_prompt_label,
)
from DepthPrompting import DepthPrompting
from ScaleAdapter import ScaleAdapter
from tools.hunyuan3d_2 import release_hunyuan3d_cache
from utils.loss_util import Completionloss
from utils.runtime import (
    cleanup_stage1_intermediates,
    cleanup_intermediates,
    data_dir,
    gt_dir,
    normalize_runtime_config,
    output_dir,
    resolve_path,
    sample_file,
)
from fpsample import fps_sampling

import warnings
warnings.filterwarnings("ignore")


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resolve_flags(cfg):
    sample_ids = list(getattr(cfg, "sample_ids", []) or [])
    if sample_ids:
        return sample_ids

    sample_ids_file = getattr(cfg, "sample_ids_file", None)
    if sample_ids_file:
        ids_path = resolve_path(sample_ids_file)
        with open(ids_path, "r") as f:
            flags = [line.strip() for line in f if line.strip()]
        max_samples = getattr(cfg, "max_samples", None)
        if max_samples:
            flags = flags[: int(max_samples)]
        return flags

    flags = sorted(path.stem for path in data_dir(cfg).glob("*.ply"))
    max_samples = getattr(cfg, "max_samples", None)
    if max_samples:
        flags = flags[: int(max_samples)]
    return flags


def resolve_input_path(cfg, flag):
    input_paths = getattr(cfg, "input_paths", {}) or {}
    if flag in input_paths:
        return str(resolve_path(input_paths[flag]))

    direct_path = Path(flag)
    if direct_path.exists():
        return str(direct_path.resolve())

    for suffix in (".ply", ".pcd"):
        candidate = data_dir(cfg) / f"{flag}{suffix}"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Input point cloud for '{flag}' not found. "
        f"Add it under {data_dir(cfg)} or set cfg.input_paths."
    )


def resolve_box_path(cfg, flag):
    input_boxes = getattr(cfg, "input_boxes", {}) or {}
    box_path = input_boxes.get(flag)
    return str(resolve_path(box_path)) if box_path else None


def resolve_gt_path(cfg, flag):
    gt_paths = getattr(cfg, "gt_paths", {}) or {}
    if flag in gt_paths:
        return str(resolve_path(gt_paths[flag]))
    return str(gt_dir(cfg) / f"{flag}.ply")


def resolve_metric_pred_path(cfg, flag):
    metric_pred_paths = getattr(cfg, "metric_pred_paths", {}) or {}
    if flag in metric_pred_paths:
        return str(resolve_path(metric_pred_paths[flag]))
    return str(output_dir(cfg) / flag / f"{flag}_fused.ply")


def normalize_with_box(xyz_np, box_path, normalize_range):
    box_xyz = np.loadtxt(box_path, dtype=np.float32)
    if box_xyz.ndim != 2 or box_xyz.shape[1] != 3:
        raise ValueError(f"Invalid box file format: {box_path}")

    box_min = box_xyz.min(axis=0)
    box_max = box_xyz.max(axis=0)
    center = (box_min + box_max) / 2.0
    scale_factor = float((box_max - box_min).max())
    scale = float(normalize_range) / 0.5
    normalized = (xyz_np - center) / scale_factor
    normalized *= scale
    return normalized


def load_sample(cfg, flag):
    xyz_np, rgb_np = load_xyz(resolve_input_path(cfg, flag))
    if getattr(cfg, "normalize_input", False):
        normalize_range = float(getattr(cfg, "input_normalize_range", 0.5))
        box_path = resolve_box_path(cfg, flag)
        if box_path:
            xyz_np = normalize_with_box(xyz_np, box_path, normalize_range)
        else:
            xyz_np, _, _ = normalize_numpy(xyz_np, range=normalize_range)
    return xyz_np, rgb_np

def metric(flag, cfg):
    """计算CD和EMD指标"""
    metric_seed_overrides = getattr(cfg, "metric_seed_overrides", {}) or {}
    metric_seed = metric_seed_overrides.get(str(flag), getattr(cfg, "metric_seed", None))

    # 读取GT和预测结果
    gt = o3d.io.read_point_cloud(resolve_gt_path(cfg, flag), format="ply")
    pred = o3d.io.read_point_cloud(resolve_metric_pred_path(cfg, flag), format="ply")
    
    # 获取点云坐标
    gt_points = np.asarray(gt.points).astype(np.float32)
    pred_points = np.asarray(pred.points).astype(np.float32)

    metric_indices_dir = getattr(cfg, "metric_indices_dir", None)
    indices_path = None
    if metric_indices_dir:
        indices_path = resolve_path(metric_indices_dir) / f"{flag}.npz"

    if indices_path is not None and indices_path.exists():
        indices = np.load(indices_path)
        gt_indices = indices["gt_indices"]
        pred_indices = indices["pred_indices"]
        gt_max_index = int(gt_indices.max()) if len(gt_indices) else -1
        pred_max_index = int(pred_indices.max()) if len(pred_indices) else -1
        if gt_max_index >= len(gt_points) or pred_max_index >= len(pred_points):
            gt_indices = None
            pred_indices = None
    else:
        gt_indices = None
        pred_indices = None

    sample_count = min(
        int(getattr(cfg, "metric_num_points", 16384)),
        len(gt_points),
        len(pred_points),
    )
    if sample_count <= 0:
        raise ValueError(f"Metric point cloud is empty for '{flag}'.")

    if gt_indices is None or pred_indices is None or len(gt_indices) != sample_count or len(pred_indices) != sample_count:
        gt_start_idx = None
        pred_start_idx = None
        if metric_seed is not None:
            rng = np.random.default_rng(int(metric_seed) % (2**32))
            gt_start_idx = int(rng.integers(0, len(gt_points)))
            pred_start_idx = int(rng.integers(0, len(pred_points)))

        gt_indices = fps_sampling(gt_points, sample_count, start_idx=gt_start_idx)
        pred_indices = fps_sampling(pred_points, sample_count, start_idx=pred_start_idx)
        if bool(getattr(cfg, "metric_save_indices", False)) and indices_path is not None:
            indices_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(indices_path, gt_indices=gt_indices, pred_indices=pred_indices)
    gt_xyz = gt_points[gt_indices]
    pred_xyz = pred_points[pred_indices]

    gt_tensor = torch.from_numpy(gt_xyz).unsqueeze(0).float().cuda()
    pred_tensor = torch.from_numpy(pred_xyz).unsqueeze(0).float().cuda()
    
    completion_cd = Completionloss(loss_func='cd_l1')
    completion_emd = Completionloss(loss_func='emd')
    
    cd = completion_cd.get_loss(gen=pred_tensor, gt=gt_tensor)
    emd = completion_emd.get_loss(gen=pred_tensor, gt=gt_tensor)
    
    try:
        label = getCategory(flag)
    except KeyError:
        label = flag
    print(f"Flag: {label}, CD-L1 x1e2: {cd.item() * 100:.3f}, EMD x1e2: {emd.item() * 100:.3f}")
    return cd.item(), emd.item()


def write_metric_results(cfg, results, verbose=False):
    if not results:
        return None, None

    output_dir(cfg).mkdir(parents=True, exist_ok=True)
    sample_metrics_path = output_dir(cfg) / "metrics_samples.csv"
    with open(sample_metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "category",
                "cd_l1",
                "emd",
                "cd_l1_x1e2",
                "emd_x1e2",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample_id": result["sample_id"],
                    "category": result["flag"],
                    "cd_l1": result["cd"],
                    "emd": result["emd"],
                    "cd_l1_x1e2": result["cd"] * 100,
                    "emd_x1e2": result["emd"] * 100,
                }
            )

    category_results = defaultdict(list)
    for result in results:
        category_results[result["flag"]].append(result)
    category_metrics_path = output_dir(cfg) / "metrics_by_category.csv"
    with open(category_metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "count",
                "cd_l1",
                "emd",
                "cd_l1_x1e2",
                "emd_x1e2",
            ],
        )
        writer.writeheader()
        for category in sorted(category_results):
            category_items = category_results[category]
            category_cd = sum(r["cd"] for r in category_items) / len(category_items)
            category_emd = sum(r["emd"] for r in category_items) / len(category_items)
            writer.writerow(
                {
                    "category": category,
                    "count": len(category_items),
                    "cd_l1": category_cd,
                    "emd": category_emd,
                    "cd_l1_x1e2": category_cd * 100,
                    "emd_x1e2": category_emd * 100,
                }
            )
            if verbose:
                print(
                    f"Category: {category}, Count: {len(category_items)}, "
                    f"CD-L1 x1e2: {category_cd * 100:.6f}, EMD x1e2: {category_emd * 100:.6f}"
                )
    return sample_metrics_path, category_metrics_path


def main(cfg):
    """主函数：按配置执行 Stage 1 / Stage 2 / metric。"""
    flags = resolve_flags(cfg)
    if not flags:
        raise FileNotFoundError(
            f"No input samples found. Add .ply files under {data_dir(cfg)} or set cfg.sample_ids."
        )

    run_stage1 = getattr(cfg, "run_stage1", True)
    run_stage2 = getattr(cfg, "run_stage2", True)
    run_metric = getattr(cfg, "run_metric", True)
    pipeline_mode = str(getattr(cfg, "pipeline_mode", "per_sample"))
    results = []
    skip_existing = bool(getattr(cfg, "skip_existing", False))

    preview_count = min(5, len(flags))
    preview = ", ".join(str(flag) for flag in flags[:preview_count])
    suffix = "..." if len(flags) > preview_count else ""
    print(f"Running {len(flags)} flags: {preview}{suffix}")

    def run_stage1_for_flag(dp, flag):
        print(f'Processing {flag}...')
        xyz_np, rgb_np = load_sample(cfg, flag)
        xyz = torch.tensor(xyz_np).to(cfg.device)
        rgb = torch.tensor(rgb_np).to(cfg.device)
        dp.getImage(xyz=xyz, flag=flag, rgb=rgb, depth_gen=True, img_gen=True)
        del xyz, rgb, xyz_np, rgb_np
        cleanup_stage1_intermediates(cfg, flag)
        free_memory()

    def record_metric(flag):
        cd, emd = metric(flag, cfg)
        results.append({
            'sample_id': str(flag),
            'flag': resolve_prompt_label(flag, cfg),
            'cd': cd,
            'emd': emd
        })
        write_metric_results(cfg, results)

    if run_stage1 and run_stage2 and pipeline_mode == "phased":
        print("\n=== Stage 1 + Stage 2: Phased pipeline ===")
        stage1_flags = []
        for flag in flags:
            fused_path = sample_file(cfg, flag, f"{flag}_fused.ply")
            stage2_done = skip_existing and fused_path.exists()
            stage1_ready = (
                sample_file(cfg, flag, "depth.png").exists()
                and sample_file(cfg, flag, "img.png").exists()
                and (stage2_done or sample_file(cfg, flag, "point_uv.npy").exists())
            )
            if skip_existing and stage1_ready:
                print(f" Skip Stage 1 for {flag}: required outputs already exist.")
                continue
            stage1_flags.append(flag)

        if stage1_flags:
            release_hunyuan3d_cache()
            free_memory()
            dp = DepthPrompting(cfg)
            try:
                for flag in stage1_flags:
                    run_stage1_for_flag(dp, flag)
            finally:
                dp.close()
                del dp
                free_memory()
        else:
            print("All Stage 1 outputs already exist; skipping DepthPrompting initialization.")

        sa = ScaleAdapter(cfg)
        try:
            for flag in flags:
                fused_path = sample_file(cfg, flag, f"{flag}_fused.ply")
                if skip_existing and fused_path.exists():
                    print(f" Skip Stage 2 for {flag}: {fused_path.name} already exists.")
                else:
                    xyz_np, _ = load_sample(cfg, flag)
                    xyz = torch.tensor(xyz_np).to(cfg.device)
                    sa.scaleAdapter(xyz, flag)
                    sa.scaleReg(flag)
                    del xyz, xyz_np
                if run_metric:
                    record_metric(flag)
                cleanup_intermediates(cfg, flag)
                free_memory()
        finally:
            del sa
            free_memory()

    elif run_stage1 and run_stage2:
        print("\n=== Stage 1 + Stage 2: Per-sample pipeline ===")
        dp = None
        sa = ScaleAdapter(cfg)
        try:
            for flag in flags:
                fused_path = sample_file(cfg, flag, f"{flag}_fused.ply")
                stage2_done = skip_existing and fused_path.exists()
                stage1_ready = (
                    sample_file(cfg, flag, "depth.png").exists()
                    and sample_file(cfg, flag, "img.png").exists()
                    and (stage2_done or sample_file(cfg, flag, "point_uv.npy").exists())
                )
                if skip_existing and stage1_ready:
                    print(f" Skip Stage 1 for {flag}: required outputs already exist.")
                else:
                    if dp is None:
                        release_hunyuan3d_cache()
                        free_memory()
                        dp = DepthPrompting(cfg)
                    run_stage1_for_flag(dp, flag)
                    dp.close()
                    del dp
                    dp = None
                    free_memory()

                if stage2_done:
                    print(f" Skip Stage 2 for {flag}: {fused_path.name} already exists.")
                else:
                    xyz_np, _ = load_sample(cfg, flag)
                    xyz = torch.tensor(xyz_np).to(cfg.device)
                    sa.scaleAdapter(xyz, flag)
                    sa.scaleReg(flag)
                    del xyz, xyz_np
                if run_metric:
                    record_metric(flag)
                cleanup_intermediates(cfg, flag)
                free_memory()
        finally:
            if dp is not None:
                del dp
            del sa
            free_memory()

    elif run_stage1:
        print("\n=== Stage 1: Depth Prompting ===")
        stage1_flags = []
        for flag in flags:
            if (
                skip_existing
                and sample_file(cfg, flag, "depth.png").exists()
                and sample_file(cfg, flag, "img.png").exists()
            ):
                print(f" Skip Stage 1 for {flag}: depth.png and img.png already exist.")
                continue
            stage1_flags.append(flag)
        if not stage1_flags:
            print("All Stage 1 outputs already exist; skipping DepthPrompting initialization.")
        else:
            dp = DepthPrompting(cfg)
            for flag in stage1_flags:
                run_stage1_for_flag(dp, flag)
                cleanup_intermediates(cfg, flag)
            del dp
            free_memory()

    elif run_stage2:
        print("\n=== Stage 2: Scale Adapter ===")
        sa = ScaleAdapter(cfg)
        for flag in flags:
            fused_path = sample_file(cfg, flag, f"{flag}_fused.ply")
            if skip_existing and fused_path.exists():
                print(f" Skip Stage 2 for {flag}: {fused_path.name} already exists.")
            else:
                xyz_np, _ = load_sample(cfg, flag)
                xyz = torch.tensor(xyz_np).to(cfg.device)
                sa.scaleAdapter(xyz, flag)
                sa.scaleReg(flag)
                del xyz, xyz_np
            if run_metric:
                record_metric(flag)
            cleanup_intermediates(cfg, flag)
            free_memory()
        del sa
        free_memory()

    if results:
        print("\n=== 结果总结 ===")
        for result in results:
            print(f"Category: {result['flag']}, CD-L1 x1e2: {result['cd'] * 100:.6f}, EMD x1e2: {result['emd'] * 100:.6f}")

        avg_cd = sum(r['cd'] for r in results) / len(results)
        avg_emd = sum(r['emd'] for r in results) / len(results)
        print(f"平均 CD-L1 x1e2: {avg_cd * 100:.6f}")
        print(f"平均 EMD x1e2: {avg_emd * 100:.6f}")

        print("\n=== 按类别结果 ===")
        sample_metrics_path, category_metrics_path = write_metric_results(
            cfg, results, verbose=True
        )
        print(f"Saved sample metrics to {sample_metrics_path}")
        print(f"Saved category metrics to {category_metrics_path}")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="*",
        help="Optional sample ids that override config sample_ids.",
    )
    parser.add_argument(
        "--workspace",
        help="Optional output workspace directory that overrides paths.output_dir.",
    )
    parser.add_argument(
        "--models_dir",
        help="Optional models directory that overrides paths.models_dir.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse existing per-sample outputs when possible.",
    )
    parser.add_argument(
        "--save_intermediates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep intermediate/debug files for this run.",
    )
    args = parser.parse_args()

    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    cfg.paths = getattr(cfg, "paths", Munch())
    if args.sample_ids is not None:
        cfg.sample_ids = args.sample_ids
    if args.workspace:
        cfg.paths.output_dir = args.workspace
    if args.models_dir:
        cfg.paths.models_dir = args.models_dir
    if args.skip_existing is not None:
        cfg.skip_existing = args.skip_existing
    if args.save_intermediates is not None:
        cfg.outputs.save_intermediates = args.save_intermediates
    normalize_runtime_config(cfg)
    main(cfg)
