import yaml
import torch
import gc
import argparse
from pathlib import Path
from munch import Munch
import numpy as np
import open3d as o3d
from utils.dataUtils import (
    getCategory,
    load_xyz,
    normalize_numpy,
    resolve_prompt_label,
)
from DepthPrompting import DepthPrompting
from ScaleAdapter import ScaleAdapter
from utils.loss_util import Completionloss
from utils.runtime import (
    cleanup_intermediates,
    data_dir,
    gt_dir,
    normalize_runtime_config,
    output_dir,
    resolve_path,
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
    else:
        gt_start_idx = None
        pred_start_idx = None
        if metric_seed is not None:
            rng = np.random.default_rng(int(metric_seed) % (2**32))
            gt_start_idx = int(rng.integers(0, len(gt_points)))
            pred_start_idx = int(rng.integers(0, len(pred_points)))

        gt_indices = fps_sampling(gt_points, 16384, start_idx=gt_start_idx)
        pred_indices = fps_sampling(pred_points, 16384, start_idx=pred_start_idx)
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
    print(f"Flag: {label}, CD: {cd.item() * 100:.3f}, EMD: {emd.item() * 100:.3f}")
    return cd.item(), emd.item()

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
    results = []

    print(f"Running flags: {flags}")

    if run_stage1:
        print("\n=== Stage 1: Depth Prompting ===")
        dp = DepthPrompting(cfg)
        for flag in flags:
            print(f'Processing {flag}...')
            xyz_np, rgb_np = load_sample(cfg, flag)
            xyz = torch.tensor(xyz_np).to(cfg.device)
            rgb = torch.tensor(rgb_np).to(cfg.device)
            dp.getImage(xyz=xyz, flag=flag, rgb=rgb, depth_gen=True, img_gen=True)
            del xyz, rgb, xyz_np, rgb_np
            free_memory()
        del dp
        free_memory()

    if run_stage2:
        print("\n=== Stage 2: Scale Adapter ===")
        sa = ScaleAdapter(cfg)
        for flag in flags:
            xyz_np, _ = load_sample(cfg, flag)
            xyz = torch.tensor(xyz_np).to(cfg.device)
            sa.scaleAdapter(xyz, flag)
            sa.scaleReg(flag)
            if run_metric:
                cd, emd = metric(flag, cfg)
                results.append({
                    'flag': resolve_prompt_label(flag, cfg),
                    'cd': cd,
                    'emd': emd
                })
            cleanup_intermediates(cfg, flag)
            del xyz, xyz_np
            free_memory()
        del sa
        free_memory()

    if results:
        print("\n=== 结果总结 ===")
        for result in results:
            print(f"Category: {result['flag']}, CD: {result['cd'] * 100:.6f}, EMD: {result['emd'] * 100:.6f}")

        avg_cd = sum(r['cd'] for r in results) / len(results)
        avg_emd = sum(r['emd'] for r in results) / len(results)
        print(f"平均 CD: {avg_cd * 100:.6f}")
        print(f"平均 EMD: {avg_emd * 100:.6f}")
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
    normalize_runtime_config(cfg)
    main(cfg)
