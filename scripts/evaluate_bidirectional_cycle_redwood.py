#!/usr/bin/env python3
"""Post-freeze Redwood CD-L1/EMD audit for bidirectional candidates and v15."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from munch import Munch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import metric
from utils.runtime import normalize_runtime_config


SAMPLES = ("01184", "05117", "05452", "06127", "06145",
           "06188", "06830", "07136", "07306", "09639")
CANDIDATE_ROOT = (
    ROOT / "gpt_version/_pixal_bidirectional_cycle_registration_forced_audit_20260823")
# Historical baselines intentionally live in the shared project root.  A
# worktree has its own source checkout but does not duplicate multi-GB frozen
# experiments, so resolving this relative to ``ROOT`` breaks reproducible
# single-sample agent audits.
V15_ROOT = PROJECT_ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822"


def load_config(path, device):
    """Local config loader keeps this post-freeze evaluator self-contained."""
    cfg = Munch.fromDict(yaml.safe_load(Path(path).read_text()))
    cfg.paths = getattr(cfg, "paths", Munch())
    cfg.device = str(device)
    return normalize_runtime_config(cfg)


def candidate_path(root: Path, sample: str, template: str) -> Path:
    return root / sample / template.format(sample=sample)


def v15_path(root: Path, sample: str) -> Path:
    stem = root / sample / f"{sample}_unified_registration_v14"
    return Path(f"{stem}_registered_100k.ply")


def evaluate_variant(base_cfg, variant: str, paths: dict[str, Path], index_root: Path,
                     samples):
    cfg = copy.deepcopy(base_cfg)
    cfg.metric_pred_paths = {key: str(value.resolve()) for key, value in paths.items()}
    # Prediction clouds may have different point order after resampling or
    # surface-mass projection.  Each geometry therefore needs its own FPS
    # indices; the shared seed still makes GT FPS equivalent across variants.
    cfg.metric_indices_dir = str((index_root / variant).resolve())
    rows = []
    for sample in samples:
        if not paths[sample].exists():
            raise FileNotFoundError(paths[sample])
        cd, emd = metric(sample, cfg)
        rows.append({
            "variant": variant,
            "sample_id": sample,
            "cd_l1_x1e2": 100.0 * float(cd),
            "emd_x1e2": 100.0 * float(emd),
        })
        torch.cuda.empty_cache()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--v15-root", type=Path, default=V15_ROOT)
    parser.add_argument("--extra-baseline-root", type=Path,
                        help="Optional frozen baseline root for an additional post-freeze comparison.")
    parser.add_argument("--extra-baseline-name", default="extra_baseline")
    parser.add_argument("--extra-baseline-template",
                        default="{sample}_bidirectional_cycle_registered_uniform.ply")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--metric-seed", type=int, default=6145)
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument(
        "--candidate-name-template",
        default="{sample}_bidirectional_cycle_registered_100k.ply",
        help="Per-sample prediction filename; supports the {sample} field.")
    args = parser.parse_args(argv)
    output_root = (args.output_root or
                   (args.candidate_root / "postfreeze_cd_emd_20260823"))
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config), "cuda")
    cfg.metric_seed = int(args.metric_seed)
    cfg.metric_seed_overrides = {}
    cfg.metric_save_indices = True

    samples = tuple(map(str, args.samples))
    variants = {
        "bidirectional_forced": {
            sample: candidate_path(
                args.candidate_root, sample, args.candidate_name_template)
            for sample in samples},
        "v15": {sample: v15_path(args.v15_root, sample) for sample in samples},
    }
    if args.extra_baseline_root is not None:
        variants[str(args.extra_baseline_name)] = {
            sample: candidate_path(
                args.extra_baseline_root, sample, args.extra_baseline_template)
            for sample in samples
        }
    rows = []
    for variant, paths in variants.items():
        rows.extend(evaluate_variant(
            cfg, variant, paths, output_root / "fps_indices", samples))

    sample_path = output_root / "metrics_samples.csv"
    with sample_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append({
            "variant": variant,
            "samples": len(selected),
            "mean_cd_l1_x1e2": float(np.mean([x["cd_l1_x1e2"] for x in selected])),
            "mean_emd_x1e2": float(np.mean([x["emd_x1e2"] for x in selected])),
        })
    summary_path = output_root / "metrics_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    protocol = {
        "post_freeze": True,
        "metrics_used_for_registration_or_routing": False,
        "metric_seed": int(args.metric_seed),
        "shared_prediction_fps_indices": False,
        "shared_metric_seed": True,
        "equivalent_gt_fps_by_shared_seed": True,
        "metric_num_points": int(getattr(cfg, "metric_num_points", 16384)),
        "candidate_name_template": args.candidate_name_template,
        "extra_baseline": (
            None if args.extra_baseline_root is None else {
                "name": str(args.extra_baseline_name),
                "root": str(args.extra_baseline_root),
                "template": args.extra_baseline_template,
            }),
        "samples": list(samples),
        "summary": summary,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
