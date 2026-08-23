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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import metric
from scripts.run_zero_shot_posterior_guidance import load_config


SAMPLES = ("01184", "05117", "05452", "06127", "06145",
           "06188", "06830", "07136", "07306", "09639")
CANDIDATE_ROOT = (
    ROOT / "gpt_version/_pixal_bidirectional_cycle_registration_forced_audit_20260823")
V15_ROOT = ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822"


def candidate_path(root: Path, sample: str) -> Path:
    return root / sample / f"{sample}_bidirectional_cycle_registered_100k.ply"


def v15_path(root: Path, sample: str) -> Path:
    stem = root / sample / f"{sample}_unified_registration_v14"
    return Path(f"{stem}_registered_100k.ply")


def evaluate_variant(base_cfg, variant: str, paths: dict[str, Path]):
    cfg = copy.deepcopy(base_cfg)
    cfg.metric_pred_paths = {key: str(value.resolve()) for key, value in paths.items()}
    rows = []
    for sample in SAMPLES:
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--metric-seed", type=int, default=6145)
    args = parser.parse_args(argv)
    output_root = (args.output_root or
                   (args.candidate_root / "postfreeze_cd_emd_20260823"))
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config), "cuda")
    cfg.metric_seed = int(args.metric_seed)
    cfg.metric_seed_overrides = {}
    cfg.metric_indices_dir = str((output_root / "fps_indices").resolve())
    cfg.metric_save_indices = True

    variants = {
        "bidirectional_forced": {
            sample: candidate_path(args.candidate_root, sample) for sample in SAMPLES},
        "v15": {sample: v15_path(args.v15_root, sample) for sample in SAMPLES},
    }
    rows = []
    for variant, paths in variants.items():
        rows.extend(evaluate_variant(cfg, variant, paths))

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
        "shared_fps_indices": True,
        "metric_num_points": int(getattr(cfg, "metric_num_points", 16384)),
        "samples": list(SAMPLES),
        "summary": summary,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
