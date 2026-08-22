#!/usr/bin/env python3
"""Evaluate an existing Redwood fused-output root with GenPC's protocol."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_registration_deformation_fusion_ablation import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"))
    parser.add_argument("--seed", type=int, default=6145)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("genpc_fused_metric", PROJECT_ROOT / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = copy.deepcopy(load_config(args.config))
    cfg.paths.output_dir = str(args.output_root)
    cfg.metric_seed = int(args.seed)

    rows = []
    for sample in map(str, args.samples):
        cd, emd = module.metric(sample, cfg)
        row = {
            "sample_id": sample,
            "cd_l1_x1e2": 100.0 * float(cd),
            "emd_x1e2": 100.0 * float(emd),
        }
        rows.append(row)
        print(f"{sample}: {row['cd_l1_x1e2']:.6f}/{row['emd_x1e2']:.6f}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "metrics_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
