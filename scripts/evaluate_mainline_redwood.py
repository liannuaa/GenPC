#!/usr/bin/env python3
"""Offline-only CD-L1/EMD audit for final mainline point clouds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mainline_paths import (
    REDWOOD10_SAMPLE_IDS,
    locate_mainline_prediction,
    redwood_ground_truth_root,
)
from src.offline_metrics import evaluate_cd_emd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, default=redwood_ground_truth_root(ROOT))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=list(REDWOOD10_SAMPLE_IDS),
                        help="Prediction/ground-truth identifiers to evaluate offline.")
    parser.add_argument("--count", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=6145)
    args = parser.parse_args()

    rows = []
    for sample in args.samples:
        prediction = locate_mainline_prediction(args.prediction_root, sample)
        cd, emd = evaluate_cd_emd(prediction, args.ground_truth_root / f"{sample}.ply", count=args.count, seed=args.seed)
        rows.append({"sample_id": sample, "cd_l1_x1e2": 100.0 * cd, "emd_x1e2": 100.0 * emd})
    summary = {
        "offline_only": True,
        "metrics_used_by_mainline": False,
        "count": int(args.count), "seed": int(args.seed),
        "mean_cd_l1_x1e2": float(np.mean([row["cd_l1_x1e2"] for row in rows])),
        "mean_emd_x1e2": float(np.mean([row["emd_x1e2"] for row in rows])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
