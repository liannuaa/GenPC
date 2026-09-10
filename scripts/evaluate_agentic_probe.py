#!/usr/bin/env python3
"""Perform an explicitly post-acceptance offline metric audit for a probe."""

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

from src.offline_metrics import evaluate_cd_emd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument(
        "--prediction",
        type=Path,
        help="Single accepted prediction for an isolated retry trace. Requires --sample-id.",
    )
    parser.add_argument("--sample-id", help="Redwood id paired with --prediction.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Exact offline-only ground-truth path for a single prediction.",
    )
    parser.add_argument("--ground-truth-root", type=Path, default=ROOT / "data" / "redwood" / "gt")
    parser.add_argument(
        "--ground-truth-template",
        default="{sample}.ply",
        help="Relative path below --ground-truth-root; '{sample}' is expanded.",
    )
    parser.add_argument("--samples", nargs="+")
    parser.add_argument(
        "--prediction-template",
        default="final/agent_selected_100k.ply",
        help="Relative prediction path below each <collection-root>/<sample>; '{sample}' is expanded.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=6145)
    args = parser.parse_args()

    if args.prediction is not None:
        if not args.sample_id:
            parser.error("--prediction requires --sample-id")
        ground_truth = args.ground_truth or args.ground_truth_root / str(args.ground_truth_template).format(
            sample=args.sample_id,
        )
        pairs = [(str(args.sample_id), args.prediction, ground_truth)]
    else:
        if args.collection_root is None or not args.samples:
            parser.error("provide --collection-root with --samples, or --prediction with --sample-id")
        template = str(args.prediction_template)
        if Path(template).is_absolute():
            parser.error("--prediction-template must be relative to each sample directory")
        pairs = [
            (
                sample,
                args.collection_root / sample / template.format(sample=sample),
                args.ground_truth_root / str(args.ground_truth_template).format(sample=sample),
            )
            for sample in args.samples
        ]

    rows = []
    for sample, prediction, ground_truth in pairs:
        if not prediction.is_file():
            raise FileNotFoundError(f"{sample} has not reached ACCEPT: {prediction}")
        cd, emd = evaluate_cd_emd(prediction, ground_truth, count=args.count, seed=args.seed)
        rows.append({"sample_id": sample, "cd_l1_x1e2": 100.0 * cd, "emd_x1e2": 100.0 * emd})
    summary = {
        "offline_only": True,
        "used_by_agent_decisions": False,
        "count": int(args.count),
        "seed": int(args.seed),
        "mean_cd_l1_x1e2": float(np.mean([row["cd_l1_x1e2"] for row in rows])),
        "mean_emd_x1e2": float(np.mean([row["emd_x1e2"] for row in rows])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
