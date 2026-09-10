#!/usr/bin/env python3
"""Offline MVP evaluation: 16,384-point prediction, CD-L2×10⁴, F-score@1%.

The script deliberately receives ground truth only after an inference
collection has completed.  It supports a subset manifest now and full-test
shards later; neither mode exposes complete clouds to any generation,
registration, or agent action.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mvp_protocol import build_test_records
from src.offline_metrics import evaluate_mvp_cd_l2_fscore


def _manifest_records(manifest_path: Path) -> list[dict[str, object]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("cases")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{manifest_path} has no non-empty cases list")
    return [dict(record) for record in records]


def _prediction_path(collection_root: Path, case_key: str) -> Path:
    return collection_root / "samples" / case_key / "final" / "agent_selected_100k.ply"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="Official MVP test H5 containing complete_pcds.")
    parser.add_argument("--collection-root", type=Path, required=True,
                        help="Root with samples/<case>/final/agent_selected_100k.ply.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="Existing subset or shard manifest.")
    mode.add_argument("--full-range", nargs=2, type=int, metavar=("START", "STOP"),
                      help="Evaluate the half-open official-test range [START, STOP).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON; adjacent CSV is also written.")
    parser.add_argument("--prediction-count", type=int, default=16384)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Write completed rows and list unfinished cases instead of failing.")
    args = parser.parse_args()
    if args.prediction_count != 16384:
        parser.error("The paper-facing MVP protocol requires --prediction-count 16384")
    if abs(args.fscore_threshold - 0.01) > 1e-12:
        parser.error("The paper-facing MVP protocol requires --fscore-threshold 0.01")

    h5_path = args.h5.resolve()
    collection_root = args.collection_root.resolve()
    if args.manifest:
        records = _manifest_records(args.manifest.resolve())
        record_source = str(args.manifest.resolve())
    else:
        with h5py.File(h5_path, "r") as handle:
            records = build_test_records(
                partial_count=len(handle["incomplete_pcds"]), complete_count=len(handle["complete_pcds"]),
                labels=np.asarray(handle["labels"], dtype=np.int64),
                start=args.full_range[0], stop=args.full_range[1],
            )
        record_source = f"official range [{args.full_range[0]}, {args.full_range[1]})"

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    with h5py.File(h5_path, "r") as handle:
        complete_clouds = handle["complete_pcds"]
        for ordinal, record in enumerate(records, start=1):
            key = str(record["case_key"])
            prediction = _prediction_path(collection_root, key)
            if not prediction.is_file():
                missing.append(key)
                if not args.allow_missing:
                    raise FileNotFoundError(f"missing prediction for {key}: {prediction}")
                continue
            metric = evaluate_mvp_cd_l2_fscore(
                prediction,
                np.asarray(complete_clouds[int(record["complete_index"])]),
                prediction_count=int(args.prediction_count),
                fscore_threshold=float(args.fscore_threshold),
            )
            row = {
                "case_key": key,
                "partial_index": int(record["partial_index"]),
                "complete_index": int(record["complete_index"]),
                "category": str(record["category"]),
                **metric,
            }
            rows.append(row)
            print(f"[{ordinal}/{len(records)}] {key}: CD-L2x1e4={metric['cd_l2_x1e4']:.6f}, "
                  f"F1@1%={metric['fscore_1pct']:.6f}", flush=True)

    means = {
        "cd_l2_x1e4": float(np.mean([float(row["cd_l2_x1e4"]) for row in rows])) if rows else None,
        "fscore_1pct": float(np.mean([float(row["fscore_1pct"]) for row in rows])) if rows else None,
        "precision_1pct": float(np.mean([float(row["precision_1pct"]) for row in rows])) if rows else None,
        "recall_1pct": float(np.mean([float(row["recall_1pct"]) for row in rows])) if rows else None,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "MVP Completion test",
        "protocol": {
            "prediction_sampling": "deterministic FPS, start_idx=0",
            "prediction_points": int(args.prediction_count),
            "complete_target_points": "native official H5 cardinality (2,048 for MVP_Test_CP.h5)",
            "metrics": ["CD-L2 x 1e4", "F-score@1%"],
            "fscore_threshold_euclidean": float(args.fscore_threshold),
            "ground_truth_used_during_inference": False,
        },
        "source_h5": str(h5_path),
        "record_source": record_source,
        "collection_root": str(collection_root),
        "requested_cases": len(records),
        "evaluated_cases": len(rows),
        "missing_cases": missing,
        "mean": means,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fieldnames = list(rows[0]) if rows else ["case_key", "partial_index", "complete_index", "category", "cd_l2_x1e4", "fscore_1pct"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(output), "csv": str(csv_path), "mean": means,
                      "evaluated": len(rows), "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
