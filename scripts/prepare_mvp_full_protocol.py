#!/usr/bin/env python3
"""Materialize a resumable inference-only shard of MVP's 41,600 test partials.

No complete point cloud is written to the shard.  Evaluation reads complete
targets directly from the official H5 only after inference finishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mvp_protocol import MVP_CATEGORIES, build_test_records


def _write_cloud(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"cannot write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True, help="Half-open test index bound.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.stop <= args.start:
        parser.error("require 0 <= start < stop")
    root, h5_path = args.output_root.resolve(), args.h5.resolve()
    with h5py.File(h5_path, "r") as handle:
        records = build_test_records(
            partial_count=len(handle["incomplete_pcds"]), complete_count=len(handle["complete_pcds"]),
            labels=np.asarray(handle["labels"], dtype=np.int64), start=args.start, stop=args.stop,
        )
        for record in records:
            partial_path = root / "inputs" / "partial" / f"{record['case_key']}.ply"
            if not partial_path.exists() or args.overwrite:
                _write_cloud(partial_path, np.asarray(handle["incomplete_pcds"][int(record["partial_index"])]))
            record["inference_input"] = str(partial_path)

    manifest = root / f"inference_manifest_{args.start:05d}_{args.stop:05d}.json"
    manifest.write_text(json.dumps({
        "dataset": "MVP Completion", "split": h5_path.name,
        "selection": {"kind": "official_contiguous_test_shard", "start": args.start,
                      "stop": args.stop, "count": len(records)},
        "strict_zero_shot": True, "ground_truth_used_during_inference": False,
        "coordinate_contract": "Supplied MVP canonical partial coordinates are preserved; no complete cloud is materialized.",
        "categories": list(MVP_CATEGORIES), "cases": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "count": len(records), "input_root": str(root / 'inputs' / 'partial')}, indent=2))


if __name__ == "__main__":
    main()
