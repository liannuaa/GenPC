#!/usr/bin/env python3
"""Materialize a class-balanced MVP subset for the bounded agentic probe.

The complete MVP shapes are emitted into a separate offline-only directory.
Every agent tool consumes only the paired partial PLY and its H5 class label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d


MVP_CATEGORIES = (
    "airplane", "cabinet", "car", "chair", "lamp", "sofa", "table", "watercraft",
    "bed", "bench", "bookshelf", "bus", "guitar", "motorbike", "pistol", "skateboard",
)


def _write_cloud(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"cannot write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.per_category <= 0:
        parser.error("--per-category must be positive")
    root, h5_path = args.output_root.resolve(), args.h5.resolve()
    manifest_path = root / "inference_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; use --overwrite")

    with h5py.File(h5_path, "r") as handle:
        partials, completes = handle["incomplete_pcds"], handle["complete_pcds"]
        labels = np.asarray(handle["labels"], dtype=np.int64)
        if len(partials) % len(completes):
            raise ValueError("partial count is not an integral number of complete-shape views")
        views = len(partials) // len(completes)
        rng = np.random.default_rng(int(args.seed))
        records: list[dict[str, object]] = []
        for label_id, category in enumerate(MVP_CATEGORIES):
            available = np.flatnonzero(labels == label_id)
            if len(available) < args.per_category:
                raise ValueError(f"{category} has only {len(available)} partials")
            selected = np.sort(rng.choice(available, size=int(args.per_category), replace=False))
            for partial_index in selected.tolist():
                complete_index, view_index = divmod(int(partial_index), int(views))
                case_key = f"mvp_test_{partial_index:05d}"
                partial_path = root / "inputs" / "partial" / f"{case_key}.ply"
                gt_path = root / "ground_truth" / f"{case_key}.ply"
                _write_cloud(partial_path, np.asarray(partials[partial_index]))
                _write_cloud(gt_path, np.asarray(completes[complete_index]))
                records.append({
                    "case_key": case_key, "source_h5": str(h5_path),
                    "partial_index": int(partial_index), "complete_index": complete_index,
                    "view_index": view_index, "views_per_complete": int(views),
                    "label_id": label_id, "category": category,
                    "inference_input": str(partial_path.resolve()),
                    "offline_ground_truth": str(gt_path.resolve()),
                })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "dataset": "MVP Completion", "split": h5_path.name,
        "selection": {"kind": "class_balanced_random_partial_views", "seed": int(args.seed),
                      "per_category": int(args.per_category), "count": len(records)},
        "strict_zero_shot": True, "ground_truth_used_during_inference": False,
        "coordinate_contract": "Supplied MVP canonical coordinates are preserved; complete clouds are offline-only.",
        "categories": list(MVP_CATEGORIES), "cases": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "count": len(records),
                      "samples": [record["case_key"] for record in records]}, indent=2))


if __name__ == "__main__":
    main()
