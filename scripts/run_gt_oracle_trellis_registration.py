#!/usr/bin/env python3
"""Offline-only TRELLIS-to-GT Sim(3) oracle; never use for inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline_metrics import evaluate_cd_emd
from src.oracle_similarity import gt_oracle_similarity
from src.pointcloud_io import jsonable, load_points, write_compare, write_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=20_000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=1.0,
        help="Complete-surface fraction used by the GT-only oracle (default: all points).",
    )
    args = parser.parse_args()

    source = load_points(args.prior)
    ground_truth = load_points(args.ground_truth)
    partial = load_points(args.partial)
    result, info = gt_oracle_similarity(
        source,
        ground_truth,
        sample_points=args.sample_points,
        iterations=args.iterations,
        trim_fraction=args.trim_fraction,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "trellis_gt_oracle_registered_100k.ply"
    write_points(prediction, result)
    write_compare(output / "gt_gray_trellis_red.ply", ground_truth, result)
    write_compare(output / "partial_gray_trellis_red.ply", partial, result)
    cd, emd = evaluate_cd_emd(prediction, args.ground_truth, count=16_384, seed=6145)
    info["offline_metrics_raw"] = {"cd_l1": float(cd), "emd": float(emd)}
    info["offline_metrics_x100"] = {"cd_l1": float(cd * 100.0), "emd": float(emd * 100.0)}
    info["inputs"] = {
        "prior": str(args.prior.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "partial": str(args.partial.resolve()),
    }
    info["output"] = str(prediction)
    (output / "trellis_gt_oracle_info.json").write_text(
        json.dumps(jsonable(info), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(prediction), "cd_l1_x100": cd * 100.0,
                      "emd_x100": emd * 100.0,
                      "scale": info["scale"]}, indent=2))


if __name__ == "__main__":
    main()
