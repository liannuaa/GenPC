#!/usr/bin/env python3
"""Uniformize frozen fusion outputs with support-aware voxel representatives."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.voxel_surface_measure_resampling import support_aware_voxel_resample


SAMPLES = ("01184", "05117", "05452", "06127", "06145",
           "06188", "06830", "07136", "07306", "09639")


def process(args, sample):
    source_dir = args.input_root / sample
    stem_name = f"{sample}_bidirectional_cycle"
    input_path = source_dir / f"{stem_name}_registered_100k.ply"
    info_path = source_dir / f"{stem_name}_surface_projection_info.json"
    source_info = json.loads(info_path.read_text())
    observation_count = int(source_info["postprocess"].get(
        "selected_info", {}).get("exact_partial_points_inserted", 0))
    points = base.load_points(input_path)
    result, info = support_aware_voxel_resample(
        points, observation_count=observation_count,
        target_points=args.target_points, outlier_k=args.outlier_k,
        outlier_mad_scale=args.outlier_mad_scale,
        support_ratio=args.support_ratio)

    partial = base.load_points(ROOT / "data" / f"{sample}.ply")
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera_root / sample / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")
    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / stem_name
    registered = Path(f"{stem}_registered_uniform.ply")
    compare = Path(f"{stem}_partial_gray_pixal_red_uniform.ply")
    projection = Path(f"{stem}_projection_uniform.png")
    diagnostics = Path(f"{stem}_voxel_resampling_info.json")
    base.write_points(registered, result)
    base.write_compare(compare, partial, result)
    base.draw_projection_overlay(
        projection, args.camera_root / sample / "img.png", partial, result, projector)
    record = {
        "sample_id": sample,
        "method": "support_aware_voxel_surface_measure_resampling",
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "sample_or_category_specific_parameters": False,
        "input": str(input_path),
        "source_surface_info": str(info_path),
        "resampling": info,
        "shared_parameters": {
            "target_points": args.target_points,
            "outlier_k": args.outlier_k,
            "outlier_mad_scale": args.outlier_mad_scale,
            "support_ratio": args.support_ratio,
        },
        "outputs": {"registered": str(registered), "compare": str(compare),
                    "projection": str(projection)},
    }
    diagnostics.write_text(json.dumps(record, indent=2))
    print(sample, len(points), "->", len(result),
          "knn_cv", f"{info['global_knn_cv_before']:.3f}", "->",
          f"{info['global_knn_cv_after']:.3f}", "outliers",
          info["observation_outliers_removed"], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--target-points", type=int, default=32768)
    parser.add_argument("--outlier-k", type=int, default=8)
    parser.add_argument("--outlier-mad-scale", type=float, default=4.0)
    parser.add_argument("--support-ratio", type=float, default=.02)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = [process(args, sample) for sample in args.samples]
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        fields = ["sample_id", "input_points", "output_points",
                  "knn_cv_before", "knn_cv_after", "outliers_removed"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in records:
            info = record["resampling"]
            writer.writerow({
                "sample_id": record["sample_id"],
                "input_points": info["input_points"],
                "output_points": info["output_points"],
                "knn_cv_before": info["global_knn_cv_before"],
                "knn_cv_after": info["global_knn_cv_after"],
                "outliers_removed": info["observation_outliers_removed"],
            })


if __name__ == "__main__":
    main()
