#!/usr/bin/env python3
"""Decode a fixed, saved-view surfel posterior from an Agent GenPC+ registration."""

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

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.bidirectional_cycle_registration import visible_score
from src.view_conditioned_surfel import (
    build_view_conditioned_surfel_field, coverage, semantic_projection,
    semantic_target, weighted_voxel_decode,
)

SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830",
           "07136", "07306", "09639")


def process(args, sample: str):
    stem = args.input_stem_template.format(sample=sample)
    reference_stem = args.reference_stem_template.format(sample=sample)
    body = base.load_points(args.input_root / sample / f"{stem}_registered_100k.ply")
    partial = base.load_points(args.partial_root / f"{sample}.ply")
    reference = base.load_points(args.reference_root / sample / f"{reference_stem}_registered_uniform.ply")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera_root / sample / "camera.pth", padding=args.padding,
        image_shape=(args.render_size, args.render_size), device="cpu")
    semantic, edge = semantic_target(args.semantic_root / sample / args.semantic_name, args.render_size)
    centres, opacity, field = build_view_conditioned_surfel_field(
        body, partial, projector, semantic, edge, diagonal, args.support_ratio)
    ids, decoder = weighted_voxel_decode(
        centres, opacity, target_points=args.target_points, diagonal=diagonal,
        binary_steps=args.binary_steps)
    candidate = centres[ids]
    before = visible_score(partial, reference, projector, diagonal, pixel_radius=5.)
    after = visible_score(partial, candidate, projector, diagonal, pixel_radius=5.)
    before_semantic = semantic_projection(projector, reference, semantic)
    after_semantic = semantic_projection(projector, candidate, semantic)
    body_coverage, partial_coverage = coverage(body, candidate, diagonal), coverage(partial, candidate, diagonal)
    extent = np.ptp(candidate, axis=0) / np.maximum(np.ptp(centres, axis=0), 1e-12)
    accepted = bool(
        len(candidate) >= int(.97 * args.target_points) and np.all(np.isfinite(candidate))
        and body_coverage["q99_ratio"] <= args.max_complete_q99_ratio
        and partial_coverage["q99_ratio"] <= args.max_partial_q99_ratio
        and float(extent.min()) >= args.min_extent_ratio
        and after["objective"] <= before["objective"] * args.visible_no_harm_ratio
        and after_semantic["coverage"] >= before_semantic["coverage"] - args.semantic_coverage_slack
        and after_semantic["iou"] >= before_semantic["iou"] - args.semantic_iou_slack)
    result = candidate if accepted else reference.copy()
    route = "view_conditioned_surfel_posterior" if accepted else "voxel_posterior_fallback"
    out = args.output_root / sample; out.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem_template.format(sample=sample)
    prediction = out / f"{output_stem}_registered_uniform.ply"
    compare = out / f"{output_stem}_partial_gray_surfel_red.ply"
    info_path = out / f"{output_stem}_view_surfel_info.json"
    base.write_points(prediction, result); base.write_compare(compare, partial, result)
    record = {"sample_id": sample, "method": "fixed_view_conditioned_surfel_posterior",
              "route": route, "strict_zero_shot": True,
              "ground_truth_used_for_inference_or_selection": False,
              "semantic_or_shape_regenerated": False, "point_centres_moved": False,
              "input": {"registered_body": str(args.input_root / sample / f"{stem}_registered_100k.ply"),
                        "reference": str(args.reference_root / sample / f"{reference_stem}_registered_uniform.ply"),
                        "semantic": str(args.semantic_root / sample / args.semantic_name)},
              "field": field, "decoder": decoder,
              "gate": {"accepted": accepted, "reference_visible": before, "candidate_visible": after,
                       "reference_semantic": before_semantic, "candidate_semantic": after_semantic,
                       "complete_coverage": body_coverage, "partial_coverage": partial_coverage,
                       "extent_ratio_xyz": extent.tolist()},
              "shared_parameters": {"render_size": args.render_size, "support_ratio": args.support_ratio,
                                    "target_points": args.target_points, "visible_no_harm_ratio": args.visible_no_harm_ratio},
              "outputs": {"prediction": prediction, "comparison": compare}}
    info_path.write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(sample, route, f"visible {before['objective']:.6f}->{after['objective']:.6f}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, default=ROOT / "data")
    parser.add_argument("--input-stem-template", default="{sample}_agent_bidirectional")
    parser.add_argument("--reference-stem-template", default="{sample}_agent_bidirectional")
    parser.add_argument("--output-stem-template", default="{sample}_agent_bidirectional")
    parser.add_argument("--semantic-name", default="gpt_image.png")
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--support-ratio", type=float, default=.02)
    parser.add_argument("--target-points", type=int, default=32768)
    parser.add_argument("--binary-steps", type=int, default=18)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--max-complete-q99-ratio", type=float, default=.022)
    parser.add_argument("--max-partial-q99-ratio", type=float, default=.022)
    parser.add_argument("--min-extent-ratio", type=float, default=.985)
    parser.add_argument("--visible-no-harm-ratio", type=float, default=1.01)
    parser.add_argument("--semantic-coverage-slack", type=float, default=.01)
    parser.add_argument("--semantic-iou-slack", type=float, default=.01)
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    records = [process(args, str(sample)) for sample in args.samples]
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "route", "before", "after"])
        writer.writeheader()
        for record in records:
            gate = record["gate"]
            writer.writerow({"sample_id": record["sample_id"], "route": record["route"],
                             "before": gate["reference_visible"]["objective"],
                             "after": gate["candidate_visible"]["objective"]})


if __name__ == "__main__":
    main()
