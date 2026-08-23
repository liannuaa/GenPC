#!/usr/bin/env python3
"""Postprocess frozen consensus registration with observation-conditioned mass."""

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
from src.observation_conditioned_surface_projection import (
    select_observation_conditioned_postprocess,
)


SAMPLES = ("01184", "05117", "05452", "06127", "06145",
           "06188", "06830", "07136", "07306", "09639")
INPUT_ROOT = ROOT / "gpt_version/_pixal_bidirectional_consensus_forced_audit_20260823"
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_bidirectional_consensus_surface_projection_20260823"


def process(args, sample):
    input_path = args.input_root / sample / f"{sample}_bidirectional_cycle_registered_100k.ply"
    body = base.load_points(input_path)
    partial = base.load_points(ROOT / "data" / f"{sample}.ply")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera_root / sample / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")
    result, info = select_observation_conditioned_postprocess(
        body, partial, projector, diagonal=diagonal, seed=args.seed,
        mass_budgets=tuple(args.mass_budgets),
        prior_mass_penalty=args.prior_mass_penalty)
    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_bidirectional_cycle"
    registered = Path(f"{stem}_registered_100k.ply")
    compare = Path(f"{stem}_partial_gray_pixal_red.ply")
    projection = Path(f"{stem}_projection.png")
    diagnostics = Path(f"{stem}_surface_projection_info.json")
    base.write_points(registered, result)
    base.write_compare(compare, partial, result)
    base.draw_projection_overlay(
        projection, args.camera_root / sample / "img.png", partial, result, projector)
    record = {
        "sample_id": sample,
        "method": "bidirectional_consensus_plus_observation_conditioned_surface_projection",
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "sample_or_category_specific_parameters": False,
        "input": input_path,
        "postprocess": info,
        "shared_parameters": {
            "mass_budgets": args.mass_budgets,
            "prior_mass_penalty": args.prior_mass_penalty,
            "seed": args.seed,
        },
        "outputs": {"registered": registered, "compare": compare,
                    "projection": projection},
    }
    diagnostics.write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(sample, info["selected_route"], info["before"]["objective"],
          "->", info["after"]["objective"], flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--mass-budgets", nargs="+", type=float,
                        default=[.04, .08, .12])
    parser.add_argument("--prior-mass-penalty", type=float, default=.008)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=6145)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = [process(args, sample) for sample in args.samples]
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        fields = ["sample_id", "route", "before", "after", "prior_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in records:
            info = record["postprocess"]
            writer.writerow({
                "sample_id": record["sample_id"],
                "route": info["selected_route"],
                "before": info["before"]["objective"],
                "after": info["after"]["objective"],
                "prior_fraction": info["selected_info"].get(
                    "complete_prior_fraction_preserved", 1.0),
            })


if __name__ == "__main__":
    main()
