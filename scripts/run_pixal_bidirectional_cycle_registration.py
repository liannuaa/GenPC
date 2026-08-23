#!/usr/bin/env python3
"""Pilot the independent bidirectional/cycle-consistent 2D+3D candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.bidirectional_cycle_registration import (
    bidirectional_cycle_step,
    invert_proper_sim3,
    sim3_parts,
    visible_score,
)
from src.ray_consistent_registration import apply_transform


V15_ROOT = ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822"
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_bidirectional_cycle_registration_pilot_20260823"
DEFAULT_SAMPLES = ("09639", "07136")


def v15_paths(root, sample):
    stem = root / sample / f"{sample}_unified_registration_v14"
    return {
        "points": Path(f"{stem}_registered_100k.ply"),
        "mesh": Path(f"{stem}_registered_mesh.glb"),
    }


def passes_final_guard(initial, final, cycle, improvement_ratio):
    return bool(
        np.isfinite(final["objective"])
        and final["objective"] <= float(improvement_ratio) * initial["objective"]
        and final["geometric"]["objective"]
            <= 1.002 * initial["geometric"]["objective"]
        and final["projection"]["coverage"]
            >= initial["projection"]["coverage"] - 0.02
        and final["projection"]["iou"] >= initial["projection"]["iou"] - 0.01
        and cycle["exact_inverse_cycle_rms"] <= 1e-9
        and cycle["independent_reverse_cycle_rms"] <= 0.03
    )


def process(args, sample):
    started = time.perf_counter()
    inputs = v15_paths(args.v15_root, sample)
    initial = base.load_points(inputs["points"])
    partial = base.load_points(ROOT / "data" / f"{sample}.ply")
    if len(initial) != 100000:
        raise ValueError(f"{sample}: frozen v15 input is not 100k points")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera_root / sample / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")
    initial_score = visible_score(
        partial, initial, projector, diagonal, args.final_pixel_radius)

    current = initial.copy()
    total_inverse = np.eye(4, dtype=np.float64)
    trace = []
    last_cycle = {
        "exact_inverse_cycle_rms": 0.0,
        "independent_reverse_cycle_rms": 0.0,
    }
    for iteration, pixel_radius in enumerate(args.pixel_schedule):
        moved, inverse_step, info = bidirectional_cycle_step(
            current, partial, projector, diagonal=diagonal,
            pixel_radius=float(pixel_radius),
            max_rotation_deg=args.max_rotation_deg,
            scale_bounds=(args.min_step_scale, args.max_step_scale),
            max_translation_ratio=args.max_translation_ratio,
            min_pairs=args.min_pairs, max_cycle_ratio=args.max_cycle_ratio,
            return_best_candidate=args.force_candidate_output,
        )
        if info["accepted"] or info.get("candidate_exposed_for_audit", False):
            current = moved
            total_inverse = inverse_step @ total_inverse
            last_cycle = info.get("cycle", last_cycle)
        trace.append({"iteration": iteration, "pixel_radius": pixel_radius, **info})

    final_score = visible_score(
        partial, current, projector, diagonal, args.final_pixel_radius)
    accepted = passes_final_guard(
        initial_score, final_score, last_cycle, args.final_improvement_ratio)
    if not accepted and not args.force_candidate_output:
        current = initial.copy()
        total_inverse = np.eye(4, dtype=np.float64)
        final_score = initial_score
        route = "exact_v15_fallback"
        final_cycle = {
            "exact_inverse_cycle_rms": 0.0,
            "independent_reverse_cycle_rms": 0.0,
        }
    else:
        route = ("bidirectional_cycle_2d3d" if accepted else
                 "bidirectional_cycle_2d3d_forced_audit")
        final_forward = invert_proper_sim3(total_inverse)
        exact = apply_transform(apply_transform(partial, final_forward), total_inverse)
        final_cycle = dict(last_cycle)
        final_cycle["cumulative_exact_inverse_cycle_rms"] = float(
            np.sqrt(np.mean(np.sum((exact - partial) ** 2, axis=1))) / diagonal)

    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_bidirectional_cycle"
    outputs = {
        "registered": Path(f"{stem}_registered_100k.ply"),
        "compare": Path(f"{stem}_partial_gray_pixal_red.ply"),
        "projection": Path(f"{stem}_projection.png"),
        "mesh": Path(f"{stem}_registered_mesh.glb"),
        "transform": Path(f"{stem}.npy"),
        "info": Path(f"{stem}_info.json"),
    }
    base.write_points(outputs["registered"], current)
    base.write_compare(outputs["compare"], partial, current)
    base.draw_projection_overlay(
        outputs["projection"], args.camera_root / sample / "img.png",
        partial, current, projector)
    np.save(outputs["transform"], total_inverse)
    mesh = trimesh.load(inputs["mesh"], force="scene", process=False)
    mesh.apply_transform(total_inverse)
    mesh.export(outputs["mesh"])
    scale, rotation, _ = sim3_parts(total_inverse)
    elapsed = float(time.perf_counter() - started)
    record = {
        "sample_id": sample,
        "method": "bidirectional_cycle_consistent_saved_camera_2d3d_proper_sim3",
        "accepted": accepted,
        "forced_candidate_output": bool(args.force_candidate_output),
        "selected_route": route,
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "anisotropic_scale_used": False,
        "v15_input_read_only": True,
        "input_v15": inputs,
        "initial_visible_score": initial_score,
        "final_visible_score": final_score,
        "final_cycle": final_cycle,
        "cumulative_pixal_to_partial_transform": total_inverse,
        "cumulative_scale": scale,
        "proper_rotation_determinant": float(np.linalg.det(rotation)),
        "point_count_preserved": len(current) == len(initial) == 100000,
        "elapsed_seconds": elapsed,
        "trace": trace,
        "shared_parameters": {
            key: value for key, value in vars(args).items() if key != "samples"
        },
        "outputs": outputs,
    }
    outputs["info"].write_text(
        json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(
        f"{sample} route={route} objective={initial_score['objective']:.8f}"
        f"->{final_score['objective']:.8f} coverage="
        f"{initial_score['projection']['coverage']:.6f}->"
        f"{final_score['projection']['coverage']:.6f} cycle="
        f"{final_cycle.get('independent_reverse_cycle_rms', 0.0):.8f} "
        f"time={elapsed:.2f}s", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15-root", type=Path, default=V15_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--pixel-schedule", nargs="+", type=float,
                        default=[8.0, 5.0, 3.0])
    parser.add_argument("--final-pixel-radius", type=float, default=5.0)
    parser.add_argument("--max-rotation-deg", type=float, default=3.0)
    parser.add_argument("--min-step-scale", type=float, default=0.96)
    parser.add_argument("--max-step-scale", type=float, default=1.04)
    parser.add_argument("--max-translation-ratio", type=float, default=0.03)
    parser.add_argument("--min-pairs", type=int, default=96)
    parser.add_argument("--max-cycle-ratio", type=float, default=0.03)
    parser.add_argument("--final-improvement-ratio", type=float, default=0.995)
    parser.add_argument(
        "--force-candidate-output", action="store_true",
        help="Export the best bidirectional candidate even when a gate rejects it.")
    parser.add_argument("--padding", type=float, default=0.15)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = [process(args, str(sample)) for sample in args.samples]
    fields = [
        "sample_id", "selected_route", "accepted", "initial_objective",
        "final_objective", "initial_coverage", "final_coverage",
        "independent_cycle_rms", "exact_cycle_rms", "elapsed_seconds",
    ]
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "sample_id": record["sample_id"],
                "selected_route": record["selected_route"],
                "accepted": record["accepted"],
                "initial_objective": record["initial_visible_score"]["objective"],
                "final_objective": record["final_visible_score"]["objective"],
                "initial_coverage": record["initial_visible_score"]["projection"]["coverage"],
                "final_coverage": record["final_visible_score"]["projection"]["coverage"],
                "independent_cycle_rms": record["final_cycle"].get(
                    "independent_reverse_cycle_rms", 0.0),
                "exact_cycle_rms": record["final_cycle"].get(
                    "cumulative_exact_inverse_cycle_rms",
                    record["final_cycle"].get("exact_inverse_cycle_rms", 0.0)),
                "elapsed_seconds": record["elapsed_seconds"],
            })


if __name__ == "__main__":
    main()
