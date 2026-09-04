#!/usr/bin/env python3
"""Refine a freshly registered Pixal prior with saved-view 2D+3D Sim(3).

Inputs must be produced in the same scratch root in the current run.  In
particular, this runner does not load replay semantic images, meshes, frozen
transforms, registered clouds, or ground truth.
"""

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
from src.agent_completion_policy import accept_registration_refinement
from src.bidirectional_consensus_registration import bidirectional_consensus_step
from src.bidirectional_cycle_registration import (
    invert_proper_sim3, partial_to_prior_inverse_step, sim3_parts, visible_score,
)
from src.ray_consistent_registration import apply_transform
from src.saved_view_tto import optimize_saved_view_sim3
from scripts.run_pixal_batched_adaptive_ttt_v12 import confidence_gate


SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830",
           "07136", "07306", "09639")


def _render_score(metrics: dict) -> float:
    return float(metrics["iou"] + .15 * metrics["coverage"] - .45 * metrics["leakage"])


def _initialization_score(full_projection: dict, low_resolution: dict,
                          surface_trim_weight: float) -> float:
    """Shared 2-D+3-D saved-view evidence for choosing an initializer.

    Full-resolution silhouette evidence alone can prefer a shrunken complete
    prior that happens to cover the partial mask.  The normalized partial-to-
    prior trimmed surface residual is already part of the PCA/SO(3) TTT
    objective, so include it symmetrically for both initialization families.
    """
    return _render_score(full_projection) - float(surface_trim_weight) * float(
        low_resolution["normalized_surface_trim70"]
    )


def initial_paths(args: argparse.Namespace, sample: str) -> tuple[dict[str, Path], dict]:
    stem = args.initial_root / sample / f"{sample}_batched_so3_sim3_ttt_v11"
    global_paths = {
        "points": Path(f"{stem}_registered_100k.ply"),
        "mesh": Path(f"{stem}_registered_mesh.glb"),
        "transform": Path(f"{stem}.npy"),
    }
    global_info = json.loads(Path(f"{stem}_info.json").read_text())
    global_projection = global_info["full_resolution_projection"]
    global_low = global_info["selected"]["metrics"]
    decision = {"route": "global", "global_projection": global_projection,
                "global_low_resolution": global_low,
                "global_confident": confidence_gate(global_low)}
    if args.pca_root is None:
        return global_paths, decision
    pca_stem = args.pca_root / sample / f"{sample}_pca_sim3_ttt_v2"
    pca_paths = {"points": Path(f"{pca_stem}_registered_100k.ply"),
                 "mesh": Path(f"{pca_stem}_registered_mesh.glb"),
                 "transform": Path(f"{pca_stem}.npy")}
    pca_info = json.loads(Path(f"{pca_stem}_info.json").read_text())
    pca_projection = pca_info["selected_full_resolution_projection"]
    pca_low = pca_info["selected_low_resolution_metrics"]
    global_score = _initialization_score(
        global_projection, global_low, args.initial_surface_trim_weight
    )
    pca_score = _initialization_score(
        pca_projection, pca_low, args.initial_surface_trim_weight
    )
    choose_global = bool(global_score >= pca_score - args.pca_score_tolerance)
    decision.update({"pca_projection": pca_projection, "global_score": global_score,
                     "pca_score": pca_score, "pca_low_resolution": pca_low,
                     "surface_trim_weight": args.initial_surface_trim_weight,
                     "route": "global" if choose_global else "pca_fallback"})
    return (global_paths if choose_global else pca_paths), decision


def final_guard(initial: dict, final: dict, cycle: dict, improvement: float) -> bool:
    return bool(
        np.isfinite(final["objective"])
        and final["objective"] <= improvement * initial["objective"]
        and final["geometric"]["objective"] <= 1.002 * initial["geometric"]["objective"]
        and final["projection"]["coverage"] >= initial["projection"]["coverage"] - .02
        and final["projection"]["iou"] >= initial["projection"]["iou"] - .01
        and cycle.get("exact_inverse_cycle_rms", 0.) <= 1e-9
        and cycle.get("independent_reverse_cycle_rms", 0.) <= .03
    )


def process(args: argparse.Namespace, sample: str) -> dict:
    started = time.perf_counter()
    inputs, initial_decision = initial_paths(args, sample)
    if not all(path.exists() for path in inputs.values()):
        raise FileNotFoundError(f"{sample}: incomplete fresh coarse input: {inputs}")
    initial = base.load_points(inputs["points"])
    partial = base.load_points(args.partial_root / f"{sample}.ply")
    if len(initial) != 100_000:
        raise ValueError(f"{sample}: expected fresh 100k prior, got {len(initial)}")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.scratch_root / sample / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device=args.device)
    initial_score = visible_score(partial, initial, projector, diagonal, args.final_pixel_radius)

    current = initial.copy()
    total_step = np.eye(4, dtype=np.float64)
    trace = []
    last_cycle = {"exact_inverse_cycle_rms": 0., "independent_reverse_cycle_rms": 0.}
    for iteration, radius in enumerate(args.pixel_schedule):
        direct_moved, direct_step, direct_info = partial_to_prior_inverse_step(
            current, partial, projector, diagonal=diagonal, pixel_radius=float(radius),
            max_rotation_deg=args.max_rotation_deg,
            scale_bounds=(args.min_step_scale, args.max_step_scale),
            max_translation_ratio=args.max_translation_ratio, min_pairs=args.min_pairs,
            return_best_candidate=False)
        consensus_moved, consensus_step, consensus_info = bidirectional_consensus_step(
            current, partial, projector, diagonal=diagonal, pixel_radius=float(radius),
            max_rotation_deg=args.max_rotation_deg,
            scale_bounds=(args.min_step_scale, args.max_step_scale),
            max_translation_ratio=args.max_translation_ratio, min_pairs=args.min_pairs,
            max_cycle_ratio=args.max_cycle_ratio, return_best_candidate=False)
        candidates = [("identity", current, np.eye(4), {"accepted": True,
                       "after": visible_score(partial, current, projector, diagonal, radius),
                       "cycle": last_cycle})]
        if direct_info["accepted"]:
            candidates.append(("partial_to_prior_inverse", direct_moved, direct_step, direct_info))
        if consensus_info["accepted"]:
            candidates.append(("bidirectional_consensus", consensus_moved, consensus_step, consensus_info))
        selected_name, moved, inverse_step, info = min(
            candidates, key=lambda item: item[3]["after"]["objective"])
        if selected_name != "identity":
            current = moved
            total_step = inverse_step @ total_step
            last_cycle = info.get("cycle", last_cycle)
        trace.append({"iteration": iteration, "pixel_radius": radius,
                      "selected_action": selected_name,
                      "direct": direct_info, "consensus": consensus_info})

    final_score = visible_score(partial, current, projector, diagonal, args.final_pixel_radius)
    accepted = final_guard(initial_score, final_score, last_cycle, args.final_improvement_ratio)
    if not accepted:
        current = initial.copy()
        total_step = np.eye(4, dtype=np.float64)
        final_score = initial_score
        selected_route = "coarse_saved_view_fallback"
        final_cycle = {"exact_inverse_cycle_rms": 0., "independent_reverse_cycle_rms": 0.}
    else:
        selected_route = "fresh_bidirectional_saved_view_2d3d_sim3"
        forward = invert_proper_sim3(total_step)
        exact = apply_transform(apply_transform(partial, forward), total_step)
        final_cycle = dict(last_cycle)
        final_cycle["cumulative_exact_inverse_cycle_rms"] = float(
            np.sqrt(np.mean(np.sum((exact - partial) ** 2, axis=1))) / diagonal)

    # An agent action, not an unconditional post-process: v22 showed that a
    # small differentiable screen/surface residual can correct a coarse
    # saved-view fit.  It is bounded to proper Sim(3) and rejected unless its
    # own saved-view evidence is no-harm.
    screen_eligible = bool(final_score["objective"] >= args.screen_trigger_objective)
    screen_action = {"enabled": bool(args.enable_screen_tto), "eligible": screen_eligible,
                     "accepted": False,
                     "reason": "high_residual_saved_view" if screen_eligible else "rigid_route_sufficient"}
    if args.enable_screen_tto and screen_eligible:
        screen_candidate, screen_step, screen_trace = optimize_saved_view_sim3(
            current, partial, projector, diagonal=diagonal,
            pixel_schedule=tuple(args.screen_pixel_schedule), steps=args.screen_steps,
            max_pairs=args.screen_max_pairs, max_rotation_deg=args.screen_max_rotation_deg,
            scale_bounds=(args.screen_min_scale, args.screen_max_scale),
            max_translation_ratio=args.screen_max_translation_ratio,
            seed=args.seed, device=args.device, screen_size=args.screen_render_size,
            screen_points=args.screen_points, screen_weight=args.screen_weight)
        screen_score = visible_score(partial, screen_candidate, projector, diagonal, args.final_pixel_radius)
        screen_accepted = accept_registration_refinement(final_score, screen_score)
        screen_action = {"enabled": True, "eligible": True, "accepted": screen_accepted,
                         "reason": "accepted" if screen_accepted else "saved_view_no_harm_gate",
                         "before": final_score, "after": screen_score, "trace": screen_trace}
        if screen_accepted:
            current = screen_candidate
            total_step = screen_step @ total_step
            final_score = screen_score
            selected_route += "+agent_gpu_screen_tto"

    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_agent_bidirectional"
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
    base.draw_projection_overlay(outputs["projection"], args.scratch_root / sample / "img.png",
                                 partial, current, projector)
    np.save(outputs["transform"], total_step)
    mesh = trimesh.load(inputs["mesh"], force="scene", process=False)
    mesh.apply_transform(total_step)
    mesh.export(outputs["mesh"])
    scale, rotation, _ = sim3_parts(total_step)
    record = {
        "sample_id": sample,
        "method": "fresh_directed_partial_to_prior_plus_bidirectional_witness_saved_view_proper_sim3",
        "strict_zero_shot": True,
        "replay_semantic_or_3d_used": False,
        "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "anisotropic_scale_used": False,
        "initial_registration": str(inputs["points"]),
        "initialization_action": initial_decision,
        "initial_visible_score": initial_score,
        "final_visible_score": final_score,
        "accepted": accepted,
        "selected_route": selected_route,
        "cycle": final_cycle,
        "cumulative_transform": total_step,
        "cumulative_scale": scale,
        "proper_rotation_determinant": float(np.linalg.det(rotation)),
        "point_count_preserved": len(current) == 100_000,
        "elapsed_seconds": float(time.perf_counter() - started),
        "trace": trace,
        "agent_actions": {"gpu_saved_view_screen_tto": screen_action},
        "shared_parameters": {k: v for k, v in vars(args).items() if k != "samples"},
        "outputs": outputs,
    }
    outputs["info"].write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(f"{sample} {selected_route} objective={initial_score['objective']:.6f}->{final_score['objective']:.6f} time={record['elapsed_seconds']:.2f}s", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--pca-root", type=Path, default=None,
                        help="Optional fresh all-PCA initialization root.")
    parser.add_argument("--pca-score-tolerance", type=float, default=.005)
    parser.add_argument("--initial-surface-trim-weight", type=float, default=8.)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, default=ROOT / "data")
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixel-schedule", nargs="+", type=float, default=[8., 5., 3.])
    parser.add_argument("--final-pixel-radius", type=float, default=5.)
    parser.add_argument("--max-rotation-deg", type=float, default=3.)
    parser.add_argument("--min-step-scale", type=float, default=.96)
    parser.add_argument("--max-step-scale", type=float, default=1.04)
    parser.add_argument("--max-translation-ratio", type=float, default=.03)
    parser.add_argument("--min-pairs", type=int, default=96)
    parser.add_argument("--max-cycle-ratio", type=float, default=.03)
    parser.add_argument("--final-improvement-ratio", type=float, default=.995)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--disable-screen-tto", dest="enable_screen_tto", action="store_false")
    parser.set_defaults(enable_screen_tto=True)
    parser.add_argument("--screen-pixel-schedule", nargs="+", type=float, default=[8., 5., 3.])
    parser.add_argument("--screen-steps", type=int, default=48)
    parser.add_argument("--screen-max-pairs", type=int, default=8000)
    parser.add_argument("--screen-max-rotation-deg", type=float, default=2.)
    parser.add_argument("--screen-min-scale", type=float, default=.985)
    parser.add_argument("--screen-max-scale", type=float, default=1.015)
    parser.add_argument("--screen-max-translation-ratio", type=float, default=.012)
    parser.add_argument("--screen-render-size", type=int, default=96)
    parser.add_argument("--screen-points", type=int, default=12000)
    parser.add_argument("--screen-weight", type=float, default=.8)
    parser.add_argument("--screen-trigger-objective", type=float, default=.10)
    parser.add_argument("--seed", type=int, default=6145)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = [process(args, str(sample)) for sample in args.samples]
    fields = ["sample_id", "selected_route", "accepted", "initial_objective", "final_objective", "elapsed_seconds"]
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({"sample_id": record["sample_id"], "selected_route": record["selected_route"], "accepted": record["accepted"], "initial_objective": record["initial_visible_score"]["objective"], "final_objective": record["final_visible_score"]["objective"], "elapsed_seconds": record["elapsed_seconds"]})


if __name__ == "__main__":
    main()
