#!/usr/bin/env python3
"""Run hierarchical residual registration from the frozen Pixal-v15 body."""

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
from src.hierarchical_residual_registration import (
    apply_transform_mesh,
    intrinsic_local_step,
    load_scene_mesh,
)
from src.ray_consistent_registration import refine_camera_sim3, soft_ray_correspondences


REGISTRATION_ROOT = ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822"
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_hierarchical_residual_registration_multiscale_pilot_20260823"
DEFAULT_SAMPLES = ("09639", "07136")


def _paths(root: Path, sample: str):
    stem = root / sample / f"{sample}_unified_registration_v14"
    return Path(f"{stem}_registered_100k.ply"), Path(f"{stem}_registered_mesh.glb")


def _passes_guard(score: dict, initial_score: dict) -> bool:
    return bool(
        np.isfinite(score["objective"])
        and score["objective"] <= .99 * initial_score["objective"]
        and score["grid_coverage"] >= initial_score["grid_coverage"] - .02
        and score["matched_coverage"] >= initial_score["matched_coverage"] - .02)


def _global_only_candidate(args, body, mesh, partial, projector, diagonal):
    candidate_body = np.asarray(body).copy()
    candidate_mesh = mesh.copy()
    transform = np.eye(4, dtype=np.float64)
    trace = []
    for iteration in range(int(args.outer_iterations)):
        refined, info = refine_camera_sim3(
            candidate_body, partial, projector, bbox_diagonal=diagonal,
            pixel_schedule=tuple(args.pixel_schedule),
            max_rotation_deg=args.max_global_rotation_deg,
            scale_bounds=(args.min_step_scale, args.max_step_scale),
            max_translation_ratio=args.max_global_translation_ratio,
            min_pairs=args.min_pairs)
        if info["accepted"]:
            delta = np.asarray(info["delta_transform"], dtype=np.float64)
            candidate_body = refined
            candidate_mesh = apply_transform_mesh(candidate_mesh, delta)
            transform = delta @ transform
        trace.append({"iteration": int(iteration), "global_sim3": info})
    score = soft_ray_correspondences(
        partial, candidate_body, projector, pixel_radius=5., trim_quantile=.75,
        max_distance_ratio=.14, bbox_diagonal=diagonal)
    return candidate_body, candidate_mesh, transform, score, trace


def process(args, sample: str):
    complete_path, mesh_path = _paths(args.registration_root, sample)
    body = base.load_points(complete_path)
    partial = base.load_points(ROOT / "data" / f"{sample}.ply")
    mesh = load_scene_mesh(mesh_path)
    initial = body.copy()
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera_root / sample / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")
    initial_score = soft_ray_correspondences(
        partial, initial, projector, pixel_radius=5., trim_quantile=.75,
        max_distance_ratio=.14, bbox_diagonal=diagonal)
    global_body, global_mesh, global_transform, global_score, global_trace = (
        _global_only_candidate(
            args, initial, mesh, partial, projector, diagonal))
    total_global = np.eye(4, dtype=np.float64)
    trace = []

    for iteration in range(int(args.outer_iterations)):
        globally_refined, global_info = refine_camera_sim3(
            body, partial, projector, bbox_diagonal=diagonal,
            pixel_schedule=tuple(args.pixel_schedule),
            max_rotation_deg=args.max_global_rotation_deg,
            scale_bounds=(args.min_step_scale, args.max_step_scale),
            max_translation_ratio=args.max_global_translation_ratio,
            min_pairs=args.min_pairs)
        if global_info["accepted"]:
            delta = np.asarray(global_info["delta_transform"], dtype=np.float64)
            body = globally_refined
            mesh = apply_transform_mesh(mesh, delta)
            total_global = delta @ total_global

        local_hypotheses = []
        for component_policy in args.local_component_policies:
            for minimum_handles in args.local_handle_hypotheses:
                candidate_mesh, candidate_body, candidate_info = intrinsic_local_step(
                    mesh, body, partial, projector, diagonal=diagonal,
                    seed=args.seed + iteration,
                    proxy_triangles=args.proxy_triangles,
                    correspondence_samples=args.correspondence_samples,
                    output_samples=args.output_points,
                    max_handles=args.max_handles,
                    min_handles=minimum_handles,
                    active_inner_ratio=args.active_inner_ratio,
                    active_outer_ratio=args.active_outer_ratio,
                    anchor_ratio=args.anchor_ratio,
                    max_handle_displacement_ratio=args.max_handle_displacement_ratio,
                    max_vertex_displacement_ratio=args.max_vertex_displacement_ratio,
                    component_policy=component_policy)
                candidate_info["minimum_handles_hypothesis"] = int(minimum_handles)
                local_hypotheses.append(
                    (candidate_info["after"]["objective"], candidate_mesh,
                     candidate_body, candidate_info))
        accepted_hypotheses = [
            item for item in local_hypotheses if item[3].get("accepted", False)]
        if accepted_hypotheses:
            _, mesh, body, local_info = min(
                accepted_hypotheses, key=lambda item: item[0])
        else:
            local_info = min(local_hypotheses, key=lambda item: item[0])[3]
        local_info["hypotheses"] = [{
            "minimum_handles": int(item[3]["minimum_handles_hypothesis"]),
            "component_policy": item[3].get("component_policy"),
            "accepted": bool(item[3].get("accepted", False)),
            "objective": float(item[0]),
            "reason": item[3].get("reason"),
        } for item in local_hypotheses]
        trace.append({
            "iteration": int(iteration),
            "global_sim3": global_info,
            "intrinsic_local": local_info,
        })

    final_score = soft_ray_correspondences(
        partial, body, projector, pixel_radius=5., trim_quantile=.75,
        max_distance_ratio=.14, bbox_diagonal=diagonal)
    hierarchical_body = body
    hierarchical_mesh = mesh
    hierarchical_transform = total_global
    hierarchical_score = final_score
    candidates = []
    if _passes_guard(global_score, initial_score):
        candidates.append((global_score["objective"], "global_residual_sim3",
                           global_body, global_mesh, global_transform, global_score))
    if _passes_guard(hierarchical_score, initial_score):
        candidates.append((hierarchical_score["objective"], "hierarchical_residual",
                           hierarchical_body, hierarchical_mesh,
                           hierarchical_transform, hierarchical_score))
    accepted = bool(candidates)
    if accepted:
        (_, selected_route, body, mesh,
         total_global, final_score) = min(candidates, key=lambda item: item[0])
    else:
        selected_route = "exact_v15_fallback"
        body = initial
        mesh = load_scene_mesh(mesh_path)
        final_score = initial_score
        total_global = np.eye(4, dtype=np.float64)

    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_hierarchical_residual"
    outputs = {
        "complete": Path(f"{stem}_complete_100k.ply"),
        "mesh": Path(f"{stem}_complete_mesh.glb"),
        "compare": Path(f"{stem}_partial_gray_complete_red.ply"),
        "projection": Path(f"{stem}_projection.png"),
        "global_transform": Path(f"{stem}_global.npy"),
        "info": Path(f"{stem}_info.json"),
    }
    base.write_points(outputs["complete"], body)
    base.write_compare(outputs["compare"], partial, body)
    mesh.export(outputs["mesh"])
    np.save(outputs["global_transform"], total_global)
    base.draw_projection_overlay(
        outputs["projection"], args.camera_root / sample / "img.png",
        partial, body, projector)
    singular = np.linalg.svd(total_global[:3, :3], compute_uv=False)
    record = {
        "sample_id": sample,
        "method": "v15_hierarchical_residual_sim3_intrinsic_graph_tto",
        "accepted": accepted,
        "selected_route": selected_route,
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "sample_specific_parameters": False,
        "category_specific_parameters": False,
        "v15_input_frozen": True,
        "initial_visible_score": initial_score,
        "final_visible_score": final_score,
        "global_delta_transform": total_global,
        "global_delta_scale": float(singular.mean()),
        "global_delta_anisotropy": float(singular.max() - singular.min()),
        "outer_trace": trace,
        "global_only_trace": global_trace,
        "route_candidates": {
            "global_residual_sim3": {
                "passes_guard": _passes_guard(global_score, initial_score),
                "score": global_score,
            },
            "hierarchical_residual": {
                "passes_guard": _passes_guard(hierarchical_score, initial_score),
                "score": hierarchical_score,
            },
        },
        "contract": {
            "input_complete_points": int(len(initial)),
            "output_complete_points": int(len(body)),
            "points_deleted": 0,
            "proper_isotropic_global_sim3": bool(
                np.linalg.det(total_global[:3, :3]) > 0.
                and singular.max() - singular.min() < 1e-7),
            "local_similarity_nullspace_projection": True,
            "fallback_is_exact_v15": selected_route == "exact_v15_fallback",
        },
        "shared_parameters": {
            key: value for key, value in vars(args).items()
            if key not in {"samples"}
        },
        "outputs": outputs,
    }
    outputs["info"].write_text(
        json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(
        sample, "accepted", accepted,
        "route", selected_route,
        "objective", initial_score["objective"], "->", final_score["objective"],
        "global_scale", record["global_delta_scale"], flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration-root", type=Path, default=REGISTRATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--pixel-schedule", nargs="+", type=float, default=[8., 5., 3.])
    parser.add_argument("--max-global-rotation-deg", type=float, default=2.0)
    parser.add_argument("--min-step-scale", type=float, default=.97)
    parser.add_argument("--max-step-scale", type=float, default=1.03)
    parser.add_argument("--max-global-translation-ratio", type=float, default=.02)
    parser.add_argument("--min-pairs", type=int, default=96)
    parser.add_argument("--proxy-triangles", type=int, default=12000)
    parser.add_argument("--correspondence-samples", type=int, default=50000)
    parser.add_argument("--output-points", type=int, default=100000)
    parser.add_argument("--max-handles", type=int, default=96)
    parser.add_argument("--local-handle-hypotheses", nargs="+", type=int,
                        default=[16, 6])
    parser.add_argument("--local-component-policies", nargs="+",
                        choices=["residual_priority", "line_priority"],
                        default=["residual_priority", "line_priority"])
    parser.add_argument("--active-inner-ratio", type=float, default=.055)
    parser.add_argument("--active-outer-ratio", type=float, default=.125)
    parser.add_argument("--anchor-ratio", type=float, default=.145)
    parser.add_argument("--max-handle-displacement-ratio", type=float, default=.055)
    parser.add_argument("--max-vertex-displacement-ratio", type=float, default=.045)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=6145)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [process(args, str(sample)) for sample in args.samples]
    fields = [
        "sample_id", "accepted", "initial_objective", "final_objective",
        "initial_grid_coverage", "final_grid_coverage", "global_delta_scale",
        "selected_route"]
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "sample_id": result["sample_id"],
                "accepted": result["accepted"],
                "initial_objective": result["initial_visible_score"]["objective"],
                "final_objective": result["final_visible_score"]["objective"],
                "initial_grid_coverage": result["initial_visible_score"]["grid_coverage"],
                "final_grid_coverage": result["final_visible_score"]["grid_coverage"],
                "global_delta_scale": result["global_delta_scale"],
                "selected_route": result["selected_route"],
            })


if __name__ == "__main__":
    main()
