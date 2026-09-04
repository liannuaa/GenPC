#!/usr/bin/env python3
"""MoGe coarse Sim(3) followed by partial-only bidirectional refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.agent_completion_policy import accept_registration_refinement
from src.bidirectional_consensus_registration import bidirectional_consensus_step
from src.bidirectional_cycle_registration import visible_score
from src.moge_bootstrap_registration import moge_coarse_proper_sim3
from src.ray_consistent_registration import apply_transform
from src.saved_view_tto import optimize_saved_view_sim3


def partial_only_refine(current: np.ndarray, partial: np.ndarray, projector, *, diagonal: float) -> tuple[np.ndarray, np.ndarray, list]:
    """Use the established hard-partial 2D+3D actions after MoGe bootstrap."""
    from src.bidirectional_cycle_registration import partial_to_prior_inverse_step

    total = np.eye(4, dtype=np.float64)
    trace = []
    for radius in (8., 5., 3.):
        direct, direct_step, direct_info = partial_to_prior_inverse_step(
            current, partial, projector, diagonal=diagonal, pixel_radius=radius,
            max_rotation_deg=3., scale_bounds=(.96, 1.04), max_translation_ratio=.03,
            min_pairs=96, return_best_candidate=False)
        consensus, consensus_step, consensus_info = bidirectional_consensus_step(
            current, partial, projector, diagonal=diagonal, pixel_radius=radius,
            max_rotation_deg=3., scale_bounds=(.96, 1.04), max_translation_ratio=.03,
            min_pairs=96, max_cycle_ratio=.03, return_best_candidate=False)
        candidates = [("identity", current, np.eye(4), visible_score(partial, current, projector, diagonal, radius))]
        if direct_info["accepted"]:
            candidates.append(("partial_to_prior_inverse", direct, direct_step, direct_info["after"]))
        if consensus_info["accepted"]:
            candidates.append(("bidirectional_consensus", consensus, consensus_step, consensus_info["after"]))
        name, selected, step, score = min(candidates, key=lambda item: item[3]["objective"])
        current, total = selected, step @ total
        trace.append({"pixel_radius": radius, "selected_action": name, "score": score,
                      "direct": direct_info, "consensus": consensus_info})
    return current, total, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True, help="Fresh unregistered Pixal 100k PLY")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--moge-soft-observation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coarse-only", action="store_true",
                        help="Export and inspect T_MoGe without partial-only refinement.")
    args = parser.parse_args()

    prior, partial = base.load_points(args.prior), base.load_points(args.partial)
    soft = np.load(args.moge_soft_observation)
    moge, confidence = soft["points"], soft["confidence"]
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(partial, args.camera, padding=.15,
                                                        image_shape=(512, 512), device=args.device)
    coarse, coarse_transform, bootstrap = moge_coarse_proper_sim3(prior, moge, confidence)
    coarse_partial_score = visible_score(partial, coarse, projector, diagonal, pixel_radius=5.)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = args.partial.stem; stem = args.output_dir / f"{sample}_moge_bootstrap"
    # This diagnostic deliberately compares only the MoGe coarse target and
    # the coarse Pixal prior, before any real-partial refinement is allowed.
    base.write_points(Path(f"{stem}_moge_coarse_100k.ply"), coarse)
    base.write_compare(Path(f"{stem}_moge_gray_pixal_red.ply"), moge, coarse)
    base.draw_projection_overlay(Path(f"{stem}_moge_projection.png"), args.semantic, moge, coarse, projector)
    if args.coarse_only:
        record = {"method": "moge_soft_visible_coarse_proper_sim3_diagnostic",
                  "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
                  "bootstrap": bootstrap, "coarse_transform": coarse_transform,
                  "coarse_partial_score_not_used_for_moge_selection": coarse_partial_score,
                  "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                             "moge_soft_observation": str(args.moge_soft_observation.resolve())}}
        Path(f"{stem}_moge_coarse_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
        print(json.dumps(base.jsonable({"moge_score": bootstrap["selected_score"],
                                        "hard_partial_score": coarse_partial_score["objective"]}), indent=2))
        return
    refined, refinement_transform, trace = partial_only_refine(coarse, partial, projector, diagonal=diagonal)
    refined_score = visible_score(partial, refined, projector, diagonal, pixel_radius=5.)
    # A final screen TTO is still partial-only.  It remains a candidate rather
    # than an unconditional update and shares the same hard-partial gate.
    screen, screen_step, screen_trace = optimize_saved_view_sim3(
        refined, partial, projector, diagonal=diagonal, pixel_schedule=(8., 5., 3.), steps=48,
        max_pairs=8000, max_rotation_deg=2., scale_bounds=(.985, 1.015), max_translation_ratio=.012,
        seed=6145, device=args.device, screen_size=96, screen_points=12000, screen_weight=.8)
    screen_score = visible_score(partial, screen, projector, diagonal, pixel_radius=5.)
    if accept_registration_refinement(refined_score, screen_score):
        result, screen_accepted = screen, True
        total_transform, final_score = screen_step @ refinement_transform @ coarse_transform, screen_score
    else:
        result, screen_accepted = refined, False
        total_transform, final_score = refinement_transform @ coarse_transform, refined_score
    base.write_points(Path(f"{stem}_registered_100k.ply"), result)
    base.write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, result)
    base.draw_projection_overlay(Path(f"{stem}_projection.png"), args.semantic, partial, result, projector)
    mesh = trimesh.load(args.mesh, force="scene", process=False); mesh.apply_transform(total_transform)
    mesh.export(Path(f"{stem}_registered_mesh.glb")); np.save(Path(f"{stem}.npy"), total_transform)
    record = {
        "method": "moge_soft_visible_coarse_proper_sim3_then_partial_only_bidirectional_refine",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "moge_role": "coarse initialization only; excluded from refinement and final acceptance",
        "bootstrap": bootstrap, "coarse_transform": coarse_transform, "coarse_partial_score": coarse_partial_score,
        "partial_only_refinement": trace, "screen_tto": {"accepted": screen_accepted, "trace": screen_trace},
        "final_partial_score": final_score, "inputs": {"prior": str(args.prior.resolve()),
            "partial": str(args.partial.resolve()), "moge_soft_observation": str(args.moge_soft_observation.resolve())},
    }
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"coarse_partial_objective": coarse_partial_score["objective"],
                      "final_partial_objective": final_score["objective"],
                      "screen_accepted": screen_accepted}, indent=2))


if __name__ == "__main__":
    main()
