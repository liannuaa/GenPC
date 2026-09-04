#!/usr/bin/env python3
"""Run the requested MoGe 2D+3D registration followed by hard partial TTO."""

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
from scripts.run_agent_moge_bootstrap_registration import partial_only_refine
from src.agent_completion_policy import accept_registration_refinement
from src.bidirectional_cycle_registration import visible_score
from src.moge_view_registration import MoGeProjector, register_prior_to_moge_2d3d
from src.ray_consistent_registration import apply_transform
from src.saved_view_tto import optimize_saved_view_sim3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--moge", type=Path, required=True, help="Native MoGe object PLY")
    parser.add_argument("--moge-info", type=Path, required=True)
    parser.add_argument("--moge-to-partial", type=Path, required=True,
                        help="NPZ from the calibrated soft-observation builder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prior, partial, moge = base.load_points(args.prior), base.load_points(args.partial), base.load_points(args.moge)
    moge_meta = json.loads(args.moge_info.read_text())
    intrinsics = np.asarray(moge_meta["moge"]["output_keys"]["intrinsics"], dtype=np.float64)
    image_shape = tuple(moge_meta["moge"]["image_hw"])
    moge_projector = MoGeProjector(intrinsics, image_shape, device=args.device)
    prior_moge, transform_prior_to_moge, moge_trace = register_prior_to_moge_2d3d(prior, moge, moge_projector)
    bridge = np.load(args.moge_to_partial)
    transform_moge_to_partial = np.asarray(bridge["transform"], dtype=np.float64)
    coarse = apply_transform(prior_moge, transform_moge_to_partial)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    partial_projector = base.SavedCameraProjector.from_partial(partial, args.camera, padding=.15,
                                                                image_shape=(512, 512), device=args.device)
    coarse_score = visible_score(partial, coarse, partial_projector, diagonal, pixel_radius=5.)
    refined, partial_step, partial_trace = partial_only_refine(coarse, partial, partial_projector, diagonal=diagonal)
    refined_score = visible_score(partial, refined, partial_projector, diagonal, pixel_radius=5.)
    screen, screen_step, screen_trace = optimize_saved_view_sim3(
        refined, partial, partial_projector, diagonal=diagonal, pixel_schedule=(8., 5., 3.), steps=48,
        max_pairs=8000, max_rotation_deg=2., scale_bounds=(.985, 1.015), max_translation_ratio=.012,
        seed=6145, device=args.device, screen_size=96, screen_points=12000, screen_weight=.8)
    screen_score = visible_score(partial, screen, partial_projector, diagonal, pixel_radius=5.)
    use_screen = accept_registration_refinement(refined_score, screen_score)
    result = screen if use_screen else refined
    final_score = screen_score if use_screen else refined_score
    total = (screen_step if use_screen else np.eye(4)) @ partial_step @ transform_moge_to_partial @ transform_prior_to_moge
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = args.partial.stem; stem = args.output_dir / f"{sample}_moge_then_partial"
    base.write_points(Path(f"{stem}_moge_registered_100k.ply"), prior_moge)
    base.write_compare(Path(f"{stem}_moge_gray_pixal_red.ply"), moge, prior_moge)
    base.write_points(Path(f"{stem}_registered_100k.ply"), result)
    base.write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, result)
    base.draw_projection_overlay(Path(f"{stem}_projection.png"), args.semantic, partial, result, partial_projector)
    mesh = trimesh.load(args.mesh, force="scene", process=False); mesh.apply_transform(total)
    mesh.export(Path(f"{stem}_registered_mesh.glb")); np.save(Path(f"{stem}.npy"), total)
    record = {
        "method": "moge_2d3d_bidirectional_proper_sim3_then_partial_2d3d_bidirectional_proper_sim3",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "stage_one_moge": moge_trace, "moge_to_partial_transform": transform_moge_to_partial,
        "stage_two_hard_partial": partial_trace, "coarse_hard_partial_score": coarse_score,
        "screen_tto": {"accepted": use_screen, "trace": screen_trace}, "final_hard_partial_score": final_score,
        "inputs": {"prior": str(args.prior.resolve()), "moge": str(args.moge.resolve()),
                   "partial": str(args.partial.resolve()), "moge_to_partial": str(args.moge_to_partial.resolve())},
    }
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"moge_before": moge_trace["before"]["objective"],
                      "moge_after": moge_trace["after"]["objective"],
                      "coarse_partial": coarse_score["objective"],
                      "final_partial": final_score["objective"]}, indent=2))


if __name__ == "__main__":
    main()
