#!/usr/bin/env python3
"""Run a larger but still global Camera-1 residual Sim(3) refinement.

This recovery runner needs only the hard partial, an already registered Pixal
prior, and the saved Redwood camera.  It is therefore usable when optional
two-camera scratch artifacts have been cleaned while preserving a completed
registration candidate.  It never regenerates a prior, edits geometry, or
uses GT/CD/EMD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from scripts.run_registration_deformation_fusion_ablation import SavedCameraProjector
from src.bidirectional_cycle_registration import visible_score
from src.ray_consistent_registration import apply_transform
from src.visible_pixel_sim3_refinement import local_camera1_visible_refine


def _compact(score: dict) -> dict:
    return {
        "objective": float(score["objective"]),
        "geometric_objective": float(score["geometric"]["objective"]),
        "pair_count": int(len(score["geometric"]["partial_ids"])),
        "projection": {key: float(value) for key, value in score["projection"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--registered-prior", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-points", type=int, default=32_000)
    parser.add_argument("--wide-tilt-search", action=argparse.BooleanOptionalAction, default=False,
                        help="Use the shared 1-degree global tilt trust region before fine steps.")
    parser.add_argument("--max-tilt-degrees", type=float, default=1.0,
                        help="Initial rotation cap when --wide-tilt-search is enabled.")
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    partial = base.load_points(args.partial)
    prior = base.load_points(args.registered_prior)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=args.padding, image_shape=(512, 512), device=args.device,
    )
    if args.max_tilt_degrees <= 0.:
        raise ValueError("--max-tilt-degrees must be positive")
    tilt = float(args.max_tilt_degrees)
    levels = ((.010, tilt, .010), (.004, .35 * tilt, .004), (.001, .10 * tilt, .001)) if args.wide_tilt_search else (
        (.006, .30, .006), (.002, .10, .002), (.0005, .025, .0005)
    )
    before = visible_score(partial, prior, projector, diagonal, pixel_radius=5.)
    step, search = local_camera1_visible_refine(
        partial, prior, projector, diagonal=diagonal, search_points=args.search_points,
        levels=levels,
    )
    candidate = apply_transform(prior, step)
    after = visible_score(partial, candidate, projector, diagonal, pixel_radius=5.)
    # The fixed coordinate-descent lattice contains identity, so this is an
    # optimization step rather than a proposal that can be rejected.
    result = candidate
    final = after

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "camera1_amplified"
    base.write_points(Path(f"{stem}_registered_100k.ply"), result)
    base.write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, result)
    base.draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, result, projector)
    np.save(Path(f"{stem}_residual.npy"), step)
    record = {
        "method": "camera1_amplified_global_proper_sim3_recovery",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "nonrigid_deformation_used": False, "fusion_run": False,
        "applied": True,
        "proposal_gates_used": False,
        "objective_improved": bool(after["objective"] < before["objective"]),
        "inputs": {"partial": str(args.partial.resolve()), "registered_prior": str(args.registered_prior.resolve()),
                   "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve())},
        "wide_tilt_search": bool(args.wide_tilt_search),
        "max_tilt_degrees": float(args.max_tilt_degrees),
        "levels": [{"scale_delta": float(scale), "rotation_deg": float(degrees),
                    "translation_ratio": float(translation)} for scale, degrees, translation in levels],
        "search": search, "before": _compact(before), "after": _compact(after), "final": _compact(final),
        "all_prior_points_preserved": bool(len(result) == len(prior)),
    }
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"applied": True, "before": before["objective"], "after": final["objective"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
