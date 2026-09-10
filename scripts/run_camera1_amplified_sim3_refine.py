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

from src.bidirectional_cycle_registration import prepare_visible_target, visible_score
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.ray_consistent_registration import apply_transform
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
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
    parser.add_argument("--candidate-workers", type=int, default=8,
                        help="Independent visible-score evaluations per local-search level.")
    parser.add_argument("--wide-tilt-search", action=argparse.BooleanOptionalAction, default=False,
                        help="Use the shared 1-degree global tilt trust region before fine steps.")
    parser.add_argument("--max-tilt-degrees", type=float, default=1.0,
                        help="Initial rotation cap when --wide-tilt-search is enabled.")
    parser.add_argument(
        "--levels-json", type=Path,
        help=(
            "Optional shared JSON list of [scale_delta, rotation_degrees, "
            "translation_ratio] levels. This exposes the existing coordinate "
            "search trust region without changing its objective."
        ),
    )
    parser.add_argument(
        "--monotonic", action="store_true",
        help="Keep the identity update when the full no-GT visible objective does not improve.",
    )
    parser.add_argument(
        "--max-passes", type=int, default=1,
        help="Maximum shared broad-to-narrow continuation passes.",
    )
    parser.add_argument(
        "--minimum-relative-improvement", type=float, default=0.0,
        help=(
            "Stop after an accepted pass whose relative no-GT objective improvement "
            "falls below this value. Zero preserves the historical single-pass behavior."
        ),
    )
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_passes < 1:
        raise ValueError("--max-passes must be positive")
    if not 0.0 <= args.minimum_relative_improvement < 1.0:
        raise ValueError("--minimum-relative-improvement must be in [0, 1)")

    partial = load_points(args.partial)
    prior = load_points(args.registered_prior)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=args.padding, image_shape=(512, 512), device=args.device,
    )
    if args.max_tilt_degrees <= 0.:
        raise ValueError("--max-tilt-degrees must be positive")
    tilt = float(args.max_tilt_degrees)
    if args.levels_json is not None:
        payload = json.loads(args.levels_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("--levels-json must contain a non-empty list")
        levels = tuple(tuple(float(value) for value in level) for level in payload)
        if any(len(level) != 3 or min(level) <= 0. for level in levels):
            raise ValueError("each search level must contain three positive values")
    else:
        levels = ((.010, tilt, .010), (.004, .35 * tilt, .004), (.001, .10 * tilt, .001)) if args.wide_tilt_search else (
            (.006, .30, .006), (.002, .10, .002), (.0005, .025, .0005)
        )
    target_cache = prepare_visible_target(partial, projector)
    before = visible_score(partial, prior, projector, diagonal, pixel_radius=5., target_cache=target_cache)
    result, final = prior, before
    total_step = np.eye(4, dtype=np.float64)
    pass_records = []
    for pass_index in range(int(args.max_passes)):
        pass_before = final
        step, search = local_camera1_visible_refine(
            partial, result, projector, diagonal=diagonal, search_points=args.search_points,
            candidate_workers=args.candidate_workers, levels=levels,
        )
        candidate = apply_transform(result, step)
        candidate_score = visible_score(
            partial, candidate, projector, diagonal, pixel_radius=5., target_cache=target_cache,
        )
        improved = candidate_score["objective"] < pass_before["objective"]
        accepted = not args.monotonic or improved
        relative_improvement = (
            (pass_before["objective"] - candidate_score["objective"])
            / max(pass_before["objective"], 1e-12)
            if accepted else 0.0
        )
        pass_records.append({
            "pass": pass_index + 1,
            "accepted": bool(accepted),
            "relative_improvement": float(relative_improvement),
            "before": _compact(pass_before),
            "candidate": _compact(candidate_score),
            "search": search,
        })
        if not accepted:
            break
        result, final = candidate, candidate_score
        total_step = step @ total_step
        if relative_improvement < float(args.minimum_relative_improvement):
            break
    after = final
    accepted = bool(np.max(np.abs(total_step - np.eye(4))) > 1e-12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "camera1_amplified"
    write_points(Path(f"{stem}_registered_100k.ply"), result)
    write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, result)
    draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, result, projector)
    np.save(Path(f"{stem}_residual.npy"), total_step)
    record = {
        "method": "camera1_amplified_global_proper_sim3_recovery",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "nonrigid_deformation_used": False, "fusion_run": False,
        "applied": bool(accepted),
        "proposal_gates_used": False,
        "objective_improved": bool(after["objective"] < before["objective"]),
        "inputs": {"partial": str(args.partial.resolve()), "registered_prior": str(args.registered_prior.resolve()),
                   "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve())},
        "wide_tilt_search": bool(args.wide_tilt_search),
        "max_tilt_degrees": float(args.max_tilt_degrees),
        "monotonic": bool(args.monotonic),
        "max_passes": int(args.max_passes),
        "minimum_relative_improvement": float(args.minimum_relative_improvement),
        "passes_run": len(pass_records),
        "levels": [{"scale_delta": float(scale), "rotation_deg": float(degrees),
                    "translation_ratio": float(translation)} for scale, degrees, translation in levels],
        "search": pass_records[-1]["search"],
        "continuation": pass_records,
        "before": _compact(before), "after": _compact(after), "final": _compact(final),
        "all_prior_points_preserved": bool(len(result) == len(prior)),
    }
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"applied": bool(accepted), "before": before["objective"], "after": final["objective"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
