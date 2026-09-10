#!/usr/bin/env python3
"""Closed-loop two-camera TTO after analytic Pixal--MoGe registration.

This script consumes existing first-stage native Pixal/MoGe assets and an
existing Camera-1/Camera-2 pixel bridge.  It writes a registration diagnostic
only: no fusion, no GT metrics, and no proposal-gated fallback.
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

from src.bidirectional_cycle_registration import invert_proper_sim3, visible_score
from src.moge_camera import MoGeProjector
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.pixal_moge_analytic_registration import pixal_moge_render_score
from src.ray_consistent_registration import apply_transform
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from src.two_camera_joint_refinement import (
    bridge_match_score,
    compose_two_camera_transform,
    joint_two_camera_refine,
)
from src.visible_pixel_sim3_refinement import (
    local_camera1_visible_refine,
    pixel_pair_residual_candidates,
)


def _subset(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= int(count):
        return points
    return points[np.linspace(0, len(points) - 1, int(count), dtype=np.int64)]


def _compact_visible(score: dict) -> dict:
    return {
        "objective": float(score["objective"]),
        "geometric_objective": float(score["geometric"]["objective"]),
        "pair_count": int(len(score["geometric"]["partial_ids"])),
        "projection": {key: float(value) for key, value in score["projection"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--pixal-prior", type=Path, required=True)
    parser.add_argument("--native-moge", type=Path, required=True)
    parser.add_argument("--native-info", type=Path, required=True)
    parser.add_argument("--bridge-transform", type=Path, required=True)
    parser.add_argument("--pixel-matches", type=Path, required=True)
    parser.add_argument("--partial-camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-points", type=int, default=32_000)
    parser.add_argument("--pixel-pair-max-points", type=int, default=10_000,
                        help="Maximum mutual Camera-1 visible pairs for the final MoGe-style 3-D residual.")
    parser.add_argument("--pixel-pair-trials", type=int, default=64)
    parser.add_argument("--camera1-fine-search-points", type=int, default=32_000)
    parser.add_argument("--candidate-workers", type=int, default=8,
                        help="Independent visible-score evaluations per local-search level.")
    parser.add_argument("--coarse-basin-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the fixed shared broad Camera-1 pixel-Sim(3) capture before residual refinement.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    partial = load_points(args.partial)
    prior = load_points(args.pixal_prior)
    moge = load_points(args.native_moge)
    matches = np.load(args.pixel_matches)
    info = json.loads(args.native_info.read_text(encoding="utf-8"))
    q_to_m = np.asarray(info["prior_to_native_moge"], dtype=np.float64)
    m_to_p = np.load(args.bridge_transform)
    native_projector = MoGeProjector(
        np.asarray(info["moge"]["output_keys"]["intrinsics"], dtype=np.float64),
        tuple(info["moge"]["image_hw"]), device=args.device,
    )
    search_native_projector = MoGeProjector(
        np.asarray(info["moge"]["output_keys"]["intrinsics"], dtype=np.float64),
        (256, 256), device=args.device,
    )
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    partial_projector = SavedCameraProjector.from_partial(
        partial, args.partial_camera, padding=.15, image_shape=(512, 512), device=args.device
    )
    prior_native = apply_transform(prior, q_to_m)
    # Search uses fixed deterministic samples solely for speed. The final
    # accept/reject evidence below is recomputed on full point sets.
    delta_q, delta_m, search = joint_two_camera_refine(
        _subset(prior_native, args.search_points), _subset(moge, args.search_points),
        _subset(partial, args.search_points), matches, search_native_projector, partial_projector,
        m_to_p, partial_diagonal=diagonal, bridge_moge=moge, bridge_partial=partial,
    )
    bridge_total = m_to_p
    total = compose_two_camera_transform(q_to_m, m_to_p, delta_q, delta_m)
    baseline = apply_transform(prior, compose_two_camera_transform(q_to_m, m_to_p))
    proposed = apply_transform(prior, total)
    native_before = pixal_moge_render_score(moge, prior_native, native_projector)
    native_after = pixal_moge_render_score(moge, apply_transform(prior_native, delta_q), native_projector)
    bridge_before = bridge_match_score(moge, partial, matches, bridge_total, diagonal=diagonal)
    bridge_after = bridge_match_score(moge, partial, matches, delta_m @ bridge_total, diagonal=diagonal)
    partial_before = visible_score(partial, baseline, partial_projector, diagonal, pixel_radius=5.)
    partial_after = visible_score(partial, proposed, partial_projector, diagonal, pixel_radius=5.)
    # The two-camera residual is the fixed initialization for the direct
    # Camera-1 solve.  Scores are diagnostics, never rejection gates.
    active_total = total
    active_native = apply_transform(prior_native, delta_q)
    active_bridge = delta_m @ bridge_total
    active_partial = proposed

    # A sparse scan or a large camera-chain discrepancy can leave the strict
    # 14%-diagonal visible-3D matcher without any pairs. In that case the
    # narrow residual lattice has no geometric basin to refine. This fixed
    # capture stage uses the same Camera-1 pixel-indexed surface pairs as the
    # ordinary residual, but exposes a single shared broad Sim(3) lattice.
    # Identity remains a candidate; no category or sample-dependent decision
    # is introduced. Once a finite 3-D score exists, the existing narrow
    # residual and continuations take over unchanged. This is a fixed mainline
    # stage; --no-coarse-basin-recovery is retained only for ablation.
    coarse_record = {
        "enabled": bool(args.coarse_basin_recovery), "applied": False,
        "selection": "disabled", "selected_action": "disabled", "search": None, "trials": [],
        "before": _compact_visible(visible_score(
            partial, active_partial, partial_projector, diagonal, pixel_radius=5.
        )),
        "after": None,
    }
    if args.coarse_basin_recovery:
        try:
            coarse_candidates, coarse_search = pixel_pair_residual_candidates(
                partial, active_partial, partial_projector, diagonal=diagonal,
                max_pairs=args.pixel_pair_max_points, trials=args.pixel_pair_trials,
                fractions=(.125, .25, .50, .75, 1.0),
                max_rotation_deg=30., scale_bounds=(.45, 2.40), max_translation_ratio=1.25,
            )
        except ValueError as exc:
            # A sparse partial may provide no reliable Camera-1 surface pairs.
            # The fixed, evidence-free action is identity; do not manufacture a
            # residual or abort the rest of the two-camera route.
            coarse_record.update({
                "enabled": True, "selection": "identity_without_visible_Camera1_pairs",
                "selected_action": "identity", "search": {"available": False, "reason": str(exc)},
                "trials": [], "after": coarse_record["before"],
            })
        else:
            coarse_trials = []
            for action, partial_residual in coarse_candidates:
                candidate = apply_transform(active_partial, partial_residual)
                native_residual = invert_proper_sim3(active_bridge) @ partial_residual @ active_bridge
                coarse_trials.append({
                    "action": action, "partial_residual": partial_residual,
                    "native_residual": native_residual,
                    "partial": _compact_visible(visible_score(
                        partial, candidate, partial_projector, diagonal, pixel_radius=5.
                    )),
                })
            selected_coarse = min(coarse_trials, key=lambda item: item["partial"]["objective"])
            active_partial = apply_transform(active_partial, selected_coarse["partial_residual"])
            active_native = apply_transform(active_native, selected_coarse["native_residual"])
            active_total = selected_coarse["partial_residual"] @ active_total
            coarse_record = {
                "enabled": True,
                "applied": bool(selected_coarse["action"] != "identity"),
                "selection": "minimum_Camera1_visible_2D3D_objective_over_shared_broad_pixel_Sim3_lattice",
                "selected_action": selected_coarse["action"],
                "search": coarse_search, "trials": coarse_trials,
                "before": coarse_record["before"], "after": selected_coarse["partial"],
            }

    # MoGe's successful correction is a robust 3-D fit on pixel-indexed
    # visible surfaces, not a silhouette-only shift.  Apply that exact
    # principle to Pixal/partial in Camera-1.  Camera-2 evidence is always
    # recorded after conjugation as a diagnostic term; the two views have
    # different monocular gauges and neither creates a fallback branch.
    try:
        pixel_candidates, pixel_search = pixel_pair_residual_candidates(
            partial, active_partial, partial_projector, diagonal=diagonal,
            max_pairs=args.pixel_pair_max_points, trials=args.pixel_pair_trials,
        )
    except ValueError as exc:
        pixel_candidates = [("identity", np.eye(4, dtype=np.float64))]
        pixel_search = {"available": False, "reason": str(exc)}
    pixel_trials = []
    for action, partial_residual in pixel_candidates:
        candidate = apply_transform(active_partial, partial_residual)
        native_residual = invert_proper_sim3(active_bridge) @ partial_residual @ active_bridge
        native_candidate = pixal_moge_render_score(
            moge, apply_transform(active_native, native_residual), native_projector
        )
        partial_candidate = visible_score(partial, candidate, partial_projector, diagonal, pixel_radius=5.)
        pixel_trials.append({
            "action": action, "partial_residual": partial_residual,
            "native_residual": native_residual, "native": native_candidate,
            "partial": _compact_visible(partial_candidate),
        })
    selected_pixel = min(pixel_trials, key=lambda item: item["partial"]["objective"])
    pixel_applied = selected_pixel["action"] != "identity"
    result = apply_transform(active_partial, selected_pixel["partial_residual"])
    final_total = selected_pixel["partial_residual"] @ active_total
    final_partial = visible_score(partial, result, partial_projector, diagonal, pixel_radius=5.)
    final_native = selected_pixel["native"]
    fine_before = final_partial
    fine_step, fine_search = local_camera1_visible_refine(
        partial, result, partial_projector, diagonal=diagonal,
        search_points=args.camera1_fine_search_points, candidate_workers=args.candidate_workers,
    )
    fine_candidate = apply_transform(result, fine_step)
    fine_partial = visible_score(partial, fine_candidate, partial_projector, diagonal, pixel_radius=5.)
    # This coordinate descent contains identity in every fixed local lattice.
    # Apply its selected Sim(3) directly; no full-resolution fallback exists.
    result = fine_candidate
    final_total = fine_step @ final_total
    final_partial = fine_partial
    selection_reason = "fixed_two_camera_pixel_sim3_camera1_continuation"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "two_camera_joint"
    np.save(Path(f"{stem}_pixal_native_residual.npy"), delta_q)
    np.save(Path(f"{stem}_moge_partial_residual.npy"), delta_m)
    np.save(Path(f"{stem}_pixal_to_partial.npy"), final_total)
    np.save(Path(f"{stem}_visible_pixel_residual.npy"), selected_pixel["partial_residual"])
    np.save(Path(f"{stem}_camera1_fine_residual.npy"), fine_step)
    write_points(Path(f"{stem}_bridge_registered_100k.ply"), baseline)
    write_points(Path(f"{stem}_registered_100k.ply"), result)
    write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, result)
    draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, result, partial_projector)
    record = {
        "method": "two_camera_closed_loop_plus_moge_style_visible_pixel_sim3", "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False, "fusion_run": False,
        "fixed_route_applied": True,
        "proposal_gates_used": False,
        "reason": selection_reason,
        "inputs": {"partial": str(args.partial.resolve()), "pixal_prior": str(args.pixal_prior.resolve()),
                   "native_moge": str(args.native_moge.resolve()), "native_info": str(args.native_info.resolve()),
                   "bridge_transform": str(args.bridge_transform.resolve()), "pixel_matches": str(args.pixel_matches.resolve())},
        "search": search, "full_evidence": {
            "native_before": native_before, "native_after": native_after,
            "bridge_before": bridge_before, "bridge_after": bridge_after,
            "partial_before": _compact_visible(partial_before), "partial_after": _compact_visible(partial_after),
            "pixel_residual_final_native": final_native,
            "pixel_residual_final_partial": _compact_visible(final_partial),
        },
        "residuals": {"pixal_native_moge": delta_q, "moge_partial": delta_m,
                      "visible_pixel_partial": selected_pixel["partial_residual"]},
        "coarse_basin_recovery": coarse_record,
        "visible_pixel_direct_refinement": {
            "applied": bool(pixel_applied),
            "selection": "minimum_Camera1_visible_2D3D_objective_over_fixed_residual_fractions",
            "search": pixel_search, "trials": pixel_trials,
        },
        "camera1_fine_refinement": {
            "applied": True, "search": fine_search,
            "full_before": _compact_visible(fine_before),
            "full_after": _compact_visible(final_partial),
        },
        "all_pixal_points_preserved": bool(len(result) == len(prior)),
    }
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"joint_applied": True, "pixel_applied": pixel_applied,
                      "partial_before": partial_before["objective"],
                      "partial_after": final_partial["objective"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
