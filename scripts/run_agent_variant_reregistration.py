#!/usr/bin/env python3
"""Re-register and GT-free-gate an arbitrary agent complete-prior proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.agent_completion_policy import accept_agent_proposal
from src.bidirectional_consensus_registration import bidirectional_consensus_step
from src.bidirectional_cycle_registration import partial_to_prior_inverse_step, sim3_parts, visible_score
from src.multiview_agent_feedback import make_orthographic_reference, measure_multiview_evidence


def run_refinement(prior, partial, projector, diagonal, pixel_schedule):
    current = np.asarray(prior, dtype=np.float64).copy()
    total = np.eye(4, dtype=np.float64)
    trace = []
    for radius in pixel_schedule:
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
        name, next_points, step, score = min(candidates, key=lambda candidate: candidate[3]["objective"])
        if name != "identity":
            current, total = next_points, step @ total
        trace.append({"pixel_radius": radius, "action": name,
                      "direct": direct_info, "consensus": consensus_info, "score": score})
    return current, total, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--proposal-mesh", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--multiview-anchor", type=Path, default=None,
                        help="Trusted registered anchor for a fixed three-view no-harm audit.")
    parser.add_argument("--multiview-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--action-name", default="prior_proposal",
        help="Auditable backend/action label; never affects registration or gating.")
    parser.add_argument("--pixel-schedule", nargs="+", type=float, default=[8., 5., 3.])
    parser.add_argument("--final-pixel-radius", type=float, default=5.)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    proposal, anchor, partial = (base.load_points(path) for path in (args.proposal, args.anchor, args.partial))
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=args.padding, image_shape=(512, 512), device="cpu")
    anchor_score = visible_score(partial, anchor, projector, diagonal, args.final_pixel_radius)
    initial_score = visible_score(partial, proposal, projector, diagonal, args.final_pixel_radius)
    registered, transform, trace = run_refinement(proposal, partial, projector, diagonal, args.pixel_schedule)
    final_score = visible_score(partial, registered, projector, diagonal, args.final_pixel_radius)
    multiview = None
    if args.multiview_anchor is not None:
        multiview_anchor = base.load_points(args.multiview_anchor)
        reference = make_orthographic_reference(partial, multiview_anchor, size=args.multiview_size)
        multiview = {
            "anchor": measure_multiview_evidence(partial, multiview_anchor, reference),
            "proposal": measure_multiview_evidence(partial, registered, reference),
        }
    accepted = accept_agent_proposal(
        anchor_score, final_score,
        anchor_multiview=None if multiview is None else multiview["anchor"],
        proposal_multiview=None if multiview is None else multiview["proposal"],
    )
    result = registered if accepted else anchor
    route = f"accepted_{args.action_name}" if accepted else "anchor_fallback"
    # Preserve the re-registered proposal even if the gate rejects it.  A
    # later bounded support-lock action may use it as a local model edit, but
    # cannot replace the anchor without running its own shared no-harm gate.
    base.write_points(args.output_dir / "proposal_refined_100k.ply", registered)
    base.write_compare(args.output_dir / "partial_gray_proposal_red.ply", partial, registered)
    base.write_points(args.output_dir / "registered_100k.ply", result)
    base.write_compare(args.output_dir / "partial_gray_prior_red.ply", partial, result)
    base.draw_projection_overlay(args.output_dir / "projection.png", args.semantic, partial, result, projector)
    mesh = trimesh.load(args.proposal_mesh, force="scene", process=False)
    mesh.apply_transform(transform)
    mesh.export(args.output_dir / "proposal_refined_mesh.glb")
    if accepted:
        mesh.export(args.output_dir / "registered_mesh.glb")
    scale, rotation, _ = sim3_parts(transform)
    record = {
        "method": "agent_proposal_reregistration",
        "action_name": args.action_name,
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "route": route, "accepted": accepted,
        "anchor_score": anchor_score, "proposal_initial_score": initial_score,
        "proposal_final_score": final_score, "trace": trace,
        "multiview": None if multiview is None else {
            key: value.to_dict() for key, value in multiview.items()
        },
        "transform": transform, "scale": scale,
        "proper_rotation_determinant": float(np.linalg.det(rotation)),
        "outputs": {"points": str(args.output_dir / "registered_100k.ply"),
                    "proposal_refined": str(args.output_dir / "proposal_refined_100k.ply"),
                    "proposal_refined_mesh": str(args.output_dir / "proposal_refined_mesh.glb")},
    }
    (args.output_dir / "reregistration_info.json").write_text(
        json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"route": route, "anchor": anchor_score["objective"],
                      "proposal_before": initial_score["objective"],
                      "proposal_after": final_score["objective"]}, indent=2))


if __name__ == "__main__":
    main()
