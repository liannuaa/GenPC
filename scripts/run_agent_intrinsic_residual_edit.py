#!/usr/bin/env python3
"""Run one guarded topology-preserving residual action for an agent prior.

This executor is deliberately narrow: a controller may choose it only after a
saved-view residual diagnosis.  It moves the *existing* complete-prior mesh in
one compact observed geodesic region; it never decodes a new global mesh,
deletes hidden geometry, or uses ground truth.  Common saved-view and fixed
PCA-view gates decide whether its result can replace the input prior.
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
from src.agent_completion_policy import accept_agent_proposal
from src.bidirectional_cycle_registration import visible_score
from src.hierarchical_residual_registration import intrinsic_local_step, load_scene_mesh
from src.multiview_agent_feedback import make_orthographic_reference, measure_multiview_evidence
from src.saved_view_text_feedback import build_saved_view_text_feedback


def _concise(score: dict) -> dict:
    return {
        "objective": float(score["objective"]),
        "geometric_objective": float(score["geometric"]["objective"]),
        "iou": float(score["projection"]["iou"]),
        "coverage": float(score["projection"]["coverage"]),
        "leakage": float(score["projection"]["leakage"]),
    }


def process(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor = base.load_points(args.anchor)
    partial = base.load_points(args.partial)
    guidance = (base.load_points(args.guidance_proposal)
                if args.guidance_proposal is not None else None)
    mesh = load_scene_mesh(args.anchor_mesh)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=args.padding, image_shape=(512, 512), device="cpu"
    )
    feedback = build_saved_view_text_feedback(partial, anchor, projector, diagonal=diagonal)
    (args.output_dir / "agent_instruction.txt").write_text(feedback.prompt + "\n", encoding="utf-8")

    anchor_score = visible_score(partial, anchor, projector, diagonal, pixel_radius=5.)
    reference = make_orthographic_reference(partial, anchor)
    anchor_multiview = measure_multiview_evidence(partial, anchor, reference)
    edited_mesh, edited, edit_info = intrinsic_local_step(
        mesh, anchor, partial, projector, diagonal=diagonal, seed=args.seed,
        proxy_triangles=args.proxy_triangles,
        correspondence_samples=args.correspondence_samples,
        output_samples=args.output_points, max_handles=args.max_handles,
        active_inner_ratio=args.active_inner_ratio,
        active_outer_ratio=args.active_outer_ratio, anchor_ratio=args.anchor_ratio,
        max_handle_displacement_ratio=args.max_handle_displacement_ratio,
        max_vertex_displacement_ratio=args.max_vertex_displacement_ratio,
        max_flipped_face_ratio=args.max_flipped_face_ratio,
        component_policy=args.component_policy, guidance_body=guidance,
        guidance_max_transfer_ratio=args.guidance_max_transfer_ratio,
    )
    edited_score = visible_score(partial, edited, projector, diagonal, pixel_radius=5.)
    edited_multiview = measure_multiview_evidence(partial, edited, reference)
    accepted = bool(edit_info.get("accepted", False) and accept_agent_proposal(
        anchor_score, edited_score, anchor_multiview=anchor_multiview,
        proposal_multiview=edited_multiview,
    ))
    result = edited if accepted else anchor
    result_mesh = edited_mesh if accepted else mesh
    route = "accepted_intrinsic_residual_edit" if accepted else "anchor_fallback"
    outputs = {
        "points": args.output_dir / "registered_100k.ply",
        "mesh": args.output_dir / "registered_mesh.glb",
        "compare": args.output_dir / "partial_gray_prior_red.ply",
        "projection": args.output_dir / "projection.png",
        "record": args.output_dir / "intrinsic_residual_action.json",
    }
    base.write_points(outputs["points"], result)
    result_mesh.export(outputs["mesh"])
    base.write_compare(outputs["compare"], partial, result)
    base.draw_projection_overlay(outputs["projection"], args.semantic, partial, result, projector)
    record = {
        "method": "agent_intrinsic_residual_mesh_edit",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "input": {"anchor": str(args.anchor), "anchor_mesh": str(args.anchor_mesh),
                  "guidance_proposal": (str(args.guidance_proposal)
                                       if args.guidance_proposal is not None else None),
                  "partial": str(args.partial), "camera": str(args.camera)},
        "agent_action": feedback.to_dict(),
        "nonrigid_deformation_used": True,
        "topology_preserved": True,
        "hidden_geometry_deleted": False,
        "anchor": _concise(anchor_score),
        "proposal": _concise(edited_score),
        "anchor_multiview": anchor_multiview.to_dict(),
        "proposal_multiview": edited_multiview.to_dict(),
        "intrinsic_solver": edit_info,
        "accepted": accepted,
        "route": route,
        "shared_parameters": {
            key: value for key, value in vars(args).items()
            if key not in {"anchor", "anchor_mesh", "partial", "camera", "semantic", "output_dir"}
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["record"].write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"route": route, "anchor": _concise(anchor_score),
                      "proposal": _concise(edited_score),
                      "action_eligible": feedback.eligible}, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-mesh", type=Path, required=True)
    parser.add_argument("--guidance-proposal", type=Path,
                        help="Generated/re-registered body used only for local residual vectors.")
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--proxy-triangles", type=int, default=12000)
    parser.add_argument("--correspondence-samples", type=int, default=50000)
    parser.add_argument("--output-points", type=int, default=100000)
    parser.add_argument("--max-handles", type=int, default=96)
    parser.add_argument("--active-inner-ratio", type=float, default=.055)
    parser.add_argument("--active-outer-ratio", type=float, default=.125)
    parser.add_argument("--anchor-ratio", type=float, default=.145)
    parser.add_argument("--max-handle-displacement-ratio", type=float, default=.055)
    parser.add_argument("--max-vertex-displacement-ratio", type=float, default=.045)
    parser.add_argument("--max-flipped-face-ratio", type=float, default=8e-4,
                        help="Shared local-edit safety limit; topology and multi-view gates still apply.")
    parser.add_argument("--guidance-max-transfer-ratio", type=float, default=.025)
    parser.add_argument("--component-policy", choices=("residual_priority", "line_priority"),
                        default="residual_priority")
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
