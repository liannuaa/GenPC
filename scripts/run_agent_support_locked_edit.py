#!/usr/bin/env python3
"""Accept a 3-D editor proposal only inside partial-supported surface tubes.

The source prior remains the complete-shape authority outside the observed
scan.  A model-generated and re-registered proposal can replace it only in a
shared-radius tube around observed partial points.  Radius selection and final
acceptance use the same saved-view 2-D+3-D evidence as every other agent
action; no ground truth, category, or sample-specific setting is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.agent_completion_policy import accept_agent_proposal
from src.bidirectional_cycle_registration import visible_score
from src.multiview_agent_feedback import make_orthographic_reference, measure_multiview_evidence
from src.view_conditioned_surfel import weighted_voxel_decode


def _compact_points(points: np.ndarray, *, target_points: int, diagonal: float) -> tuple[np.ndarray, dict]:
    # Every centre comes from either the source prior or the editor proposal;
    # opacity is constant, so the deterministic voxel decoder is purely a
    # surface-measure resampler and never creates/deforms geometry.
    opacity = np.ones(len(points), dtype=np.float64)
    ids, info = weighted_voxel_decode(
        points, opacity, target_points=target_points, diagonal=diagonal
    )
    compact = points[ids]
    if len(compact) < target_points:
        fill = compact[np.arange(target_points - len(compact), dtype=np.int64) % len(compact)]
        compact = np.concatenate((compact, fill), axis=0)
        info["repeated_to_target"] = int(target_points - len(ids))
    else:
        info["repeated_to_target"] = 0
    return compact, info


def support_locked_proposal(anchor: np.ndarray, edited: np.ndarray, partial: np.ndarray,
                            *, radius_ratio: float, target_points: int) -> tuple[np.ndarray, dict]:
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    radius = float(radius_ratio) * diagonal
    tree = cKDTree(partial)
    anchor_distance = tree.query(anchor, k=1, workers=-1)[0]
    edited_distance = tree.query(edited, k=1, workers=-1)[0]
    # Replace rather than append within support: this avoids duplicated visible
    # sheets while retaining the source completion everywhere unobserved.
    retained = anchor[anchor_distance > radius]
    local_edit = edited[edited_distance <= radius]
    if len(local_edit) < 128 or len(retained) < 128:
        raise ValueError("support tube produced an invalid local/source split")
    compact, resampling = _compact_points(
        np.concatenate((retained, local_edit), axis=0), target_points=target_points,
        diagonal=diagonal,
    )
    return compact, {
        "radius_ratio": float(radius_ratio), "radius": radius,
        "anchor_replaced": int(len(anchor) - len(retained)),
        "edited_inserted": int(len(local_edit)), "input_points": int(len(retained) + len(local_edit)),
        "resampling": resampling,
    }


def concise(score: dict) -> dict:
    return {
        "objective": float(score["objective"]),
        "geometric_objective": float(score["geometric"]["objective"]),
        "iou": float(score["projection"]["iou"]),
        "coverage": float(score["projection"]["coverage"]),
        "leakage": float(score["projection"]["leakage"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--edited-proposal", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radii", nargs="+", type=float, default=(.015, .025, .035, .05))
    parser.add_argument("--target-points", type=int, default=100000)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    if any(radius <= 0. for radius in args.radii):
        raise ValueError("all support radii must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor, edited, partial = (base.load_points(path) for path in
                               (args.anchor, args.edited_proposal, args.partial))
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=args.padding, image_shape=(512, 512), device="cpu"
    )
    anchor_score = visible_score(partial, anchor, projector, diagonal, pixel_radius=5.)
    multiview_reference = make_orthographic_reference(partial, anchor)
    anchor_multiview = measure_multiview_evidence(partial, anchor, multiview_reference)
    candidates = []
    for radius in args.radii:
        proposal, support = support_locked_proposal(
            anchor, edited, partial, radius_ratio=radius, target_points=args.target_points
        )
        score = visible_score(partial, proposal, projector, diagonal, pixel_radius=5.)
        multiview_score = measure_multiview_evidence(partial, proposal, multiview_reference)
        path = args.output_dir / f"candidate_radius_{radius:.3f}.ply"
        base.write_points(path, proposal)
        candidates.append({"radius_ratio": radius, "points": proposal, "score": score,
                           "multiview_score": multiview_score,
                           "support": support, "path": str(path)})
    # The controller first enforces no-harm for every shared-radius proposal,
    # then selects the lowest objective only among valid actions.  Selecting an
    # invalid lower-objective radius and subsequently falling back would throw
    # away an already-safe local edit.
    eligible = [candidate for candidate in candidates if accept_agent_proposal(
        anchor_score, candidate["score"], anchor_multiview=anchor_multiview,
        proposal_multiview=candidate["multiview_score"]
    )]
    selected = min(eligible or candidates, key=lambda candidate: candidate["score"]["objective"])
    accepted = bool(eligible)
    result = selected["points"] if accepted else anchor
    route = "accepted_support_locked_edit" if accepted else "anchor_fallback"
    base.write_points(args.output_dir / "registered_100k.ply", result)
    base.write_compare(args.output_dir / "partial_gray_prior_red.ply", partial, result)
    base.draw_projection_overlay(args.output_dir / "projection.png", args.semantic, partial, result, projector)
    record = {
        "method": "agent_support_locked_model_edit", "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False, "sample_or_category_specific_parameters": False,
        "nonrigid_deformation_used": False, "anisotropic_scale_used": False,
        "anchor": str(args.anchor), "edited_proposal": str(args.edited_proposal),
        "shared_radii": [float(radius) for radius in args.radii], "accepted": accepted,
        "route": route, "anchor_score": concise(anchor_score),
        "anchor_multiview": anchor_multiview.to_dict(),
        "eligible_radii": [float(candidate["radius_ratio"]) for candidate in eligible],
        "selected": {key: (value.to_dict() if key == "multiview_score" else value)
                     for key, value in selected.items() if key != "points"},
        "candidates": [{key: (value.to_dict() if key == "multiview_score" else value)
                        for key, value in candidate.items() if key != "points"}
                       for candidate in candidates],
        "outputs": {"points": str(args.output_dir / "registered_100k.ply")},
    }
    (args.output_dir / "support_locked_info.json").write_text(
        json.dumps(base.jsonable(record), indent=2), encoding="utf-8"
    )
    print(json.dumps({"route": route, "anchor": concise(anchor_score),
                      "selected": concise(selected["score"]),
                      "radius": selected["radius_ratio"]}, indent=2))


if __name__ == "__main__":
    main()
