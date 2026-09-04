#!/usr/bin/env python3
"""Execute a no-GT proposal-guided local surfel action."""

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
from src.agent_proposal_guided_surfel import proposal_guided_surfel_candidates
from src.bidirectional_cycle_registration import visible_score
from src.multiview_agent_feedback import make_orthographic_reference, measure_multiview_evidence


def concise(score: dict) -> dict:
    return {"objective": float(score["objective"]),
            "geometric_objective": float(score["geometric"]["objective"]),
            "iou": float(score["projection"]["iou"]),
            "coverage": float(score["projection"]["coverage"]),
            "leakage": float(score["projection"]["leakage"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transfer-ratio", type=float, default=.025)
    parser.add_argument("--sigma-ratio", type=float, default=.035)
    parser.add_argument("--support-ratio", type=float, default=.11)
    parser.add_argument("--max-displacement-ratio", type=float, default=.028)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor, guidance, partial = (base.load_points(path) for path in
                                 (args.anchor, args.guidance, args.partial))
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=.15, image_shape=(512, 512), device="cpu")
    anchor_score = visible_score(partial, anchor, projector, diagonal, pixel_radius=5.)
    reference = make_orthographic_reference(partial, anchor)
    anchor_multi = measure_multiview_evidence(partial, anchor, reference)
    candidates, action = proposal_guided_surfel_candidates(
        anchor, guidance, partial, projector, diagonal=diagonal,
        transfer_ratio=args.transfer_ratio, sigma_ratio=args.sigma_ratio,
        support_ratio=args.support_ratio, max_displacement_ratio=args.max_displacement_ratio)
    evaluated = []
    for fraction, points in candidates:
        score = visible_score(partial, points, projector, diagonal, pixel_radius=5.)
        multi = measure_multiview_evidence(partial, points, reference)
        # A loop state update must be monotone in its own observed objective.
        # The broader common gate retains a small numerical tolerance for
        # independent one-shot proposals; retaining a merely tolerated
        # regression here would accumulate drift over multiple agent rounds.
        accepted = bool(score["objective"] < anchor_score["objective"] and
                        accept_agent_proposal(anchor_score, score,
                                              anchor_multiview=anchor_multi,
                                              proposal_multiview=multi))
        evaluated.append((score["objective"], fraction, points, score, multi, accepted))
    eligible = [item for item in evaluated if item[-1]]
    selected = min(eligible or evaluated or [(0., 0., anchor, anchor_score, anchor_multi, False)],
                   key=lambda item: item[0])
    accepted = bool(eligible)
    result = selected[2] if accepted else anchor
    route = "accepted_proposal_guided_surfel" if accepted else "anchor_fallback"
    outputs = {"points": args.output_dir / "registered_100k.ply",
               "compare": args.output_dir / "partial_gray_prior_red.ply",
               "projection": args.output_dir / "projection.png",
               "record": args.output_dir / "proposal_guided_surfel.json"}
    base.write_points(outputs["points"], result)
    base.write_compare(outputs["compare"], partial, result)
    base.draw_projection_overlay(outputs["projection"], args.semantic, partial, result, projector)
    record = {"method": "agent_text_3d_proposal_guided_surfel_edit",
              "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
              "sample_or_category_specific_parameters": False,
              "generated_guidance_replaces_complete_prior": False,
              "hidden_geometry_deleted": False, "anchor": concise(anchor_score),
              "anchor_multiview": anchor_multi.to_dict(), "action": action,
              "candidates": [{"fraction": float(item[1]), "score": concise(item[3]),
                              "multiview": item[4].to_dict(), "accepted": bool(item[5])}
                             for item in evaluated], "route": route, "accepted": accepted,
              "selected_fraction": float(selected[1]) if accepted else 0.,
              "selected": concise(selected[3]),
              "inputs": {"anchor": str(args.anchor), "guidance": str(args.guidance),
                         "partial": str(args.partial)},
              "shared_parameters": {key: value for key, value in vars(args).items()
                                  if key not in {"anchor", "guidance", "partial", "camera", "semantic", "output_dir"}},
              "outputs": {key: str(value) for key, value in outputs.items()}}
    outputs["record"].write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"route": route, "anchor": concise(anchor_score),
                      "selected": concise(selected[3]), "fraction": selected[1]}, indent=2))


if __name__ == "__main__":
    main()
