#!/usr/bin/env python3
"""Run and gate a category-free visible structural deformation-graph action."""

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
from src.agent_pareto_policy import registration_objectives, select_pareto_knee
from src.bidirectional_cycle_registration import visible_score
from src.visibility_conditioned_correspondence import visible_structural_correspondences
from src.visibility_graph_deformation import solve_visibility_deformation_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prior, partial = base.load_points(args.prior), base.load_points(args.partial)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(partial, args.camera, padding=.15,
                                                        image_shape=(512, 512), device=args.device)
    before = visible_score(partial, prior, projector, diagonal, pixel_radius=5.)
    pairs = visible_structural_correspondences(partial, prior, projector, diagonal=diagonal)
    proposed, warp = solve_visibility_deformation_graph(
        prior, partial, pairs["partial_ids"], pairs["prior_ids"], pairs["confidence"], seed=args.seed)
    candidates = []
    for fraction in (.25, .5, .75, 1.):
        points = prior + float(fraction) * (proposed - prior)
        score = visible_score(partial, points, projector, diagonal, pixel_radius=5.)
        candidates.append({"fraction": fraction, "points": points, "score": score,
                           "deformation_ratio": float(np.linalg.norm(points - prior, axis=1).mean() / diagonal),
                           "accepted": accept_registration_refinement(before, score)})
    valid = [item for item in candidates if item["accepted"]]
    for item in valid:
        item["objectives"] = registration_objectives(item["score"], deformation_ratio=item["deformation_ratio"])
    decision = select_pareto_knee(valid) if valid else None
    selected = decision["selected"] if decision else None
    result = selected["points"] if selected else prior
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "07136_visibility_graph"
    base.write_points(Path(f"{stem}_registered_100k.ply"), result)
    base.write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, result)
    base.draw_projection_overlay(Path(f"{stem}_projection.png"), args.semantic, partial, result, projector)
    # The graph action edits sampled surface geometry.  Preserve the source
    # mesh unchanged rather than falsely claiming a mesh warp exists.
    trimesh.load(args.mesh, force="scene", process=False).export(Path(f"{stem}_source_mesh.glb"))
    record = {"method": "visibility_conditioned_structural_deformation_graph", "strict_zero_shot": True,
              "ground_truth_cd_emd_used": False, "accepted": selected is not None, "before": before,
              "after": selected["score"] if selected else before, "correspondence": {k: v for k, v in pairs.items()
                  if k not in {"partial_ids", "prior_ids", "cost", "confidence"}}, "warp": warp,
              "candidates": [{"fraction": x["fraction"], "score": x["score"], "accepted": x["accepted"],
                              "deformation_ratio": x["deformation_ratio"], "objectives": x.get("objectives")}
                             for x in candidates], "agent_decision": None if decision is None else {
                                  "selection": decision["selection"], "regret": decision["regret"],
                                  "pareto_fractions": [x["fraction"] for x in decision["archive"]],
                                  "selected_fraction": selected["fraction"]},
              "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve())}}
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"accepted": selected is not None, "before": before["objective"],
                      "after": record["after"]["objective"], "warp": warp}, indent=2))


if __name__ == "__main__":
    main()
