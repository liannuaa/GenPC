#!/usr/bin/env python3
"""Apply a no-GT structural-patch Sim(3) action to one registered prior."""

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
from src.bidirectional_cycle_registration import interpolate_sim3, visible_score
from src.ray_consistent_registration import apply_transform
from src.structural_patch_tto import optimize_structural_patch_sim3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prior, partial = base.load_points(args.prior), base.load_points(args.partial)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(partial, args.camera, padding=.15,
                                                        image_shape=(512, 512), device=args.device)
    before = visible_score(partial, prior, projector, diagonal, pixel_radius=5.)
    candidate, transform, trace = optimize_structural_patch_sim3(
        prior, partial, projector, diagonal=diagonal, seed=args.seed, device=args.device)
    # A structural term can favor a small coherent shrink that lowers its local
    # plane residual but harms silhouette coverage.  Test only bounded
    # fractions of the proper total Sim(3) under the common no-harm gate.
    candidates = []
    for fraction in (.25, .5, .75, 1.):
        step = interpolate_sim3(transform, fraction)
        moved = apply_transform(prior, step)
        score = visible_score(partial, moved, projector, diagonal, pixel_radius=5.)
        candidates.append({"fraction": fraction, "transform": step, "points": moved, "score": score,
                           "accepted": accept_registration_refinement(before, score)})
    valid = [item for item in candidates if item["accepted"]]
    selected = min(valid, key=lambda item: item["score"]["objective"]) if valid else None
    accepted = selected is not None
    result = selected["points"] if selected else prior
    transform = selected["transform"] if selected else np.eye(4)
    after = selected["score"] if selected else before
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "07136_structural_patch"
    base.write_points(Path(f"{stem}_registered_100k.ply"), result)
    base.write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, result)
    base.draw_projection_overlay(Path(f"{stem}_projection.png"), args.semantic, partial, result, projector)
    mesh = trimesh.load(args.mesh, force="scene", process=False); mesh.apply_transform(transform)
    mesh.export(Path(f"{stem}_registered_mesh.glb")); np.save(Path(f"{stem}.npy"), transform)
    record = {"method": "saved_view_structural_patch_proper_sim3", "strict_zero_shot": True,
              "ground_truth_cd_emd_used": False, "accepted": accepted, "before": before, "after": after,
              "trace": trace, "fraction_candidates": [{"fraction": item["fraction"], "score": item["score"],
                                                          "accepted": item["accepted"]} for item in candidates],
              "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve())}}
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"accepted": accepted, "objective_before": before["objective"], "objective_after": after["objective"]}, indent=2))


if __name__ == "__main__":
    main()
