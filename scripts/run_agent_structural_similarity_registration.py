#!/usr/bin/env python3
"""Saved-view structural-match proper Sim(3) action with a no-harm gate."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.agent_completion_policy import accept_registration_refinement
from src.bidirectional_cycle_registration import interpolate_sim3, visible_score
from src.ray_consistent_registration import apply_transform
from src.structural_similarity_registration import robust_structural_similarity
from src.visibility_conditioned_correspondence import visible_structural_correspondences


def main():
    p = argparse.ArgumentParser()
    for name in ("prior", "mesh", "partial", "camera", "semantic", "output_dir"):
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=6145); p.add_argument("--device", default="cpu")
    args = p.parse_args()
    prior, partial = base.load_points(args.prior), base.load_points(args.partial)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(partial, args.camera, padding=.15, image_shape=(512, 512), device=args.device)
    before = visible_score(partial, prior, projector, diagonal, pixel_radius=5.)
    pairs = visible_structural_correspondences(partial, prior, projector, diagonal=diagonal)
    transform, fit = robust_structural_similarity(prior[pairs["prior_ids"]], partial[pairs["partial_ids"]],
                                                  pairs["confidence"], diagonal=diagonal, seed=args.seed)
    candidates = []
    for fraction in (.25, .5, .75, 1.):
        step = interpolate_sim3(transform, fraction); points = apply_transform(prior, step)
        score = visible_score(partial, points, projector, diagonal, pixel_radius=5.)
        candidates.append({"fraction": fraction, "step": step, "points": points, "score": score,
                           "accepted": accept_registration_refinement(before, score)})
    valid = [x for x in candidates if x["accepted"]]; selected = min(valid, key=lambda x:x["score"]["objective"]) if valid else None
    result = selected["points"] if selected else prior; step = selected["step"] if selected else np.eye(4)
    args.output_dir.mkdir(parents=True, exist_ok=True); stem=args.output_dir/'07136_structural_similarity'
    base.write_points(Path(f'{stem}_registered_100k.ply'), result); base.write_compare(Path(f'{stem}_partial_gray_pixal_red.ply'),partial,result)
    base.draw_projection_overlay(Path(f'{stem}_projection.png'),args.semantic,partial,result,projector)
    mesh=trimesh.load(args.mesh,force='scene',process=False); mesh.apply_transform(step); mesh.export(Path(f'{stem}_registered_mesh.glb'))
    record={"method":"visible_structural_similarity_proper_sim3","strict_zero_shot":True,"ground_truth_cd_emd_used":False,
            "accepted":selected is not None,"before":before,"after":selected['score'] if selected else before,"fit":fit,
            "candidates":[{"fraction":x['fraction'],"score":x['score'],"accepted":x['accepted']} for x in candidates]}
    Path(f'{stem}_info.json').write_text(json.dumps(base.jsonable(record),indent=2),encoding='utf-8')
    print(json.dumps({"accepted":record['accepted'],"before":before['objective'],"after":record['after']['objective'],"fit":fit},indent=2))
if __name__=='__main__': main()
