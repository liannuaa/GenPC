#!/usr/bin/env python3
"""Run the minimal partial-anchored 3DGS local geometry action.

One registered dual Gaussian field enters the action.  Partial anchors never
move; only saved-view-visible editable prior Gaussians receive a bounded
camera-ray depth update.  The controller compares preserve, conservative, and
standard corrections without GT.
"""

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
from src.agent_completion_policy import accept_agent_proposal
from src.agent_pareto_policy import registration_objectives, select_pareto_knee
from src.bidirectional_cycle_registration import visible_score
from src.multiview_agent_feedback import make_orthographic_reference, measure_multiview_evidence
from src.ray_conditioned_gaussian_edit import interpolate_ray_edit, saved_view_ray_edit


def _save_field(path: Path, source: dict[str, np.ndarray], means: np.ndarray) -> None:
    payload = {name: value for name, value in source.items()}
    payload["means"] = means.astype(np.float32)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=Path, required=True, help="dual_gaussian_fields.npz")
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--soft-observation", type=Path,
                        help="Optional gated MoGe NPZ; only fills scan-empty saved-view pixels.")
    parser.add_argument("--soft-weight", type=float, default=.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--max-depth-ratio", type=float, default=.035)
    args = parser.parse_args()

    loaded = np.load(args.field)
    state = {name: loaded[name] for name in loaded.files}
    if not {"means", "observed_anchor"}.issubset(state):
        raise ValueError("field must contain means and observed_anchor arrays")
    anchors = state["observed_anchor"].astype(bool)
    initial_means = state["means"].astype(np.float64)
    prior = initial_means[~anchors]
    partial = base.load_points(args.partial)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=.15, image_shape=(512, 512), device=args.device)
    before = visible_score(partial, prior, projector, diagonal, pixel_radius=5.)
    reference = make_orthographic_reference(partial, prior)
    anchor_multiview = measure_multiview_evidence(partial, prior, reference)
    soft_points = soft_confidence = None
    if args.soft_observation is not None:
        soft_state = np.load(args.soft_observation)
        soft_points = soft_state["points"]
        soft_confidence = soft_state["confidence"]
    corrected, edit = saved_view_ray_edit(
        partial, prior, projector, grid_size=args.grid_size,
        max_depth_ratio=args.max_depth_ratio, diagonal=diagonal, soft_points=soft_points,
        soft_confidence=soft_confidence, soft_weight=args.soft_weight)
    candidates = []
    for fraction in (.5, 1.):
        points = interpolate_ray_edit(prior, corrected, fraction)
        score = visible_score(partial, points, projector, diagonal, pixel_radius=5.)
        multiview = measure_multiview_evidence(partial, points, reference)
        deformation_ratio = float(np.linalg.norm(points - prior, axis=1).mean() / diagonal)
        candidates.append({
            "action": "conservative_ray_correction" if fraction < 1. else "standard_ray_correction",
            "fraction": fraction, "points": points, "score": score, "multiview": multiview,
            "deformation_ratio": deformation_ratio,
            "accepted": accept_agent_proposal(before, score, anchor_multiview=anchor_multiview,
                                                proposal_multiview=multiview),
        })
    valid = [item for item in candidates if item["accepted"]]
    for item in valid:
        item["objectives"] = registration_objectives(item["score"], deformation_ratio=item["deformation_ratio"])
    decision = select_pareto_knee(valid) if valid else None
    selected = decision["selected"] if decision else None
    result = selected["points"] if selected is not None else prior
    output_means = initial_means.copy(); output_means[~anchors] = result
    if not np.array_equal(output_means[anchors], initial_means[anchors]):
        raise RuntimeError("partial anchors changed during 3DGS edit")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = args.partial.stem
    stem = args.output_dir / f"{sample}_ray_conditioned_3dgs"
    _save_field(Path(f"{stem}_field.npz"), state, output_means)
    colors = state.get("colors")
    prior_colors = colors[~anchors] if colors is not None else None
    trimesh.points.PointCloud(result, colors=prior_colors).export(Path(f"{stem}_editable_prior_gaussians.ply"))
    base.write_compare(Path(f"{stem}_partial_gray_prior_red.ply"), partial, result)
    base.draw_projection_overlay(Path(f"{stem}_projection.png"), args.semantic, partial, result, projector)
    record = {
        "method": "partial_anchored_3dgs_saved_view_ray_depth_edit",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "global_motion": "frozen registered proper Sim(3); no additional global transform",
        "edit_support": "only editable prior Gaussians visible in shared saved-camera partial/prior support",
        "partial_anchor_policy": "means and scales immutable", "edit": edit,
        "before": before, "after": selected["score"] if selected else before,
        "anchor_multiview": anchor_multiview.to_dict(),
        "selected_multiview": selected["multiview"].to_dict() if selected else anchor_multiview.to_dict(),
        "accepted": selected is not None,
        "candidates": [{
            "action": item["action"], "fraction": item["fraction"], "score": item["score"],
            "multiview": item["multiview"].to_dict(), "deformation_ratio": item["deformation_ratio"],
            "accepted": item["accepted"], "objectives": item.get("objectives"),
        } for item in candidates],
        "agent_decision": None if decision is None else {
            "selection": decision["selection"], "regret": decision["regret"],
            "selected_action": selected["action"],
            "pareto_actions": [item["action"] for item in decision["archive"]],
        },
        "anchor_mean_drift": float(np.abs(output_means[anchors] - initial_means[anchors]).max(initial=0.)),
        "inputs": {"field": str(args.field.resolve()), "partial": str(args.partial.resolve()),
                   "soft_observation": (str(args.soft_observation.resolve())
                                        if args.soft_observation is not None else None),
                   "soft_weight": float(args.soft_weight) if args.soft_observation is not None else 0.},
    }
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps(base.jsonable({"accepted": record["accepted"], "before": before["objective"],
                                    "after": record["after"]["objective"], "edit": edit}), indent=2))


if __name__ == "__main__":
    main()
