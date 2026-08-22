#!/usr/bin/env python3
"""Residual-driven partial-core editing for registered Pixal3D geometry.

This is the conservative successor to v16.  It uses only the poorly aligned
tail of same-camera visible correspondences as deformation handles.  The raw
partial scan is immutable, all 100k Pixal points are retained, and compact
support leaves already-aligned and unobserved geometry unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh

import scripts.run_pixal_pca_sim3_ttt_v2 as base
import scripts.run_pixal_partial_core_local_edit_v16 as v16
from scripts.run_registration_deformation_fusion_ablation import estimate_normals


ROOT = base.ROOT
REGISTRATION_ROOT = (
    ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822")
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_residual_local_edit_v17_20260822"


def aggregate_residual_pairs(partial, generated, partial_ids, generated_ids):
    """Aggregate rays, then retain only a shared high-residual handle tail."""
    source_ids, targets = v16._aggregate_pairs_v16(
        partial, generated, partial_ids, generated_ids)
    if not len(source_ids):
        return source_ids, targets
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    residual = np.linalg.norm(generated[source_ids] - targets, axis=1)
    threshold = max(
        v16.RESIDUAL_MIN_RATIO * diagonal,
        float(np.quantile(residual, v16.RESIDUAL_QUANTILE)))
    keep = residual >= threshold
    return source_ids[keep], targets[keep]


def process(args, sample):
    started = time.perf_counter()
    registration_dir = args.registration_root / sample
    registered_path = registration_dir / (
        f"{sample}_unified_registration_v14_registered_100k.ply")
    generated = base.load_points(registered_path)
    partial = base.load_points(ROOT / "data" / f"{sample}.ply")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    partial_normals = estimate_normals(partial, radius=.03 * diagonal)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera_root / sample / "camera.pth",
        padding=args.padding, image_shape=(512, 512), device="cpu")
    before_projection = base.mask_metrics(
        base.render_mask(projector, partial, 512),
        base.render_mask(projector, generated, 512))
    deformed, edit_info, state = v16.optimize_local_edit(
        generated, partial, partial_normals, projector, args)
    after_projection = base.mask_metrics(
        base.render_mask(projector, partial, 512),
        base.render_mask(projector, deformed, 512))

    active_ratio = (
        edit_info.get("active_nodes", 0)
        / max(edit_info.get("graph_nodes", 1), 1))
    accepted = bool(
        state is not None
        and edit_info["handle_improvement_ratio"] >= args.min_improvement_ratio
        and active_ratio <= args.max_active_node_ratio
        and edit_info["far_max_displacement_ratio"]
        <= args.far_max_displacement_ratio
        and after_projection["iou"]
        >= before_projection["iou"] - args.max_iou_drop
        and after_projection["coverage"]
        >= before_projection["coverage"] - args.max_coverage_drop
        and after_projection["leakage"]
        <= before_projection["leakage"] + args.max_leakage_increase)
    if not accepted:
        deformed = generated.copy()
        after_projection = before_projection.copy()
    edit_info["active_node_ratio"] = float(active_ratio)
    edit_info["accepted"] = accepted
    edit_info["reason"] = (
        "residual_local_edit_accepted" if accepted
        else "residual_local_edit_rejected")

    fused = np.concatenate([deformed, partial], axis=0)
    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_residual_local_edit_v17"
    paths = {
        "deformed_complete": Path(f"{stem}_deformed_complete_100k.ply"),
        "fused": Path(f"{stem}_fused.ply"),
        "compare": Path(f"{stem}_partial_gray_deformed_red.ply"),
        "mesh": Path(f"{stem}_deformed_mesh.glb"),
    }
    base.write_points(paths["deformed_complete"], deformed)
    base.write_points(paths["fused"], fused)
    base.write_compare(paths["compare"], partial, deformed)
    mesh = trimesh.load(
        args.gpt_root / sample / "pixal3d.glb", force="scene", process=False)
    registration_transform = np.load(
        registration_dir / f"{sample}_unified_registration_v14.npy")
    mesh.apply_transform(registration_transform)
    if accepted and state is not None:
        for geometry in mesh.geometry.values():
            geometry.vertices = v16.deform_points(
                np.asarray(geometry.vertices), state["graph"],
                state["translations"])
    mesh.export(paths["mesh"])

    result = {
        "sample_id": sample,
        "method": "residual_driven_partial_core_compact_graph_v17",
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "registered_complete_input": registered_path,
        "partial_input": ROOT / "data" / f"{sample}.ply",
        "before_projection": before_projection,
        "after_projection": after_projection,
        "local_edit": edit_info,
        "fusion_contract": {
            "deformed_complete_points": int(len(deformed)),
            "partial_core_points": int(len(partial)),
            "fused_points": int(len(fused)),
            "all_pixal_points_retained": len(deformed) == len(generated) == 100000,
            "partial_core_exact": bool(np.array_equal(fused[len(deformed):], partial)),
            "points_deleted": False,
            "local_scale_or_shear_used": False,
            "residual_only_handles": True,
            "far_geometry_identity_guard": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "shared_parameters": vars(args),
        "outputs": paths,
    }
    Path(f"{stem}_info.json").write_text(
        json.dumps(base.jsonable(result), indent=2))
    print(json.dumps(base.jsonable(result), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--registration-root", type=Path, default=REGISTRATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=["09639", "07136"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--correspondence-distance-ratio", type=float, default=.08)
    parser.add_argument("--min-handles", type=int, default=256)
    parser.add_argument("--node-spacing-ratio", type=float, default=.045)
    parser.add_argument("--min-nodes", type=int, default=96)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--graph-knn", type=int, default=6)
    parser.add_argument("--influence-k", type=int, default=4)
    parser.add_argument("--support-radius-ratio", type=float, default=.055)
    parser.add_argument("--far-identity-radius-ratio", type=float, default=.09)
    parser.add_argument("--max-displacement-ratio", type=float, default=.03)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--trim-quantile", type=float, default=.80)
    parser.add_argument("--smooth-weight", type=float, default=35.)
    parser.add_argument("--identity-weight", type=float, default=14.)
    parser.add_argument("--residual-quantile", type=float, default=.55)
    parser.add_argument("--residual-min-ratio", type=float, default=.006)
    parser.add_argument("--max-active-node-ratio", type=float, default=.80)
    parser.add_argument("--min-improvement-ratio", type=float, default=.08)
    parser.add_argument("--far-max-displacement-ratio", type=float, default=.001)
    parser.add_argument("--max-iou-drop", type=float, default=.015)
    parser.add_argument("--max-coverage-drop", type=float, default=.015)
    parser.add_argument("--max-leakage-increase", type=float, default=.02)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    # v16 owns the tested optimizer; replace only its generic pair selector.
    v16._aggregate_pairs_v16 = v16.aggregate_pairs
    v16.RESIDUAL_QUANTILE = args.residual_quantile
    v16.RESIDUAL_MIN_RATIO = args.residual_min_ratio
    v16.aggregate_pairs = aggregate_residual_pairs
    for sample in args.samples:
        process(args, str(sample))


if __name__ == "__main__":
    main()
