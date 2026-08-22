#!/usr/bin/env python3
"""GT-free confidence router for fast SO(3) TTT and GenPC/PCA fallback.

The new global search is a confidence-gated residual upgrade over the existing
PCA/Sim(3) path.  Every sample uses the same observable-evidence gate.  The
router never loads GT or metric results and never changes the frozen Pixal body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from scripts.run_pixal_batched_adaptive_ttt_v12 import confidence_gate


ROOT = base.ROOT
FAST_ROOT = ROOT / "gpt_version/_pixal_batched_adaptive_ttt_v12_20260822"
FALLBACK_ROOT = ROOT / "gpt_version/_pixal_scale_ttt_v8_20260822"
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_unified_registration_v14_20260822"


def process(args, sample):
    fast_dir = args.fast_root / sample
    fast_info_path = fast_dir / f"{sample}_batched_so3_sim3_ttt_v11_info.json"
    fast_info = json.loads(fast_info_path.read_text())
    fast_metrics = fast_info["selected"]["metrics"]
    fast_confident = confidence_gate(fast_metrics)
    if fast_confident:
        route = "fast_global_so3_ttt"
        transform = np.asarray(fast_info["transform"], dtype=np.float64)
        source_info = fast_info_path
    else:
        route = "genpc_pca_sim3_fallback"
        transform = np.load(
            args.fallback_root / sample / f"{sample}_scale_ttt_v8.npy")
        source_info = (
            args.fallback_root / sample / f"{sample}_scale_ttt_v8_info.json")

    source_full = base.load_points(
        args.gpt_root / sample / "pixal3d_sampled_100k.ply")
    partial_full = base.load_points(ROOT / "data" / f"{sample}.ply")
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    scale = float(singular.mean())
    rotation = transform[:3, :3] / scale
    registered = base.apply_sim3(source_full, rotation, scale, transform[:3, 3])
    camera_dir = args.camera_root / sample
    projector = base.SavedCameraProjector.from_partial(
        partial_full, camera_dir / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")
    full_projection = base.mask_metrics(
        base.render_mask(projector, partial_full, 512),
        base.render_mask(projector, registered, 512))

    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_unified_registration_v14"
    paths = {
        "registered": Path(f"{stem}_registered_100k.ply"),
        "compare": Path(f"{stem}_partial_gray_pixal_red.ply"),
        "transform": Path(f"{stem}.npy"),
        "mesh": Path(f"{stem}_registered_mesh.glb"),
        "projection": Path(f"{stem}_projection.png"),
    }
    base.write_points(paths["registered"], registered)
    base.write_compare(paths["compare"], partial_full, registered)
    np.save(paths["transform"], transform)
    mesh = trimesh.load(
        args.gpt_root / sample / "pixal3d.glb", force="scene", process=False)
    mesh.apply_transform(transform)
    mesh.export(paths["mesh"])
    base.draw_projection_overlay(paths["projection"], camera_dir / "img.png",
                                 partial_full, registered, projector)
    result = {
        "sample_id": sample,
        "method": "confidence_gated_fast_so3_residual_over_genpc_pca_v14",
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "sample_specific_parameters": False,
        "category_specific_parameters": False,
        "fusion_run": False,
        "selected_route": route,
        "fast_candidate_confident": fast_confident,
        "fast_candidate_metrics": fast_metrics,
        "fast_confidence_gate": {
            "iou_min": .85, "coverage_min": .90, "leakage_max": .08,
            "normalized_visible_depth_max": .10,
            "normalized_surface_trim70_max": .012,
        },
        "selected_source_info": source_info,
        "transform": transform,
        "full_resolution_projection": full_projection,
        "transform_contract": {
            "singular_values": singular,
            "isotropic_scale": scale,
            "proper_rotation": bool(np.linalg.det(rotation) > .999999),
            "all_100k_pixal_points_preserved": len(registered) == 100000,
            "nonrigid_deformation_used": False,
            "anisotropic_scale_used": False,
        },
        "outputs": paths,
    }
    Path(f"{stem}_info.json").write_text(
        json.dumps(base.jsonable(result), indent=2))
    print(f"{sample}: {route}, full_iou={full_projection['iou']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--fast-root", type=Path, default=FAST_ROOT)
    parser.add_argument("--fallback-root", type=Path, default=FALLBACK_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=[
        "01184", "05117", "05452", "06127", "06145", "06188", "06830",
        "07136", "07306", "09639"])
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    results = [process(args, str(sample)) for sample in args.samples]
    summary = {
        "method": "confidence_gated_fast_so3_residual_over_genpc_pca_v14",
        "ground_truth_used_for_inference_or_selection": False,
        "samples": [{"sample_id": item["sample_id"],
                     "route": item["selected_route"],
                     "full_projection": item["full_resolution_projection"]}
                    for item in results],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "redwood_10_unified_registration_summary.json").write_text(
        json.dumps(base.jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
