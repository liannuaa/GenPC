#!/usr/bin/env python3
"""GT-free local 7-DoF Sim(3) refinement after v3 symmetry disambiguation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

import scripts.run_pixal_pca_sim3_ttt_v2 as base
import scripts.run_pixal_pca_depth_sim3_ttt_v3 as depth_v3


ROOT = base.ROOT
INPUT_ROOT = ROOT / "gpt_version/_pixal_pca_depth_sim3_ttt_v3_20260822"
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_local_sim3_ttt_v4_20260822"


def refine(source, partial, rotation, scale, translation, projector, target_mask,
           diagonal, render_size, levels):
    evaluate = depth_v3.depth_aware_evaluate
    value = evaluate(source, partial, rotation, scale, translation, projector,
                     target_mask, diagonal, render_size)
    trace = [{"level": -1, "rotation": rotation, "scale": scale,
              "translation": translation, "metrics": value}]
    angle_step = np.deg2rad(8.)
    scale_step = .04
    translation_step = .045 * diagonal
    axes = np.eye(3)
    for level in range(levels):
        for _ in range(10):
            proposals = []
            for sign in (-1., 1.):
                for axis in range(3):
                    delta = Rotation.from_rotvec(sign * angle_step * axes[axis]).as_matrix()
                    proposals.append((delta @ rotation, scale, translation.copy(), f"r{axis}"))
                candidate_scale = scale + sign * scale_step
                if .45 <= candidate_scale <= 1.20:
                    proposals.append((rotation, candidate_scale, translation.copy(), "scale"))
                for axis in range(3):
                    candidate_translation = translation.copy()
                    candidate_translation[axis] += sign * translation_step
                    proposals.append((rotation, scale, candidate_translation, f"t{axis}"))
            best = (value["score"], rotation, scale, translation, value, None)
            for candidate_rotation, candidate_scale, candidate_translation, coordinate in proposals:
                candidate = evaluate(
                    source, partial, candidate_rotation, candidate_scale,
                    candidate_translation, projector, target_mask, diagonal,
                    render_size)
                if candidate["score"] > best[0]:
                    best = (candidate["score"], candidate_rotation, candidate_scale,
                            candidate_translation, candidate, coordinate)
            if best[0] <= value["score"] + 1e-10:
                break
            _, rotation, scale, translation, value, coordinate = best
            trace.append({"level": level, "coordinate": coordinate,
                          "rotation": rotation, "scale": scale,
                          "translation": translation, "metrics": value})
        angle_step *= .5
        scale_step *= .5
        translation_step *= .5
    return rotation, scale, translation, value, trace


def process(args, sample):
    sample_dir = args.gpt_root / sample
    camera_dir = args.camera_root / sample
    input_info = json.loads((args.input_root / sample /
        f"{sample}_pca_sim3_ttt_v2_info.json").read_text())
    selected = next(x for x in input_info["ranking"] if x["rank"] == 1)
    source_full = base.load_points(sample_dir / "pixal3d_sampled_100k.ply")
    partial_full = base.load_points(ROOT / "data" / f"{sample}.ply")
    source = base.subset(source_full, args.source_points)
    partial = base.subset(partial_full, args.partial_points)
    diagonal = max(float(np.linalg.norm(np.ptp(partial_full, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial_full, camera_dir / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")
    target_mask = base.render_mask(projector, partial_full, args.render_size)
    rotation, scale, translation, metrics, trace = refine(
        source, partial, np.asarray(selected["rotation"]), float(selected["scale"]),
        np.asarray(selected["translation"]), projector, target_mask, diagonal,
        args.render_size, args.levels)
    transform = base.make_transform(rotation, scale, translation)
    registered = base.apply_sim3(source_full, rotation, scale, translation)
    output_dir = args.output_root / sample; output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_local_sim3_ttt_v4"
    paths = {"registered": Path(f"{stem}_registered_100k.ply"),
             "compare": Path(f"{stem}_partial_gray_pixal_red.ply"),
             "transform": Path(f"{stem}.npy"),
             "mesh": Path(f"{stem}_registered_mesh.glb"),
             "projection": Path(f"{stem}_projection.png")}
    base.write_points(paths["registered"], registered)
    base.write_compare(paths["compare"], partial_full, registered)
    np.save(paths["transform"], transform)
    mesh = trimesh.load(sample_dir / "pixal3d.glb", force="scene", process=False)
    mesh.apply_transform(transform); mesh.export(paths["mesh"])
    base.draw_projection_overlay(paths["projection"], camera_dir / "img.png",
                                 partial_full, registered, projector)
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    full_metrics = base.mask_metrics(base.render_mask(projector, partial_full, 512),
                                     base.render_mask(projector, registered, 512))
    info = {"sample_id": sample,
            "method": "v3_all24_depth_disambiguation_plus_local_7dof_sim3_ttt_v4",
            "strict_zero_shot": True,
            "ground_truth_used_for_inference_or_selection": False,
            "sample_specific_parameters": False, "fusion_run": False,
            "initial_v3_metrics": selected["metrics"], "final_metrics": metrics,
            "full_resolution_projection": full_metrics,
            "transform": transform, "trace": trace,
            "transform_contract": {"singular_values": singular,
                "isotropic_scale": float(singular.mean()),
                "proper_rotation": bool(np.linalg.det(rotation) > .999999),
                "all_100k_pixal_points_preserved": len(registered) == 100000,
                "nonrigid_deformation_used": False, "anisotropic_scale_used": False},
            "shared_parameters": vars(args), "outputs": paths}
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(info), indent=2))
    print(json.dumps(base.jsonable({"sample": sample, "metrics": metrics,
                                    "full_projection": full_metrics,
                                    "contract": info["transform_contract"],
                                    "outputs": paths}), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=["07136"])
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--render-size", type=int, default=128)
    parser.add_argument("--source-points", type=int, default=12000)
    parser.add_argument("--partial-points", type=int, default=6000)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    for sample in args.samples: process(args, str(sample))


if __name__ == "__main__": main()
