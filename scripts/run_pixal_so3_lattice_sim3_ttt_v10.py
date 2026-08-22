#!/usr/bin/env python3
"""GT-free coarse-to-fine SO(3) lattice plus proper Sim(3) TTT.

This is a category-agnostic repair for cases where partial-view PCA does not
put the correct complete-to-partial rotation in the usual 24 hypotheses.  GT
is never loaded.  The frozen Pixal3D geometry is transformed only by one
proper rotation, one isotropic scale, and one translation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

import scripts.run_pixal_pca_sim3_ttt_v2 as base
import scripts.run_pixal_pca_depth_sim3_ttt_v3 as depth
import scripts.run_pixal_local_sim3_ttt_v4 as local


ROOT = base.ROOT
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_so3_lattice_sim3_v10_20260822"


def rotation_lattice(source, partial, offset_degrees):
    """Compose a shared Euler lattice with the 24 proper PCA hypotheses."""
    candidates = []
    seen = set()
    offsets = itertools.product(offset_degrees, repeat=3)
    offset_rotations = [
        (angles, Rotation.from_euler("xyz", angles, degrees=True).as_matrix())
        for angles in offsets
    ]
    for pca_id, item in enumerate(base.proper_pca_rotations(source, partial)):
        for angles, delta in offset_rotations:
            matrix = delta @ item["rotation"]
            key = tuple(np.round(matrix, 7).ravel())
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "pca_rotation_id": pca_id,
                "offset_degrees_xyz": angles,
                "rotation": matrix,
            })
    return candidates


def centered_trials(source, partial, rotation, scales, evaluate):
    source_center = np.median(source, axis=0)
    target_center = np.median(partial, axis=0)
    trials = []
    for scale in scales:
        translation = target_center - scale * (rotation @ source_center)
        metrics = evaluate(rotation, scale, translation)
        trials.append((metrics["score"], scale, translation, metrics))
    return max(trials, key=lambda item: item[0])


def optimize_scale_translation(source, partial, rotation, scales, projector,
                               target_mask, diagonal, render_size, levels,
                               min_scale, max_scale):
    def evaluate(r, s, t):
        return depth.depth_aware_evaluate(
            source, partial, r, s, t, projector, target_mask, diagonal,
            render_size)

    _, scale, translation, metrics = centered_trials(
        source, partial, rotation, scales, evaluate)
    scale_step = .10
    translation_step = .075 * diagonal
    trace = []
    for level_id in range(levels):
        for _ in range(8):
            proposals = []
            for sign in (-1., 1.):
                candidate_scale = scale + sign * scale_step
                if min_scale <= candidate_scale <= max_scale:
                    proposals.append((candidate_scale, translation.copy(), "scale"))
                for axis in range(3):
                    candidate_translation = translation.copy()
                    candidate_translation[axis] += sign * translation_step
                    proposals.append((scale, candidate_translation, f"t{axis}"))
            best = (metrics["score"], scale, translation, metrics, None)
            for candidate_scale, candidate_translation, coordinate in proposals:
                candidate_metrics = evaluate(
                    rotation, candidate_scale, candidate_translation)
                if candidate_metrics["score"] > best[0]:
                    best = (candidate_metrics["score"], candidate_scale,
                            candidate_translation, candidate_metrics, coordinate)
            if best[0] <= metrics["score"] + 1e-10:
                break
            _, scale, translation, metrics, coordinate = best
            trace.append({"level": level_id, "coordinate": coordinate,
                          "scale": scale, "translation": translation.copy(),
                          "metrics": metrics})
        scale_step *= .5
        translation_step *= .5
    return scale, translation, metrics, trace


def process(args, sample):
    sample_dir = args.gpt_root / sample
    camera_dir = args.camera_root / sample
    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    source_full = base.load_points(sample_dir / "pixal3d_sampled_100k.ply")
    partial_full = base.load_points(ROOT / "data" / f"{sample}.ply")
    if len(source_full) != 100000:
        raise ValueError("Frozen Pixal PLY is not 100k")
    diagonal = max(float(np.linalg.norm(np.ptp(partial_full, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial_full, camera_dir / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")

    coarse_source = base.subset(source_full, args.coarse_source_points)
    coarse_partial = base.subset(partial_full, args.coarse_partial_points)
    coarse_mask = base.render_mask(projector, partial_full, args.coarse_render_size)
    lattice = rotation_lattice(source_full, partial_full, args.offset_degrees)
    coarse = []
    for candidate_id, item in enumerate(lattice):
        def coarse_evaluate(r, s, t):
            return depth.depth_aware_evaluate(
                coarse_source, coarse_partial, r, s, t, projector, coarse_mask,
                diagonal, args.coarse_render_size)
        _, scale, translation, metrics = centered_trials(
            coarse_source, coarse_partial, item["rotation"],
            args.coarse_scales, coarse_evaluate)
        coarse.append({"candidate_id": candidate_id, **item, "scale": scale,
                       "translation": translation, "metrics": metrics})
        if candidate_id % 50 == 0:
            print(f"{sample} coarse {candidate_id + 1}/{len(lattice)}", flush=True)
    coarse.sort(key=lambda item: item["metrics"]["score"], reverse=True)

    fine_source = base.subset(source_full, args.fine_source_points)
    fine_partial = base.subset(partial_full, args.fine_partial_points)
    fine_mask = base.render_mask(projector, partial_full, args.fine_render_size)
    fine = []
    for coarse_rank, item in enumerate(coarse[:args.fine_candidates], start=1):
        nearby_scales = sorted(set(args.fine_scales + [float(item["scale"])]))
        scale, translation, metrics, trace = optimize_scale_translation(
            fine_source, fine_partial, item["rotation"], nearby_scales,
            projector, fine_mask, diagonal, args.fine_render_size,
            args.scale_translation_levels, args.min_scale, args.max_scale)
        fine.append({**item, "coarse_rank": coarse_rank, "scale": scale,
                     "translation": translation, "metrics": metrics,
                     "scale_translation_trace": trace})
        print(f"{sample} fine {coarse_rank}/{args.fine_candidates} "
              f"score={metrics['score']:.6f} iou={metrics['iou']:.4f}", flush=True)
    fine.sort(key=lambda item: item["metrics"]["score"], reverse=True)

    refined = []
    for fine_rank, item in enumerate(fine[:args.rotation_refine_candidates], start=1):
        rotation, scale, translation, metrics, trace = local.refine(
            fine_source, fine_partial, item["rotation"], item["scale"],
            item["translation"], projector, fine_mask, diagonal,
            args.fine_render_size, args.rotation_levels)
        refined.append({**item, "fine_rank": fine_rank, "rotation": rotation,
                        "scale": scale, "translation": translation,
                        "metrics": metrics, "rotation_trace": trace})
        print(f"{sample} refine {fine_rank}/{args.rotation_refine_candidates} "
              f"score={metrics['score']:.6f} iou={metrics['iou']:.4f}", flush=True)
    refined.sort(key=lambda item: item["metrics"]["score"], reverse=True)
    selected = refined[0]

    transform = base.make_transform(
        selected["rotation"], selected["scale"], selected["translation"])
    registered = base.apply_sim3(
        source_full, selected["rotation"], selected["scale"],
        selected["translation"])
    stem = output_dir / f"{sample}_so3_lattice_sim3_ttt_v10"
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
    mesh = trimesh.load(sample_dir / "pixal3d.glb", force="scene", process=False)
    mesh.apply_transform(transform)
    mesh.export(paths["mesh"])
    base.draw_projection_overlay(paths["projection"], camera_dir / "img.png",
                                 partial_full, registered, projector)
    full_metrics = base.mask_metrics(
        base.render_mask(projector, partial_full, 512),
        base.render_mask(projector, registered, 512))
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    result = {
        "sample_id": sample,
        "method": "shared_so3_lattice_coarse_to_fine_proper_sim3_ttt_v10",
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "category_specific_parameters": False,
        "sample_specific_parameters": False,
        "fusion_run": False,
        "lattice_candidate_count": len(lattice),
        "selected": selected,
        "coarse_ranking": [
            {"rank": rank + 1, **item}
            for rank, item in enumerate(coarse[:args.saved_coarse_candidates])
        ],
        "fine_ranking": [
            {"rank": rank + 1, **item} for rank, item in enumerate(fine)
        ],
        "refined_ranking": [
            {"rank": rank + 1, **item} for rank, item in enumerate(refined)
        ],
        "full_resolution_projection": full_metrics,
        "transform": transform,
        "transform_contract": {
            "singular_values": singular,
            "isotropic_scale": float(singular.mean()),
            "proper_rotation": bool(np.linalg.det(selected["rotation"]) > .999999),
            "all_100k_pixal_points_preserved": len(registered) == 100000,
            "nonrigid_deformation_used": False,
            "anisotropic_scale_used": False,
        },
        "shared_parameters": vars(args),
        "outputs": paths,
    }
    Path(f"{stem}_info.json").write_text(
        json.dumps(base.jsonable(result), indent=2))
    print(json.dumps(base.jsonable({
        "sample": sample,
        "selected_candidate_id": selected["candidate_id"],
        "selected_coarse_rank": selected["coarse_rank"],
        "selected_fine_rank": selected["fine_rank"],
        "metrics": selected["metrics"],
        "full_projection": full_metrics,
        "contract": result["transform_contract"],
        "outputs": paths,
    }), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=["06830"])
    parser.add_argument("--offset-degrees", nargs="*", type=float,
                        default=[-45., 0., 45.])
    parser.add_argument("--coarse-scales", nargs="*", type=float,
                        default=[.6, .8, 1., 1.2, 1.4])
    parser.add_argument("--fine-scales", nargs="*", type=float,
                        default=[.6, .8, 1., 1.2, 1.4])
    parser.add_argument("--min-scale", type=float, default=.4)
    parser.add_argument("--max-scale", type=float, default=1.7)
    parser.add_argument("--coarse-render-size", type=int, default=64)
    parser.add_argument("--fine-render-size", type=int, default=128)
    parser.add_argument("--coarse-source-points", type=int, default=2500)
    parser.add_argument("--coarse-partial-points", type=int, default=1500)
    parser.add_argument("--fine-source-points", type=int, default=10000)
    parser.add_argument("--fine-partial-points", type=int, default=6000)
    parser.add_argument("--fine-candidates", type=int, default=24)
    parser.add_argument("--rotation-refine-candidates", type=int, default=8)
    parser.add_argument("--saved-coarse-candidates", type=int, default=48)
    parser.add_argument("--scale-translation-levels", type=int, default=5)
    parser.add_argument("--rotation-levels", type=int, default=5)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample in args.samples:
        process(args, str(sample))


if __name__ == "__main__":
    main()
