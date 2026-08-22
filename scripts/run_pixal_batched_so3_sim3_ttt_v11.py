#!/usr/bin/env python3
"""Fast shared Pixal registration with batched coarse render-and-compare.

The coarse stage ranks the same category-agnostic SO(3)/scale hypothesis set
as v10, but performs GPU-batched silhouette and visible-depth rendering.  Only
the shortlist receives CPU 3D surface scoring and local proper-Sim(3) TTT.
GT is never loaded, and all frozen Pixal points are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import trimesh

import scripts.run_pixal_pca_sim3_ttt_v2 as base
import scripts.run_pixal_pca_depth_sim3_ttt_v3 as depth
import scripts.run_pixal_local_sim3_ttt_v4 as local
import scripts.run_pixal_so3_lattice_sim3_ttt_v10 as v10


ROOT = base.ROOT
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_batched_so3_sim3_v11_20260822"


def batched_coarse_rank(source, partial, lattice, scales, projector,
                        render_size, batch_size, splat_radius, device):
    """Rank orientation/scale hypotheses with batched hard z-buffer renders."""
    source = np.asarray(source, dtype=np.float32)
    partial = np.asarray(partial, dtype=np.float32)
    source_center = np.median(source, axis=0).astype(np.float32)
    target_center = np.median(partial, axis=0).astype(np.float32)
    hypotheses = []
    for item in lattice:
        for scale in scales:
            rotation = np.asarray(item["rotation"], dtype=np.float32)
            translation = target_center - float(scale) * (rotation @ source_center)
            hypotheses.append({**item, "scale": float(scale),
                               "translation": translation})

    target_depth, target_mask = depth.render_depth_mask(
        projector, partial, render_size, splat=max(0, int(splat_radius)))
    values = target_depth[target_mask]
    depth_span = max(float(np.quantile(values, .99) - np.quantile(values, .01)),
                     1e-8)
    pixels = int(render_size) * int(render_size)
    target_mask_t = torch.as_tensor(
        target_mask.reshape(-1), dtype=torch.bool, device=device)
    target_depth_t = torch.as_tensor(
        target_depth.reshape(-1), dtype=torch.float32, device=device)
    target_count = torch.count_nonzero(target_mask_t).float().clamp_min(1.)
    source_t = torch.as_tensor(source, dtype=torch.float32, device=device)
    center_t = torch.as_tensor(
        projector.center_xy, dtype=torch.float32, device=device)
    camera_scale = float(projector.scale_xy)
    padding = float(projector.padding)
    offsets = [(0, 0)]
    if int(splat_radius) > 0:
        offsets += [(-1, 0), (1, 0), (0, -1), (0, 1)]

    ranking = []
    with torch.no_grad():
        for start in range(0, len(hypotheses), int(batch_size)):
            chunk = hypotheses[start:start + int(batch_size)]
            rotations = torch.as_tensor(
                np.stack([x["rotation"] for x in chunk]),
                dtype=torch.float32, device=device)
            scales_t = torch.as_tensor(
                [x["scale"] for x in chunk], dtype=torch.float32,
                device=device)
            translations = torch.as_tensor(
                np.stack([x["translation"] for x in chunk]),
                dtype=torch.float32, device=device)
            moved = torch.einsum("bij,nj->bni", rotations, source_t)
            moved = moved * scales_t[:, None, None] + translations[:, None, :]
            camera_points = projector.camera.transform(
                moved.reshape(-1, 3)).reshape(len(chunk), len(source), 3)
            uv = (camera_points[..., :2] - center_t) / camera_scale
            uv = uv * (1. - 2. * padding) + .5
            uv[..., 1] = 1. - uv[..., 1]
            xy = torch.round(uv * float(render_size - 1)).long()
            z = camera_points[..., 2]
            valid_base = torch.isfinite(camera_points).all(dim=-1) & (z > 1e-8)
            rendered = torch.full(
                (len(chunk) * pixels,), float("inf"), dtype=torch.float32,
                device=device)
            batch_ids = torch.arange(len(chunk), device=device)[:, None].expand_as(z)
            for dx, dy in offsets:
                xx = xy[..., 0] + int(dx)
                yy = xy[..., 1] + int(dy)
                valid = (valid_base & (xx >= 0) & (xx < render_size)
                         & (yy >= 0) & (yy < render_size))
                flat_index = batch_ids[valid] * pixels + yy[valid] * render_size + xx[valid]
                rendered.scatter_reduce_(
                    0, flat_index, z[valid], reduce="amin", include_self=True)
            rendered = rendered.reshape(len(chunk), pixels)
            predicted_mask = torch.isfinite(rendered)
            overlap = predicted_mask & target_mask_t[None, :]
            intersection = overlap.sum(dim=1).float()
            predicted_count = predicted_mask.sum(dim=1).float().clamp_min(1.)
            union = (predicted_mask | target_mask_t[None, :]).sum(dim=1).float()
            iou = intersection / union.clamp_min(1.)
            coverage = intersection / target_count
            leakage = (predicted_count - intersection) / predicted_count
            depth_error = torch.abs(rendered - target_depth_t[None, :])
            depth_error = torch.where(
                overlap, depth_error.clamp(max=.10 * depth_span),
                torch.zeros_like(depth_error))
            normalized_depth = (
                depth_error.sum(dim=1) / intersection.clamp_min(1.) / depth_span)
            score = iou + .15 * coverage - .45 * leakage - .20 * normalized_depth
            for local_id, item in enumerate(chunk):
                ranking.append({
                    **item,
                    "coarse_hypothesis_id": start + local_id,
                    "coarse_metrics": {
                        "score": float(score[local_id].cpu()),
                        "iou": float(iou[local_id].cpu()),
                        "coverage": float(coverage[local_id].cpu()),
                        "leakage": float(leakage[local_id].cpu()),
                        "normalized_visible_depth": float(
                            normalized_depth[local_id].cpu()),
                    },
                })
    ranking.sort(key=lambda item: item["coarse_metrics"]["score"], reverse=True)
    return ranking


def process(args, sample):
    started = time.perf_counter()
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
        image_shape=(512, 512), device=args.device)
    lattice = v10.rotation_lattice(source_full, partial_full, args.offset_degrees)
    coarse_source = base.subset(source_full, args.coarse_source_points)
    coarse_partial = base.subset(partial_full, args.coarse_partial_points)
    coarse_started = time.perf_counter()
    coarse = batched_coarse_rank(
        coarse_source, coarse_partial, lattice, args.coarse_scales, projector,
        args.coarse_render_size, args.coarse_batch_size,
        args.coarse_splat_radius, args.device)
    coarse_seconds = time.perf_counter() - coarse_started
    print(f"{sample} batched coarse {len(coarse)} hypotheses in "
          f"{coarse_seconds:.2f}s", flush=True)

    fine_source = base.subset(source_full, args.fine_source_points)
    fine_partial = base.subset(partial_full, args.fine_partial_points)
    fine_mask = base.render_mask(projector, partial_full, args.fine_render_size)
    fine = []
    fine_started = time.perf_counter()
    for coarse_rank, item in enumerate(coarse[:args.fine_candidates], start=1):
        nearby_scales = sorted(set(args.fine_scales + [float(item["scale"])]))
        scale, translation, metrics, trace = v10.optimize_scale_translation(
            fine_source, fine_partial, item["rotation"], nearby_scales,
            projector, fine_mask, diagonal, args.fine_render_size,
            args.scale_translation_levels, args.min_scale, args.max_scale)
        fine.append({**item, "coarse_rank": coarse_rank, "scale": scale,
                     "translation": translation, "metrics": metrics,
                     "scale_translation_trace": trace})
        print(f"{sample} fine {coarse_rank}/{args.fine_candidates} "
              f"score={metrics['score']:.6f} iou={metrics['iou']:.4f}", flush=True)
    fine.sort(key=lambda item: item["metrics"]["score"], reverse=True)
    fine_seconds = time.perf_counter() - fine_started

    refined = []
    refine_started = time.perf_counter()
    refine_count = min(args.rotation_refine_candidates, len(fine))
    for fine_rank, item in enumerate(fine[:refine_count], start=1):
        rotation, scale, translation, metrics, trace = local.refine(
            fine_source, fine_partial, item["rotation"], item["scale"],
            item["translation"], projector, fine_mask, diagonal,
            args.fine_render_size, args.rotation_levels)
        refined.append({**item, "fine_rank": fine_rank, "rotation": rotation,
                        "scale": scale, "translation": translation,
                        "metrics": metrics, "rotation_trace": trace})
        print(f"{sample} refine {fine_rank}/{refine_count} "
              f"score={metrics['score']:.6f} iou={metrics['iou']:.4f}", flush=True)
    refined.sort(key=lambda item: item["metrics"]["score"], reverse=True)
    refine_seconds = time.perf_counter() - refine_started
    selected = refined[0]

    transform = base.make_transform(
        selected["rotation"], selected["scale"], selected["translation"])
    registered = base.apply_sim3(
        source_full, selected["rotation"], selected["scale"],
        selected["translation"])
    stem = output_dir / f"{sample}_batched_so3_sim3_ttt_v11"
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
    elapsed = time.perf_counter() - started
    result = {
        "sample_id": sample,
        "method": "batched_coarse_so3_lattice_proper_sim3_ttt_v11",
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "category_specific_parameters": False,
        "sample_specific_parameters": False,
        "fusion_run": False,
        "selected": selected,
        "coarse_ranking": [
            {"rank": i + 1, **x}
            for i, x in enumerate(coarse[:args.saved_coarse_candidates])],
        "fine_ranking": [
            {"rank": i + 1, **x} for i, x in enumerate(fine)],
        "refined_ranking": [
            {"rank": i + 1, **x} for i, x in enumerate(refined)],
        "full_resolution_projection": full_metrics,
        "transform": transform,
        "timing_seconds": {"coarse": coarse_seconds, "fine": fine_seconds,
                           "rotation_refine": refine_seconds, "total": elapsed},
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
        "sample": sample, "timing_seconds": result["timing_seconds"],
        "metrics": selected["metrics"], "full_projection": full_metrics,
        "contract": result["transform_contract"], "outputs": paths,
    }), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=["06145", "06830"])
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--coarse-batch-size", type=int, default=64)
    parser.add_argument("--coarse-splat-radius", type=int, default=1)
    parser.add_argument("--fine-candidates", type=int, default=12)
    parser.add_argument("--rotation-refine-candidates", type=int, default=8)
    parser.add_argument("--saved-coarse-candidates", type=int, default=48)
    parser.add_argument("--scale-translation-levels", type=int, default=5)
    parser.add_argument("--rotation-levels", type=int, default=5)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v11 batched coarse stage")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample in args.samples:
        process(args, str(sample))


if __name__ == "__main__":
    main()
