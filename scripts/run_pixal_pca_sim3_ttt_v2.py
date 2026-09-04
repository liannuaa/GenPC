#!/usr/bin/env python3
"""GT-free all-orientation isotropic Sim(3) TTT for frozen Pixal3D assets."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_registration_deformation_fusion_ablation import (  # noqa: E402
    SavedCameraProjector, draw_projection_overlay, load_points, write_compare,
    write_points,
)
from scripts.run_render_to_moge_sim3 import zbuffer_depth_with_indices  # noqa: E402

CAMERA_ROOT = ROOT / "workspace/redwood_onestage_rawdepth_512_stage2_20260714"
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_pca_sim3_ttt_v2_20260822"


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def subset(points, count):
    if len(points) <= count: return np.asarray(points)
    return np.asarray(points)[np.linspace(0, len(points) - 1, count, dtype=int)]


def pca_basis(points):
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    kept = points[radius <= np.quantile(radius, .98)]
    _, basis = np.linalg.eigh(np.cov(kept - np.median(kept, axis=0), rowvar=False))
    basis = basis[:, ::-1]
    if np.linalg.det(basis) < 0: basis[:, -1] *= -1
    return center, basis


def proper_pca_rotations(source, target):
    _, bs = pca_basis(source); _, bt = pca_basis(target)
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            p = np.zeros((3, 3)); p[np.arange(3), permutation] = signs
            rotation = bt @ p @ bs.T
            if np.linalg.det(rotation) > .999999:
                result.append({"permutation": permutation, "signs": signs,
                               "rotation": rotation})
    if len(result) != 24: raise RuntimeError(f"Expected 24 rotations, got {len(result)}")
    return result


def _pca_basis_gpu(points, device):
    """Return the robust PCA frame on CUDA without changing the 24-way contract."""
    values = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device)
    # ``np.median`` uses the interpolated middle value for an even population;
    # use the same convention so the CUDA orientation set matches the legacy
    # NumPy reference exactly enough for a no-regression shortlist.
    center = torch.quantile(values, .5, dim=0)
    radius = torch.linalg.vector_norm(values - center, dim=1)
    kept = values[radius <= torch.quantile(radius, .98)]
    # np.cov centers columns internally.  The preceding robust shift in the
    # reference implementation therefore has no effect on covariance; use the
    # arithmetic mean here, not the robust median.
    centered = kept - kept.mean(dim=0)
    covariance = centered.T @ centered / max(int(len(centered)) - 1, 1)
    _, basis = torch.linalg.eigh(covariance)
    basis = basis.flip(dims=(1,))
    if float(torch.linalg.det(basis)) < 0.:
        basis[:, -1] *= -1.
    return center, basis


def proper_pca_rotations_gpu(source, target, device):
    """GPU PCA followed by the same 24 proper signed axis permutations."""
    _, bs = _pca_basis_gpu(source, device)
    _, bt = _pca_basis_gpu(target, device)
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            permutation_matrix = torch.zeros((3, 3), dtype=torch.float32, device=device)
            permutation_matrix[torch.arange(3, device=device), list(permutation)] = torch.tensor(
                signs, dtype=torch.float32, device=device
            )
            rotation = bt @ permutation_matrix @ bs.T
            if float(torch.linalg.det(rotation)) > .999999:
                result.append({"permutation": permutation, "signs": signs,
                               "rotation": rotation.detach().cpu().numpy().astype(np.float64)})
    if len(result) != 24:
        raise RuntimeError(f"Expected 24 rotations, got {len(result)}")
    return result


def gpu_prescreen_rotation_ids(items, source, partial, projector, target_mask, args):
    """Batch the inexpensive saved-view initialization test on GPU.

    This only selects which PCA orientations receive the original CPU 2-D+3-D
    coordinate-descent refinement.  The final ranking therefore remains the
    exact historical objective, while the expensive 24-way exhaustive loop is
    reduced to a small, generic shortlist.
    """
    device = torch.device(args.gpu_device)
    source_t = torch.as_tensor(np.asarray(source), dtype=torch.float32, device=device)
    rotations = torch.as_tensor(np.stack([item["rotation"] for item in items]),
                                dtype=torch.float32, device=device)
    source_center = source_t.median(dim=0).values
    target_center = torch.as_tensor(np.median(partial, axis=0), dtype=torch.float32, device=device)
    scales = torch.as_tensor(args.initial_scales, dtype=torch.float32, device=device)
    size = int(args.render_size)
    target = torch.as_tensor(target_mask.reshape(-1), dtype=torch.bool, device=device)
    target_count = target.sum().float().clamp_min(1.)
    center_xy = torch.as_tensor(projector.center_xy, dtype=torch.float32, device=device)

    def scores(rotations_b, scales_b, translations_b):
        """Hard saved-view silhouette score for a batch of candidate Sim(3)s."""
        batch = len(scales_b)
        moved = (scales_b[:, None, None]
                 * torch.matmul(source_t.unsqueeze(0), rotations_b.transpose(1, 2))
                 + translations_b[:, None, :])
        camera = projector.camera.transform(moved.reshape(-1, 3))
        uv = (camera[:, :2] - center_xy) / float(projector.scale_xy)
        uv = uv * (1. - 2. * float(projector.padding)) + .5
        uv[:, 1] = 1. - uv[:, 1]
        xy = torch.round(uv * float(size - 1)).long()
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
                 & torch.isfinite(camera[:, 2]) & (camera[:, 2] > 1e-8))
        trial_id = torch.arange(batch, device=device).repeat_interleave(len(source_t))
        flat_pixel = trial_id[valid] * (size * size) + xy[valid, 1] * size + xy[valid, 0]
        occupancy = torch.zeros(batch * size * size, dtype=torch.int32, device=device)
        occupancy.scatter_add_(0, flat_pixel, torch.ones_like(flat_pixel, dtype=torch.int32))
        predicted = occupancy.reshape(batch, size * size) > 0
        intersection = (predicted & target).sum(dim=1).float()
        predicted_count = predicted.sum(dim=1).float()
        iou = intersection / (predicted_count + target_count - intersection).clamp_min(1.)
        coverage = intersection / target_count
        leakage = (predicted_count - intersection) / predicted_count.clamp_min(1.)
        return iou + .15 * coverage - .45 * leakage

    # Same multi-scale centroid initialization as the exact CPU loop.
    rotation_ids = torch.arange(len(items), device=device).repeat_interleave(len(scales))
    trial_scales = scales.repeat(len(items))
    trial_rotations = rotations[rotation_ids]
    trial_translation = target_center.unsqueeze(0) - trial_scales[:, None] * torch.einsum(
        "bij,j->bi", trial_rotations, source_center
    )
    initial = scores(trial_rotations, trial_scales, trial_translation).reshape(len(items), len(scales))
    scale_ids = initial.argmax(dim=1)
    current_scale = scales[scale_ids]
    current_translation = target_center.unsqueeze(0) - current_scale[:, None] * torch.einsum(
        "bij,j->bi", rotations, source_center
    )
    current_score = initial.gather(1, scale_ids[:, None]).squeeze(1)

    # Preserve the original coordinate-search schedule, but score all 24x8
    # proposals in one CUDA pass at every step.  This does not select a final
    # transform; it only makes the shortlist robust to large initial offsets.
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    scale_step, translation_step = .10, .09 * diagonal
    candidate_ids = torch.arange(len(items), device=device)
    for _ in range(int(args.levels)):
        for _ in range(8):
            proposal_scale = current_scale[:, None].repeat(1, 8)
            proposal_translation = current_translation[:, None, :].repeat(1, 8, 1)
            proposal_scale[:, 0] += scale_step
            proposal_scale[:, 1] -= scale_step
            for axis in range(3):
                proposal_translation[:, 2 + 2 * axis, axis] += translation_step
                proposal_translation[:, 3 + 2 * axis, axis] -= translation_step
            flat_scale = proposal_scale.reshape(-1)
            flat_translation = proposal_translation.reshape(-1, 3)
            proposal_scores = scores(rotations.repeat_interleave(8, dim=0), flat_scale, flat_translation)
            proposal_scores = proposal_scores.reshape(len(items), 8)
            proposal_scores[(proposal_scale < .45) | (proposal_scale > 1.20)] = -torch.inf
            best_score, best_coordinate = proposal_scores.max(dim=1)
            improve = best_score > current_score + 1e-10
            if not bool(improve.any()):
                break
            chosen_scale = proposal_scale.gather(1, best_coordinate[:, None]).squeeze(1)
            chosen_translation = proposal_translation[
                candidate_ids, best_coordinate
            ]
            current_scale = torch.where(improve, chosen_scale, current_scale)
            current_translation = torch.where(improve[:, None], chosen_translation, current_translation)
            current_score = torch.where(improve, best_score, current_score)
        scale_step *= .5
        translation_step *= .5

    best = current_score
    order = torch.argsort(best, descending=True)
    keep = order[:min(int(args.gpu_topk), len(items))].detach().cpu().tolist()
    detail = {"enabled": True, "device": str(device), "shortlist_size": len(keep),
              "rotation_scores": best.detach().cpu().numpy(), "selected_rotation_ids": keep,
              "screen_coordinate_descent": {"levels": int(args.levels), "steps_per_level": 8}}
    return keep, detail


def make_transform(rotation, scale, translation):
    transform = np.eye(4); transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def apply_sim3(points, rotation, scale, translation):
    return scale * (points @ rotation.T) + translation


def render_mask(projector, points, size, splat=1):
    pixel, depth = projector.project(points)
    h, w = projector.image_shape
    pixel[:, 0] *= (size - 1) / max(w - 1, 1)
    pixel[:, 1] *= (size - 1) / max(h - 1, 1)
    return zbuffer_depth_with_indices(pixel, depth, (size, size),
                                      splat_radius=splat)[1]


def mask_metrics(target, predicted):
    intersection = np.count_nonzero(target & predicted)
    union = np.count_nonzero(target | predicted)
    nt = np.count_nonzero(target); npred = np.count_nonzero(predicted)
    return {"iou": float(intersection / max(union, 1)),
            "coverage": float(intersection / max(nt, 1)),
            "leakage": float((npred - intersection) / max(npred, 1))}


def evaluate(source, partial, rotation, scale, translation, projector,
             target_mask, diagonal, render_size):
    moved = apply_sim3(source, rotation, scale, translation)
    metrics = mask_metrics(target_mask, render_mask(projector, moved, render_size))
    distances = cKDTree(moved).query(partial, workers=-1)[0]
    q70 = float(np.quantile(distances, .70))
    trim70 = float(distances[distances <= q70].mean())
    normalized = trim70 / diagonal
    score = (metrics["iou"] + .15 * metrics["coverage"]
             - .45 * metrics["leakage"] - 8. * normalized)
    return {"score": float(score), **metrics,
            "partial_to_complete_trim70": trim70,
            "partial_to_complete_q70": q70,
            "normalized_surface_trim70": normalized}


def optimize(source, partial, rotation, projector, target_mask, diagonal, args):
    cs = np.median(source, axis=0); ct = np.median(partial, axis=0)
    trials = []
    for scale in args.initial_scales:
        translation = ct - scale * (rotation @ cs)
        value = evaluate(source, partial, rotation, scale, translation, projector,
                         target_mask, diagonal, args.render_size)
        trials.append((value["score"], scale, translation, value))
    _, scale, translation, value = max(trials, key=lambda x: x[0])
    scale_step, translation_step, trace = .10, .09 * diagonal, []
    for level in range(args.levels):
        for _ in range(8):
            proposals = []
            for sign in (-1., 1.):
                candidate_scale = scale + sign * scale_step
                if .45 <= candidate_scale <= 1.20:
                    proposals.append((candidate_scale, translation.copy(), "scale"))
                for axis in range(3):
                    candidate_translation = translation.copy()
                    candidate_translation[axis] += sign * translation_step
                    proposals.append((scale, candidate_translation, f"t{axis}"))
            best = (value["score"], scale, translation, value, None)
            for candidate_scale, candidate_translation, coordinate in proposals:
                candidate = evaluate(source, partial, rotation, candidate_scale,
                                     candidate_translation, projector, target_mask,
                                     diagonal, args.render_size)
                if candidate["score"] > best[0]:
                    best = (candidate["score"], candidate_scale,
                            candidate_translation, candidate, coordinate)
            if best[0] <= value["score"] + 1e-10: break
            _, scale, translation, value, coordinate = best
            trace.append({"level": level, "coordinate": coordinate, "scale": scale,
                          "translation": translation.copy(), "metrics": value})
        scale_step *= .5; translation_step *= .5
    return scale, translation, value, trace


def process(args, sample):
    sample_dir = args.gpt_root / sample; camera_dir = args.camera_root / sample
    output_dir = args.output_root / sample; output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = ROOT / "data" / f"{sample}.ply"
    pixal_path = sample_dir / "pixal3d_sampled_100k.ply"
    partial_full = load_points(partial_path); source_full = load_points(pixal_path)
    if len(source_full) != 100000: raise ValueError("Frozen Pixal PLY is not 100k")
    diagonal = max(float(np.linalg.norm(np.ptp(partial_full, axis=0))), 1e-8)
    projector = SavedCameraProjector.from_partial(
        partial_full, camera_dir / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device=args.gpu_device if args.gpu_prescreen else "cpu")
    source = subset(source_full, args.source_points)
    partial = subset(partial_full, args.partial_points)
    target_mask = render_mask(projector, partial_full, args.render_size)
    items = (proper_pca_rotations_gpu(source_full, partial_full, args.gpu_device)
             if args.gpu_prescreen else proper_pca_rotations(source_full, partial_full))
    if args.gpu_prescreen:
        shortlisted_ids, prescreen = gpu_prescreen_rotation_ids(
            items, source, partial_full, projector, target_mask, args
        )
    else:
        shortlisted_ids, prescreen = list(range(len(items))), {"enabled": False}
    candidates = []
    for rotation_id in shortlisted_ids:
        item = items[rotation_id]
        scale, translation, metrics, trace = optimize(
            source, partial, item["rotation"], projector, target_mask, diagonal, args)
        candidate = {"rotation_id": rotation_id, **item, "scale": scale,
                     "translation": translation, "metrics": metrics,
                     "optimization_updates": trace}
        candidates.append(candidate)
        print(f"{sample} r={rotation_id:02d} score={metrics['score']:.6f} "
              f"iou={metrics['iou']:.4f} trim70={metrics['partial_to_complete_trim70']:.6f}",
              flush=True)
    ranked = sorted(candidates, key=lambda x: x["metrics"]["score"], reverse=True)
    selected = ranked[0]
    transform = make_transform(selected["rotation"], selected["scale"],
                               selected["translation"])
    registered = apply_sim3(source_full, selected["rotation"], selected["scale"],
                            selected["translation"])
    stem = output_dir / f"{sample}_pca_sim3_ttt_v2"
    paths = {"registered": Path(f"{stem}_registered_100k.ply"),
             "compare": Path(f"{stem}_partial_gray_pixal_red.ply"),
             "transform": Path(f"{stem}.npy"),
             "mesh": Path(f"{stem}_registered_mesh.glb"),
             "projection": Path(f"{stem}_projection.png")}
    write_points(paths["registered"], registered)
    write_compare(paths["compare"], partial_full, registered)
    np.save(paths["transform"], transform)
    mesh = trimesh.load(sample_dir / "pixal3d.glb", force="scene", process=False)
    mesh.apply_transform(transform); mesh.export(paths["mesh"])
    draw_projection_overlay(paths["projection"], camera_dir / "img.png",
                            partial_full, registered, projector)
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    full_metrics = mask_metrics(render_mask(projector, partial_full, 512),
                                render_mask(projector, registered, 512))
    info = {"sample_id": sample,
            "method": "all24_pca_orientation_uniform_sim3_2d3d_ttt_v2",
            "strict_zero_shot": True,
            "ground_truth_used_for_inference_or_selection": False,
            "category_specific_parameters": False,
            "sample_specific_parameters": False, "fusion_run": False,
            "selected_rotation_id": selected["rotation_id"],
            "selected_transform": transform,
            "selected_low_resolution_metrics": selected["metrics"],
            "selected_full_resolution_projection": full_metrics,
            "transform_contract": {"singular_values": singular,
                "isotropic_scale": float(singular.mean()),
                "proper_rotation": bool(np.linalg.det(selected["rotation"]) > .999999),
                "all_100k_pixal_points_preserved": len(registered) == 100000,
                "nonrigid_deformation_used": False, "anisotropic_scale_used": False},
            "ranking": [{"rank": i + 1, **item} for i, item in enumerate(ranked)],
            "gpu_prescreen": prescreen,
            "shared_parameters": vars(args), "outputs": paths}
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(info), indent=2))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=CAMERA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=["07136"])
    parser.add_argument("--initial-scales", nargs="*", type=float,
                        default=[.5, .6, .7, .8, .9, 1., 1.1])
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--render-size", type=int, default=128)
    parser.add_argument("--source-points", type=int, default=12000)
    parser.add_argument("--partial-points", type=int, default=6000)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--gpu-prescreen", action="store_true",
                        help="Use CUDA PCA and batch saved-view shortlist before exact refinement.")
    parser.add_argument("--gpu-topk", type=int, default=8,
                        help="Number of generic PCA orientations retained after the GPU prescreen.")
    parser.add_argument("--gpu-device", default="cuda")
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    infos = [process(args, str(sample)) for sample in args.samples]
    print(json.dumps(jsonable({"outputs": [x["outputs"] for x in infos],
                               "fusion_run": False}), indent=2))


if __name__ == "__main__": main()
