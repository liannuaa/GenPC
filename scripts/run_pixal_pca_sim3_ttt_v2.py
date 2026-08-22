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
        image_shape=(512, 512), device="cpu")
    source = subset(source_full, args.source_points)
    partial = subset(partial_full, args.partial_points)
    target_mask = render_mask(projector, partial_full, args.render_size)
    candidates = []
    for rotation_id, item in enumerate(proper_pca_rotations(source_full, partial_full)):
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
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    infos = [process(args, str(sample)) for sample in args.samples]
    print(json.dumps(jsonable({"outputs": [x["outputs"] for x in infos],
                               "fusion_run": False}), indent=2))


if __name__ == "__main__": main()
