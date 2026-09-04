#!/usr/bin/env python3
"""Register a Pixal prior in the exact MoGe camera used for Pixal input.

This deliberately runs MoGe on ``pixal3d_input.png`` rather than an upstream
semantic image.  Pixal3D estimated its input camera from this same prepared
image, so the resulting MoGe point map is the appropriate first-stage target.
The output remains in native MoGe camera coordinates; mapping to a scan and
any partial-only refinement are intentionally separate later stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from scripts.run_moge_pixel_index_bridge import (
    filter_moge_points_by_object_mask,
    prepare_object_mask,
    run_moge_with_pixels,
    run_rmbg_mask,
    save_mask_png,
)
from src.moge_camera import MoGeProjector
from src.pixal_moge_analytic_registration import (
    analytic_pixal_to_moge_initial,
    local_pixal_moge_refine,
    pixal_moge_render_score,
    visible_ray_depth_scale_step,
    visible_mask_translation_step,
)
from src.ray_consistent_registration import apply_transform


def native_pixal_moge_observation(
    image_path: Path,
    moge_model: Path,
    rmbg_model: Path,
    output_dir: Path,
    *,
    device: str,
    fp16: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Infer ``pixal3d_input.png``'s Pixal-compatible MoGe map.

    Pixal3D's ``get_camera_params_wild_moge`` uses precisely ``PIL RGB ->
    float/255 CHW -> MoGeModel.infer`` on its already preprocessed image.
    ``run_moge_with_pixels`` follows that same tensor contract while retaining
    pixels and foreground RGB needed by registration.  The only difference is
    that its checkpoint is the local, reproducible Pixal MoGe-2 weight.
    """
    points, colors, pixels, moge_info = run_moge_with_pixels(
        image_path=image_path, pretrained=moge_model, device=device, fp16=fp16
    )
    rgba_path = output_dir / "pixal_input_rmbg.png"
    alpha = run_rmbg_mask(image_path, rgba_path, rmbg_model)
    mask = prepare_object_mask(alpha, alpha_threshold=128, erode_pixels=0)
    save_mask_png(output_dir / "pixal_input_object_mask.png", mask)
    foreground = filter_moge_points_by_object_mask(
        points=points, colors=colors, pixel_xy=pixels, object_mask=mask,
        alpha_threshold=128, erode_pixels=0,
    )
    if len(foreground.points) < 96:
        raise RuntimeError("Pixal-input MoGe foreground has too few points")
    moge_info["foreground_points"] = int(len(foreground.points))
    moge_info["foreground_mask"] = str((output_dir / "pixal_input_object_mask.png").resolve())
    moge_info["rmbg_rgba"] = str(rgba_path.resolve())
    moge_info["pixal3d_moge_contract"] = {
        "input": "exact saved pixal3d_input.png after Pixal3D preprocess_image",
        "inference": "PIL RGB -> float/255 CHW -> MoGeModel.infer",
        "camera_consistency": "matches Pixal3D inference.py:get_camera_params_wild_moge",
    }
    return foreground.points, foreground.colors, moge_info


def load_points_and_colors(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    colors = np.asarray(cloud.colors, dtype=np.float64)
    return points, colors if colors.shape == points.shape else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--pixal-metadata", type=Path, required=True)
    parser.add_argument("--pixal-input", type=Path, required=True)
    parser.add_argument("--moge-model", type=Path)
    parser.add_argument("--rmbg-model", type=Path)
    parser.add_argument("--cached-moge", type=Path,
                        help="Optional foreground MoGe PLY from an earlier identical Pixal input")
    parser.add_argument("--cached-moge-info", type=Path,
                        help="Optional pixal_native_moge_info.json paired with --cached-moge")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior, prior_colors = load_points_and_colors(args.prior)
    if (args.cached_moge is None) != (args.cached_moge_info is None):
        parser.error("--cached-moge and --cached-moge-info must be supplied together")
    if args.cached_moge is not None:
        moge, moge_colors = load_points_and_colors(args.cached_moge)
        if moge_colors is None:
            raise ValueError("cached MoGe PLY must retain per-point semantic RGB")
        cached = json.loads(args.cached_moge_info.read_text(encoding="utf-8"))
        moge_info = cached.get("moge", cached)
        cache_mode = True
    else:
        if args.moge_model is None or args.rmbg_model is None:
            parser.error("--moge-model and --rmbg-model are required without cached MoGe")
        moge, moge_colors, moge_info = native_pixal_moge_observation(
            args.pixal_input, args.moge_model, args.rmbg_model, args.output_dir,
            device=args.device, fp16=bool(args.fp16),
        )
        cache_mode = False
    intrinsics = np.asarray(moge_info["output_keys"]["intrinsics"], dtype=np.float64)
    projector = MoGeProjector(intrinsics, tuple(moge_info["image_hw"]), device=args.device)
    pixal_metadata = json.loads(args.pixal_metadata.read_text(encoding="utf-8"))
    camera = pixal_metadata["camera"]
    analytic_initial = analytic_pixal_to_moge_initial(prior, moge, float(camera["distance"]))
    mask_translation_step, mask_translation_info = visible_mask_translation_step(
        moge, apply_transform(prior, analytic_initial), projector
    )
    initial = mask_translation_step @ analytic_initial
    registered, transform, trace = local_pixal_moge_refine(
        prior, moge, projector, initial, moge_colors=moge_colors, prior_colors=prior_colors
    )
    depth_step, depth_info = visible_ray_depth_scale_step(moge, registered, projector)
    depth_choices = []
    for fraction in (0.0, .5, .75, 1.0):
        candidate_step = np.eye(4, dtype=np.float64)
        candidate_step[:3, :3] *= 1.0 + float(fraction) * (float(depth_step[0, 0]) - 1.0)
        candidate = apply_transform(registered, candidate_step)
        score = pixal_moge_render_score(
            moge, candidate, projector, moge_colors=moge_colors, prior_colors=prior_colors
        )
        depth_choices.append((score, fraction, candidate_step, candidate))
    depth_score, selected_depth_fraction, selected_depth_step, registered = min(
        depth_choices, key=lambda item: item[0]["objective"]
    )
    depth_info["selected_fraction"] = selected_depth_fraction
    depth_info["selected_score"] = depth_score
    depth_info["candidate_scores"] = [
        {"fraction": fraction, "score": score} for score, fraction, _, _ in depth_choices
    ]
    transform = selected_depth_step @ transform
    stem = args.output_dir / "pixal_native_moge"
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(moge)
    target_cloud.colors = o3d.utility.Vector3dVector(moge_colors)
    o3d.io.write_point_cloud(str(Path(f"{stem}_points.ply")), target_cloud)
    base.write_points(Path(f"{stem}_registered_100k.ply"), registered)
    base.write_compare(Path(f"{stem}_gray_pixal_red.ply"), moge, registered)
    base.draw_projection_overlay(
        Path(f"{stem}_projection.png"), args.pixal_input, moge, registered, projector
    )
    record = {
        "method": "pixal_camera_anchored_native_moge_local_render_sim3",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "pixal_input": str(args.pixal_input.resolve()),
        "pixal_metadata": str(args.pixal_metadata.resolve()),
        "pixal_camera": camera,
        "prior": str(args.prior.resolve()),
        "moge": moge_info,
        "registration": trace, "analytic_initial": analytic_initial,
        "mask_translation": mask_translation_info, "camera_initialized": initial,
        "visible_ray_depth_scale": depth_info,
        "photometric_auxiliary": bool(prior_colors is not None and len(moge_colors) == len(moge)),
        "reused_cached_moge": cache_mode,
        "prior_to_native_moge": transform,
    }
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({
        "moge_points": int(len(moge)),
        "before": trace["before"]["objective"], "after": trace["after"]["objective"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
