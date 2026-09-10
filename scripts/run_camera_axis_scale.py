#!/usr/bin/env python3
"""Estimate one Camera-1-supported global relative-axis scale candidate.

The input is an already registered complete carrier. The transform is applied
to every carrier slot about a robust visible-prior centre, so it preserves the
complete object's connectivity. Proper Sim(3) continues to own global pose,
translation, and uniform scale; the subsequent Gaussian stage remains local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.partial_anchored_gaussian_edit import (
    visible_axis_stretch, visible_depth_extent_calibration, visible_principal_extent_calibration,
)
from src.pointcloud_io import jsonable, load_points, write_points
from src.saved_camera import SavedCameraProjector, world_to_camera_axes
from src.visible_pixel_sim3_refinement import visible_pixel_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixel-distance", type=float, default=1.0)
    parser.add_argument("--fit-mode", choices=("matched_relative_axis", "visible_depth_extent", "visible_principal_extent"),
                        default="matched_relative_axis")
    parser.add_argument("--max-anchor-residual-ratio", type=float, default=.075)
    parser.add_argument("--max-log-stretch", type=float, default=.16)
    parser.add_argument("--minimum-pairs", type=int, default=512)
    parser.add_argument("--minimum-anisotropy", type=float, default=.025)
    parser.add_argument("--minimum-residual-reduction", type=float, default=.03)
    parser.add_argument("--trim-quantile", type=float, default=.75,
                        help="Per-iteration visible-fit residual quantile; 1 retains all cap-bounded pairs.")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of deterministic visible-fit reweighting iterations.")
    parser.add_argument("--retain-isotropic-component", action="store_true",
                        help="Keep the fully supported uniform scale component as a whole-carrier proposal.")
    parser.add_argument("--extent-quantile", type=float, default=.01)
    parser.add_argument("--extent-minimum-visible-points", type=int, default=512)
    parser.add_argument("--extent-minimum-log-expansion", type=float, default=.025)
    parser.add_argument("--extent-max-log-expansion", type=float, default=.25)
    parser.add_argument("--principal-minimum-axis-anisotropy", type=float, default=1.15)
    parser.add_argument("--principal-anchor-tolerance", type=float, default=.03)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.max_anchor_residual_ratio, args.max_log_stretch, args.minimum_anisotropy,
           args.minimum_residual_reduction, args.padding) <= 0. or args.minimum_pairs < 6:
        raise ValueError("camera-axis scale bounds and support requirements must be positive")
    if not 0. < args.trim_quantile <= 1. or args.iterations < 1:
        raise ValueError("--trim-quantile must lie in (0, 1] and --iterations must be positive")

    prior, partial = load_points(args.prior), load_points(args.partial)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    pairing = None
    if args.fit_mode == "matched_relative_axis":
        pixel_pairs, pairing = visible_pixel_pairs(
            partial, prior, projector, max_pixel_distance=float(args.max_pixel_distance),
            max_pairs=min(len(partial), len(prior)),
        )
        displacement, estimate = visible_axis_stretch(
            prior, partial, pixel_pairs,
            max_anchor_residual=float(args.max_anchor_residual_ratio) * diagonal,
            max_log_stretch=float(args.max_log_stretch), minimum_pairs=int(args.minimum_pairs),
            minimum_anisotropy=float(args.minimum_anisotropy),
            minimum_residual_reduction=float(args.minimum_residual_reduction),
            axis_basis=world_to_camera_axes(projector.camera),
            trim_quantile=float(args.trim_quantile),
            iterations=int(args.iterations),
            retain_isotropic_component=bool(args.retain_isotropic_component),
        )
    elif args.fit_mode == "visible_depth_extent":
        displacement, estimate = visible_depth_extent_calibration(
            prior, partial, projector, axis_basis=world_to_camera_axes(projector.camera),
            quantile=float(args.extent_quantile),
            minimum_visible_points=int(args.extent_minimum_visible_points),
            minimum_log_expansion=float(args.extent_minimum_log_expansion),
            max_log_expansion=float(args.extent_max_log_expansion),
        )
    else:
        displacement, estimate = visible_principal_extent_calibration(
            prior, partial, projector, quantile=float(args.extent_quantile),
            minimum_visible_points=int(args.extent_minimum_visible_points),
            minimum_axis_anisotropy=float(args.principal_minimum_axis_anisotropy),
            anchor_tolerance=float(args.principal_anchor_tolerance),
            minimum_log_expansion=float(args.extent_minimum_log_expansion),
            max_log_expansion=float(args.extent_max_log_expansion),
        )
    scaled = prior + displacement if bool(estimate.get("active", False)) else prior.copy()
    if scaled.shape != prior.shape or not np.isfinite(scaled).all():
        raise ValueError("camera-axis scale produced an invalid carrier")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "camera_axis_scaled_prior_100k.ply"
    write_points(output, scaled)
    record = {
        "method": "camera1_supported_global_relative_axis_scale",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                   "camera": str(args.camera.resolve())},
        "fit_mode": str(args.fit_mode), "pairing": pairing,
        "estimate": estimate,
        "parameters": {"max_pixel_distance": float(args.max_pixel_distance),
                       "max_anchor_residual_ratio": float(args.max_anchor_residual_ratio),
                       "max_log_stretch": float(args.max_log_stretch),
                       "minimum_pairs": int(args.minimum_pairs),
                       "minimum_anisotropy": float(args.minimum_anisotropy),
                       "minimum_residual_reduction": float(args.minimum_residual_reduction),
                       "trim_quantile": float(args.trim_quantile),
                       "iterations": int(args.iterations),
                       "retain_isotropic_component": bool(args.retain_isotropic_component),
                       "extent_quantile": float(args.extent_quantile),
                       "extent_minimum_visible_points": int(args.extent_minimum_visible_points),
                       "extent_minimum_log_expansion": float(args.extent_minimum_log_expansion),
                       "extent_max_log_expansion": float(args.extent_max_log_expansion),
                       "principal_minimum_axis_anisotropy": float(args.principal_minimum_axis_anisotropy),
                       "principal_anchor_tolerance": float(args.principal_anchor_tolerance)},
        "output": str(output.resolve()),
        "carrier_slots_preserved": bool(len(scaled) == len(prior)),
    }
    (args.output_dir / "camera_axis_scale_info.json").write_text(
        json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve()), "estimate": estimate}, indent=2))


if __name__ == "__main__":
    main()
