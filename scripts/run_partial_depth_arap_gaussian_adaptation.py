#!/usr/bin/env python3
"""Run the isolated partial-depth ARAP Gaussian adaptation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_camera_conditioned_gaussian_adaptation import save_camera_status_overlay
from src.arap_gaussian_reallocation import partial_depth_arap_gaussian_adaptation
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--virtual-views", type=int, default=0,
                        help="Additional positive-only PCA partial self-views (0 disables them).")
    parser.add_argument("--virtual-resolution", type=int, default=384)
    args = parser.parse_args()
    prior, partial = load_points(args.prior), load_points(args.partial)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    edited, estimate, masks = partial_depth_arap_gaussian_adaptation(
        prior, partial, projector, virtual_views=int(args.virtual_views),
        virtual_resolution=int(args.virtual_resolution),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "partial_depth_arap_gaussian"
    write_points(Path(f"{stem}_prior_100k.ply"), edited)
    write_compare(Path(f"{stem}_partial_gray_prior_red.ply"), partial, edited)
    draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, edited, projector)
    status_masks = {
        "protected": ~masks["influence"], "moved": masks["moved"],
        "locked": masks["stable"] | masks["boundary"], "editable": masks["anchors"],
    }
    status_path = Path(f"{stem}_status_projection.png")
    save_camera_status_overlay(status_path, args.semantic, prior, status_masks, projector)
    np.savez_compressed(Path(f"{stem}_field.npz"), means=edited.astype(np.float32), **masks)
    record = {
        "method": "partial_depth_arap_gaussian_adaptation", "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                   "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve())},
        "estimate": estimate,
        "parameters": {"render_size": int(args.render_size), "padding": float(args.padding),
                       "virtual_views": int(args.virtual_views), "virtual_resolution": int(args.virtual_resolution)},
        "outputs": {"prior": str(Path(f"{stem}_prior_100k.ply").resolve()),
                    "comparison": str(Path(f"{stem}_partial_gray_prior_red.ply").resolve()),
                    "saved_view": str(Path(f"{stem}_saved_view_projection.png").resolve()),
                    "status_view": str(status_path.resolve()),
                    "field": str(Path(f"{stem}_field.npz").resolve())},
    }
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"estimate": estimate, "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
