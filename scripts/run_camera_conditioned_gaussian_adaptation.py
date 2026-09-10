#!/usr/bin/env python3
"""Run one isolated Camera-1 projection-and-depth Gaussian adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_conditioned_gaussian_adaptation import camera_conditioned_gaussian_adaptation
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay, world_to_camera_axes
from src.zbuffer import zbuffer_depth_with_indices


def save_camera_status_overlay(path: Path, semantic: Path, prior: np.ndarray,
                               masks: dict[str, np.ndarray], projector) -> None:
    """Render the no-GT Gaussian edit decision in the saved partial camera."""
    image = Image.open(semantic).convert("RGB")
    height, width = projector.image_shape
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    uv, depth = projector.project(prior)
    _, visible, indices = zbuffer_depth_with_indices(uv, depth, projector.image_shape, splat_radius=0)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    # Draw the mutually exclusive state hierarchy: selected controls are
    # orange, low-residual fixed evidence green, and propagated means blue.
    styles = (
        ("protected", (110, 110, 110, 60), 1),
        ("moved", (42, 130, 255, 150), 1),
        ("locked", (35, 220, 110, 210), 2),
        ("editable", (255, 175, 20, 255), 2),
    )
    yy, xx = np.where(visible)
    ids = indices[yy, xx]
    for name, colour, radius in styles:
        keep = masks[name][ids]
        for x, y in zip(xx[keep], yy[keep]):
            draw.ellipse((int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius), fill=colour)
    canvas.save(path)


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
    parser.add_argument("--data-weight", type=float, default=.25,
                        help="Soft Camera-1 projection-and-depth control weight.")
    parser.add_argument("--residual-ratio", type=float, default=.12,
                        help="Minimum partial-scale residual for an editable coherent component.")
    parser.add_argument("--stable-ratio", type=float, default=.045,
                        help="Maximum partial-scale residual for a locked visible Gaussian.")
    parser.add_argument("--max-pixel-distance", type=float, default=1.,
                        help="Maximum Camera-1 pixel gap when forming a partial/prior correspondence.")
    parser.add_argument("--minimum-component-pairs", type=int, default=128,
                        help="Minimum screen-connected partial/prior matches for one editable component.")
    parser.add_argument("--maximum-displacement-ratio", type=float, default=.30,
                        help="Per-stage Gaussian-mean displacement cap as a partial-diagonal ratio.")
    parser.add_argument("--locked-mask-field", type=Path,
                        help="Optional prior-stage NPZ whose 'locked' slots remain fixed in this stage.")
    args = parser.parse_args()
    prior, partial = load_points(args.prior), load_points(args.partial)
    inherited_locked = None
    if args.locked_mask_field is not None:
        state = np.load(args.locked_mask_field)
        if "locked" not in state:
            raise ValueError("--locked-mask-field must contain a 'locked' boolean array")
        inherited_locked = state["locked"].astype(bool)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    edited, estimate, masks = camera_conditioned_gaussian_adaptation(
        prior, partial, projector, camera_axes=world_to_camera_axes(projector.camera),
        data_weight=float(args.data_weight), residual_ratio=float(args.residual_ratio),
        stable_ratio=float(args.stable_ratio), max_pixel_distance=float(args.max_pixel_distance),
        minimum_component_pairs=int(args.minimum_component_pairs),
        maximum_displacement_ratio=float(args.maximum_displacement_ratio),
        locked_prior_mask=inherited_locked,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "camera_conditioned_gaussian"
    write_points(Path(f"{stem}_editable_prior_100k.ply"), edited)
    write_compare(Path(f"{stem}_partial_gray_prior_red.ply"), partial, edited)
    draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, edited, projector)
    status_path = Path(f"{stem}_status_projection.png")
    save_camera_status_overlay(status_path, args.semantic, prior, masks, projector)
    np.savez_compressed(Path(f"{stem}_field.npz"), means=edited.astype(np.float32), **masks)
    record = {
        "method": "camera_conditioned_projection_depth_gaussian_mean_adaptation",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                   "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve())},
        "estimate": estimate,
        "parameters": {"render_size": int(args.render_size), "padding": float(args.padding),
                       "data_weight": float(args.data_weight),
                       "residual_ratio": float(args.residual_ratio), "stable_ratio": float(args.stable_ratio),
                       "max_pixel_distance": float(args.max_pixel_distance),
                       "minimum_component_pairs": int(args.minimum_component_pairs),
                       "maximum_displacement_ratio": float(args.maximum_displacement_ratio),
                       "locked_mask_field": None if args.locked_mask_field is None else str(args.locked_mask_field.resolve())},
        "outputs": {
            "editable_prior": str(Path(f"{stem}_editable_prior_100k.ply").resolve()),
            "comparison": str(Path(f"{stem}_partial_gray_prior_red.ply").resolve()),
            "saved_view": str(Path(f"{stem}_saved_view_projection.png").resolve()),
            "status_view": str(status_path.resolve()),
            "field": str(Path(f"{stem}_field.npz").resolve()),
        },
    }
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"estimate": estimate, "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
