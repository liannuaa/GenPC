#!/usr/bin/env python3
"""Render an orbit atlas, then save a GPT-visually-selected partial camera.

The atlas is only an inspection surface for the agent; it is not scored or
ranked by geometry.  A subsequent invocation supplies the continuous azimuth
and elevation chosen by the visual agent and writes a Camera-1-compatible
camera, depth raster, and projection metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import kaolin as kal
import numpy as np
from PIL import Image, ImageDraw
import torch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DepthPrompting import DepthPrompting
from src.mainline_data import load_partial
from src.zbuffer import zbuffer_depth_with_indices


def view_direction(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth, elevation = map(math.radians, (azimuth_degrees, elevation_degrees))
    return np.asarray((
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.sin(azimuth),
    ), dtype=np.float64)


def camera_parameters(points: np.ndarray, azimuth_degrees: float,
                      elevation_degrees: float, fov_degrees: float) -> dict[str, np.ndarray | float]:
    lower, upper = points.min(axis=0), points.max(axis=0)
    target = (lower + upper) * .5
    extent = float(np.linalg.norm(upper - lower))
    distance = max(1.6 * extent, 1e-3)
    direction = view_direction(azimuth_degrees, elevation_degrees)
    eye = target + direction * distance
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    world_up = np.asarray((0., 1., 0.), dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > .98:
        world_up = np.asarray((0., 0., 1.), dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    return {
        "target": target, "eye": eye, "up": up, "direction": direction,
        "distance": distance, "fov_degrees": float(fov_degrees),
    }


def make_camera(parameters: dict[str, np.ndarray | float], resolution: int):
    return kal.render.camera.Camera.from_args(
        eye=torch.as_tensor(parameters["eye"], dtype=torch.float32),
        at=torch.as_tensor(parameters["target"], dtype=torch.float32),
        up=torch.as_tensor(parameters["up"], dtype=torch.float32),
        fov=math.radians(float(parameters["fov_degrees"])),
        width=int(resolution), height=int(resolution), device="cpu",
    )


def project(points: np.ndarray, parameters: dict[str, np.ndarray | float], resolution: int):
    camera = make_camera(parameters, resolution)
    with torch.no_grad():
        camera_points = camera.transform(torch.as_tensor(points, dtype=torch.float32)).cpu().numpy()
    xy = camera_points[:, :2]
    center = (xy.min(axis=0) + xy.max(axis=0)) * .5
    scale = max(float(np.ptp(xy, axis=0).max()), 1e-8)
    uv = (xy - center) / scale * .70 + .5
    uv[:, 1] = 1. - uv[:, 1]
    return camera, uv, camera_points[:, 2], center, scale


def render_depth(points: np.ndarray, colors: np.ndarray,
                 parameters: dict[str, np.ndarray | float], resolution: int):
    """Reuse the frozen mainline DepthPrompting sparse-depth raster exactly."""
    _, uv, depth, center, scale = project(points, parameters, resolution)
    pixels = torch.as_tensor(np.clip(uv * float(resolution), 0., float(resolution - 1)), dtype=torch.long)
    pixels = torch.stack((pixels[:, 1], pixels[:, 0]), dim=1)
    # `_rasterise_depth` reads only these three frozen image parameters.  It
    # paints one depth pixel per observation and applies OpenCV inpainting only
    # to the five-by-five local ring around it; empty background stays empty.
    rasterizer = object.__new__(DepthPrompting)
    rasterizer.cfg = SimpleNamespace(res=int(resolution), point_size=1, mask_pixel_rate=3)
    rasterizer.device = torch.device("cpu")
    sparse_depth, hole_mask = rasterizer._rasterise_depth(
        pixels, torch.as_tensor(depth, dtype=torch.float32), torch.as_tensor(colors, dtype=torch.float32)
    )
    raw = (sparse_depth.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    inpaint_mask = (hole_mask.permute(1, 2, 0).cpu().numpy()[..., 0] * 255).astype(np.uint8)
    depth_image = cv2.inpaint(raw, inpaint_mask, 2, cv2.INPAINT_NS)
    valid = raw[..., 0] > 0
    local_support = valid | (inpaint_mask > 0)
    return raw, depth_image, valid, local_support, inpaint_mask, uv, depth, center, scale


def atlas_cell(points: np.ndarray, azimuth: float, elevation: float, resolution: int) -> Image.Image:
    """Fast visual-only orbit render; it is never used for final depth export."""
    parameters = camera_parameters(points, azimuth, elevation, 49.1)
    _, uv, depth, _, _ = project(points, parameters, resolution)
    rendered, visible, _ = zbuffer_depth_with_indices(
        uv * float(resolution - 1), depth, (resolution, resolution), splat_radius=1,
    )
    rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    if visible.any():
        values = rendered[visible]
        shade = np.rint(255. * (1. - (values - values.min()) / max(float(np.ptp(values)), 1e-8))).astype(np.uint8)
        rgb[visible] = np.stack((shade, np.minimum(255, shade + 35), 255 - shade // 3), axis=1)
    cell = Image.fromarray(rgb)
    ImageDraw.Draw(cell).rectangle((0, 0, resolution - 1, 20), fill=(0, 0, 0))
    ImageDraw.Draw(cell).text((4, 4), f"az {azimuth:5.1f}  el {elevation:4.1f}", fill=(255, 255, 255))
    return cell


def write_orbit_atlas(points: np.ndarray, output: Path, resolution: int) -> None:
    azimuths = tuple(float(value) for value in range(0, 360, 45))
    elevations = (-20., 10., 35.)
    sheet = Image.new("RGB", (len(azimuths) * resolution, len(elevations) * resolution), color=(0, 0, 0))
    for row, elevation in enumerate(elevations):
        for column, azimuth in enumerate(azimuths):
            sheet.paste(atlas_cell(points, azimuth, elevation, resolution), (column * resolution, row * resolution))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--azimuth-degrees", type=float)
    parser.add_argument("--elevation-degrees", type=float)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--atlas-resolution", type=int, default=192)
    parser.add_argument("--fov-degrees", type=float, default=49.1)
    parser.add_argument("--render-atlas-only", action="store_true")
    parser.add_argument("--reuse-atlas", action="store_true",
                        help="Reuse an existing visual-review atlas when exporting the selected view.")
    parser.add_argument("--rationale", default="")
    args = parser.parse_args()
    if args.resolution < 32 or args.atlas_resolution < 64:
        raise ValueError("image resolutions are too small")
    if args.render_atlas_only == (args.azimuth_degrees is not None or args.elevation_degrees is not None):
        raise ValueError("use --render-atlas-only, or specify both selected view angles")
    if args.azimuth_degrees is None and not args.render_atlas_only:
        raise ValueError("both --azimuth-degrees and --elevation-degrees are required")
    if args.azimuth_degrees is not None and args.elevation_degrees is None:
        raise ValueError("both --azimuth-degrees and --elevation-degrees are required")

    partial = args.partial.resolve()
    output = args.output_dir.resolve()
    points, colors = load_partial(partial)
    output.mkdir(parents=True, exist_ok=True)
    atlas = output / "gpt_orbit_atlas.png"
    if args.reuse_atlas:
        if not atlas.is_file():
            raise FileNotFoundError(f"--reuse-atlas requires {atlas}")
    else:
        write_orbit_atlas(points, atlas, int(args.atlas_resolution))
    if args.render_atlas_only:
        print(json.dumps({"partial": str(partial), "atlas": str(atlas), "selection": "pending_gpt_visual_review"}, indent=2))
        return

    parameters = camera_parameters(points, float(args.azimuth_degrees), float(args.elevation_degrees), float(args.fov_degrees))
    camera, uv, _, xy_center, xy_scale = project(points, parameters, int(args.resolution))
    raw, depth, valid, local_support, inpaint_mask, _, point_depth, _, _ = render_depth(
        points, colors, parameters, int(args.resolution)
    )
    Image.fromarray(raw).save(output / "raw_depth.png")
    Image.fromarray(depth).save(output / "depth.png")
    Image.fromarray(inpaint_mask).save(output / "mask.png")
    Image.fromarray((local_support * 255).astype(np.uint8)).save(output / "support_mask.png")
    np.save(output / "depth.npy", (depth[..., 0].astype(np.float32) / 255.))
    np.save(output / "camera_point_depth.npy", point_depth.astype(np.float32))
    np.save(output / "point_uv.npy", uv.astype(np.float32))
    np.save(output / "viewpoint.npy", np.asarray(parameters["eye"], dtype=np.float32))
    np.save(output / "target.npy", np.asarray(parameters["target"], dtype=np.float32))
    torch.save(camera, output / "camera.pth")
    record = {
        "partial": str(partial),
        "selection_source": "GPT-5.6 Terra visual inspection of gpt_orbit_atlas.png; no geometry-score ranking",
        "rationale": str(args.rationale),
        "azimuth_degrees": float(args.azimuth_degrees) % 360.,
        "elevation_degrees": float(args.elevation_degrees),
        "eye": np.asarray(parameters["eye"]).tolist(),
        "target": np.asarray(parameters["target"]).tolist(),
        "up": np.asarray(parameters["up"]).tolist(),
        "view_direction": np.asarray(parameters["direction"]).tolist(),
        "camera_distance": float(parameters["distance"]),
        "fov_degrees": float(args.fov_degrees),
        "image_size": int(args.resolution),
        "projection_normalization": {"xy_center_camera": xy_center.tolist(), "xy_scale_camera": float(xy_scale), "frame_fill": .70},
        "visible_pixels": int(valid.sum()),
        "local_support_pixels": int(local_support.sum()),
        "inpaint_pixels": int((inpaint_mask > 0).sum()),
        "source_points": int(len(points)),
    }
    (output / "semantic_view_selection.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"atlas": str(atlas), "output": str(output), **record}, indent=2))


if __name__ == "__main__":
    main()
