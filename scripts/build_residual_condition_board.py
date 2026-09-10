#!/usr/bin/env python3
"""Build a category-independent multi-view prior/partial residual board."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pointcloud_io import load_points
from src.zbuffer import zbuffer_depth_with_indices


def _project(points: np.ndarray, pose: np.ndarray, fov_degrees: float, resolution: int):
    inverse = np.linalg.inv(pose)
    camera = points @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -camera[:, 2]
    focal = 0.5 * resolution / math.tan(math.radians(fov_degrees) * 0.5)
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = focal * camera[:, 0] / np.maximum(depth, 1e-8) + resolution * 0.5
    uv[:, 1] = -focal * camera[:, 1] / np.maximum(depth, 1e-8) + resolution * 0.5
    return uv, depth


def _depth_panel(uv, depth, resolution: int) -> np.ndarray:
    rendered, mask, _ = zbuffer_depth_with_indices(
        uv, depth, (resolution, resolution), splat_radius=2,
    )
    panel = np.full((resolution, resolution, 3), 18, dtype=np.uint8)
    if mask.any():
        lower, upper = np.quantile(rendered[mask], (0.02, 0.98))
        normalised = np.clip((rendered - lower) / max(float(upper - lower), 1e-8), 0.0, 1.0)
        colour = cv2.applyColorMap((255 * (1.0 - normalised)).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel[mask] = colour[mask, ::-1]
    return panel


def _residual_panel(
    prior_uv, prior_depth, partial_uv, partial_depth,
    matched_prior_uv, residual_distance, resolution: int,
) -> np.ndarray:
    _, prior_mask, _ = zbuffer_depth_with_indices(
        prior_uv, prior_depth, (resolution, resolution), splat_radius=1,
    )
    _, partial_mask, _ = zbuffer_depth_with_indices(
        partial_uv, partial_depth, (resolution, resolution), splat_radius=2,
    )
    panel = np.full((resolution, resolution, 3), 18, dtype=np.uint8)
    panel[prior_mask] = (235, 55, 55)
    panel[partial_mask] = (20, 225, 245)
    panel[prior_mask & partial_mask] = (245, 245, 245)

    finite = (
        np.isfinite(partial_uv).all(axis=1) & np.isfinite(matched_prior_uv).all(axis=1)
        & (partial_depth > 0) & (residual_distance > np.quantile(residual_distance, 0.70))
    )
    ids = np.flatnonzero(finite)
    if len(ids):
        # Spatially balanced deterministic subset of the largest residuals.
        ids = ids[np.argsort(residual_distance[ids])[::-1]]
        occupied = set()
        selected = []
        cell = max(8, resolution // 24)
        for index in ids:
            key = tuple(np.floor(partial_uv[index] / cell).astype(int))
            if key in occupied:
                continue
            occupied.add(key)
            selected.append(index)
            if len(selected) >= 96:
                break
        for index in selected:
            start = tuple(np.rint(matched_prior_uv[index]).astype(int))
            end = tuple(np.rint(partial_uv[index]).astype(int))
            if all(-32 <= value < resolution + 32 for value in (*start, *end)):
                cv2.arrowedLine(panel, start, end, (255, 215, 35), 1, cv2.LINE_AA, tipLength=0.25)
    return panel


def _label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 205, 35), radius=5, fill=(0, 0, 0, 190))
    draw.text((16, 15), text, fill="white")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registered-prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--object-type", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prior, partial = load_points(args.registered_prior), load_points(args.partial)
    distances, nearest = cKDTree(prior).query(partial, workers=-1)
    matched_prior = prior[nearest]
    panels = []
    view_records = []
    for view in manifest["views"]:
        pose = np.asarray(view["camera_pose"], dtype=np.float64)
        prior_uv, prior_depth = _project(prior, pose, manifest["field_of_view_degrees"], args.resolution)
        partial_uv, partial_depth = _project(partial, pose, manifest["field_of_view_degrees"], args.resolution)
        matched_uv, _ = _project(matched_prior, pose, manifest["field_of_view_degrees"], args.resolution)
        rgb = Image.open(view["image"]).convert("RGB").resize(
            (args.resolution, args.resolution), Image.Resampling.LANCZOS,
        )
        depth = Image.fromarray(_depth_panel(partial_uv, partial_depth, args.resolution))
        residual = Image.fromarray(_residual_panel(
            prior_uv, prior_depth, partial_uv, partial_depth,
            matched_uv, distances, args.resolution,
        ))
        panels.append([
            _label(rgb, f"{view['name'].upper()} - PRIOR RGB"),
            _label(depth, f"{view['name'].upper()} - PARTIAL DEPTH"),
            _label(residual, f"{view['name'].upper()} - RESIDUAL"),
        ])
        view_records.append({"name": view["name"], "camera_pose": view["camera_pose"]})

    gutter = 8
    board = Image.new(
        "RGB",
        (args.resolution * len(panels) + gutter * (len(panels) - 1),
         args.resolution * 3 + gutter * 2),
        (245, 245, 245),
    )
    for column, column_panels in enumerate(panels):
        for row, panel in enumerate(column_panels):
            board.paste(panel, (column * (args.resolution + gutter), row * (args.resolution + gutter)))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    board_path = output / "prior_partial_multiview_residual_board.png"
    board.save(board_path)
    for record, column_panels in zip(view_records, panels, strict=True):
        evidence = Image.new(
            "RGB",
            (args.resolution, args.resolution * 3 + gutter * 2),
            (245, 245, 245),
        )
        for row, panel in enumerate(column_panels):
            evidence.paste(panel, (0, row * (args.resolution + gutter)))
        evidence_path = output / f"{record['name']}_residual_evidence.png"
        evidence.save(evidence_path)
        record["evidence_image"] = str(evidence_path)
    prompt = f"""Use case: precise-object-edit
Asset type: geometry-conditioned multi-view 3D regeneration input
Primary request: Produce one clean horizontal three-panel image showing the same complete {args.object_type} from FRONT, SIDE, and BACK, in that order, on a pure white background.
Input image: a diagnostic board. In each column, the top panel is the current complete prior RGB view, the middle panel is observed partial depth, and the bottom panel overlays the prior in red and the observed partial in cyan; yellow arrows indicate observed geometric residual directions.
Geometry rule: adjust the complete object's shape and articulated geometry so that every reliably observed cyan/depth-supported region is consistent with the partial evidence in all supplied views. Interpret the arrows and depth maps jointly, not as decorative marks. Propagate supported displacement smoothly through connected structure so no part detaches, tears, duplicates, shrinks away, or intersects unnaturally.
Invariants: preserve object identity, material, texture, camera orientation, view order, lighting, topology, all already-consistent regions, and complete geometry where the partial provides no evidence. Keep one identical object across all three views with strict cross-view consistency.
Output constraints: output only the clean three-panel RGB render; do not reproduce diagnostic labels, colors, arrows, depth maps, borders, captions, or text. No additional objects, no crop, no watermark.
"""
    prompt_path = output / "residual_guided_multiview_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    record = {
        "method": "category_parameterized_multiview_residual_conditioning",
        "ground_truth_used": False,
        "object_type": args.object_type,
        "registered_prior": str(args.registered_prior.resolve()),
        "partial": str(args.partial.resolve()),
        "manifest": str(args.manifest.resolve()),
        "board": str(board_path),
        "prompt": str(prompt_path),
        "colour_legend": {
            "red": "registered complete prior projection",
            "cyan": "observed partial projection",
            "white": "projected overlap",
            "yellow_arrows": "nearest-surface residual from prior toward partial",
            "turbo": "partial depth, near-to-far",
        },
        "views": view_records,
    }
    (output / "residual_condition_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"board": str(board_path), "prompt": str(prompt_path)}, indent=2))


if __name__ == "__main__":
    main()
