#!/usr/bin/env python3
"""Fuse a deformed complete posterior with a partial using four saved views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multiview_partial_evidence import compose_grid, load_manifest_projectors
from src.observation_anchored_fusion import (
    ObservationAnchoredFusionConfig,
    fuse_observation_anchors,
)
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.zbuffer import zbuffer_depth_with_indices


def _save_four_view_overlay(
    output: Path,
    partial: np.ndarray,
    fused: np.ndarray,
    projectors: list,
) -> Path:
    panels: list[Image.Image] = []
    for projector in projectors:
        p_uv, p_depth = projector.project(partial)
        f_uv, f_depth = projector.project(fused)
        _, p_mask, _ = zbuffer_depth_with_indices(
            p_uv, p_depth, projector.image_shape, splat_radius=1,
        )
        _, f_mask, _ = zbuffer_depth_with_indices(
            f_uv, f_depth, projector.image_shape, splat_radius=1,
        )
        height, width = projector.image_shape
        panel = np.full((height, width, 3), 255, dtype=np.uint8)
        panel[f_mask] = np.array((232, 45, 45), dtype=np.uint8)
        # Measured partial stays visible on top of the complete red carrier.
        panel[p_mask] = np.array((92, 92, 92), dtype=np.uint8)
        panels.append(Image.fromarray(panel))
    return compose_grid(panels, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--multiview-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pixel-radius", type=float, default=1.0)
    parser.add_argument("--cross-view-pixel-radius", type=float, default=2.0)
    parser.add_argument("--cross-view-depth-ratio", type=float, default=0.075)
    parser.add_argument("--anchor-residual-ratio", type=float, default=0.075)
    args = parser.parse_args()

    posterior = load_points(args.posterior)
    partial = load_points(args.partial)
    projectors = load_manifest_projectors(
        args.multiview_manifest, resolution=int(args.resolution),
    )
    config = ObservationAnchoredFusionConfig(
        num_views=4,
        pixel_radius=float(args.pixel_radius),
        cross_view_pixel_radius=float(args.cross_view_pixel_radius),
        cross_view_depth_ratio=float(args.cross_view_depth_ratio),
        anchor_residual_ratio=float(args.anchor_residual_ratio),
    )
    fused, anchors, pairs, info = fuse_observation_anchors(
        posterior, partial, projectors, config=config,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "observation_anchored_fused_100k.ply"
    comparison = output / "partial_gray_fused_red.ply"
    overlay = output / "fused_four_view_projection.png"
    write_points(prediction, fused)
    write_compare(comparison, partial, fused)
    _save_four_view_overlay(overlay, partial, fused, projectors)
    np.save(output / "fusion_anchor_pairs.npy", anchors)
    np.save(output / "fusion_positive_pairs.npy", pairs)
    record = {
        "inputs": {
            "posterior": str(args.posterior.resolve()),
            "partial": str(args.partial.resolve()),
            "multiview_manifest": str(args.multiview_manifest.resolve()),
        },
        "fusion": info,
        "outputs": {
            "prediction": str(prediction.resolve()),
            "comparison": str(comparison.resolve()),
            "four_view_projection": str(overlay.resolve()),
            "anchors": str((output / "fusion_anchor_pairs.npy").resolve()),
            "positive_pairs": str((output / "fusion_positive_pairs.npy").resolve()),
        },
    }
    (output / "fusion_info.json").write_text(
        json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(jsonable(info), indent=2))


if __name__ == "__main__":
    main()
