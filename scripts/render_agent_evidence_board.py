#!/usr/bin/env python3
"""Render a deterministic saved-view board for a visual agent observation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base


def mask_panel(mask: np.ndarray, color: tuple[int, int, int], size: int) -> Image.Image:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[mask] = color
    return Image.fromarray(image, mode="RGB")


def label(image: Image.Image, text: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 28), "white")
    panel.paste(image, (0, 28))
    ImageDraw.Draw(panel).text((8, 6), text, fill="black")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    partial, prior = base.load_points(args.partial), base.load_points(args.prior)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=args.padding, image_shape=(args.size, args.size), device="cpu")
    partial_mask = base.render_mask(projector, partial, args.size, splat=1)
    prior_mask = base.render_mask(projector, prior, args.size, splat=1)
    overlay = np.zeros((args.size, args.size, 3), dtype=np.uint8)
    overlay[partial_mask] = (145, 145, 145)
    overlay[prior_mask] = (225, 30, 30)
    overlay[partial_mask & prior_mask] = (242, 185, 40)
    semantic = Image.open(args.semantic).convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
    panels = [
        label(mask_panel(partial_mask, (180, 180, 180), args.size), "Observed partial (gray)"),
        label(mask_panel(prior_mask, (225, 30, 30), args.size), "Current prior (red)"),
        label(Image.fromarray(overlay), "Overlay: yellow agreement"),
        label(semantic, "Semantic image / saved view"),
    ]
    board = Image.new("RGB", (2 * args.size, 2 * (args.size + 28)), "white")
    for index, panel in enumerate(panels):
        board.paste(panel, ((index % 2) * args.size, (index // 2) * (args.size + 28)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.save(args.output)


if __name__ == "__main__":
    main()
