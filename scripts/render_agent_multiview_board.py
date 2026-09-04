#!/usr/bin/env python3
"""Render three deterministic orthographic point-cloud evidence views.

These views are for the agent's visual observation only.  They are derived
from the registered partial/prior geometry and never introduce a second camera
or a generated multiview image into the completion method.
"""

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


def pca_frame(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    centered = values - np.median(values, axis=0)
    _, frame = np.linalg.eigh(centered.T @ centered / max(len(values) - 1, 1))
    frame = frame[:, ::-1]
    if np.linalg.det(frame) < 0.0:
        frame[:, -1] *= -1.0
    return frame


def rasterize(points: np.ndarray, axes: tuple[int, int], plane_range, size: int, canvas):
    xy = points[:, axes]
    low, high = plane_range[:, 0], plane_range[:, 1]
    uv = (xy - low) / np.maximum(high - low, 1e-8)
    px = np.rint(uv * (size - 1)).astype(np.int64)
    valid = ((px[:, 0] >= 0) & (px[:, 0] < size) & (px[:, 1] >= 0) & (px[:, 1] < size))
    canvas[size - 1 - px[valid, 1], px[valid, 0]] = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=60000)
    args = parser.parse_args()
    partial = base.subset(base.load_points(args.partial), args.max_points)
    prior = base.subset(base.load_points(args.prior), args.max_points)
    frame = pca_frame(np.concatenate((partial, prior), axis=0))
    centre = np.median(np.concatenate((partial, prior), axis=0), axis=0)
    partial, prior = tuple((points - centre) @ frame for points in (partial, prior))
    all_points = np.concatenate((partial, prior), axis=0)
    title = ("long × middle", "long × short", "middle × short")
    axis_pairs = ((0, 1), (0, 2), (1, 2))
    panels = []
    for label, axes in zip(title, axis_pairs):
        combined = all_points[:, axes]
        limits = np.stack((combined.min(axis=0), combined.max(axis=0)), axis=1)
        padding = .04 * np.maximum(limits[:, 1] - limits[:, 0], 1e-8)
        limits[:, 0] -= padding; limits[:, 1] += padding
        partial_mask = np.zeros((args.size, args.size), dtype=bool)
        prior_mask = np.zeros((args.size, args.size), dtype=bool)
        rasterize(partial, axes, limits, args.size, partial_mask)
        rasterize(prior, axes, limits, args.size, prior_mask)
        image = np.zeros((args.size, args.size, 3), dtype=np.uint8)
        image[partial_mask] = (155, 155, 155)
        image[prior_mask] = (225, 30, 30)
        image[partial_mask & prior_mask] = (242, 185, 40)
        panel = Image.new("RGB", (args.size, args.size + 26), "white")
        panel.paste(Image.fromarray(image), (0, 26))
        ImageDraw.Draw(panel).text((8, 5), f"{label}: red prior / gray partial / yellow overlap", fill="black")
        panels.append(panel)
    board = Image.new("RGB", (3 * args.size, args.size + 26), "white")
    for index, panel in enumerate(panels):
        board.paste(panel, (index * args.size, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.save(args.output)


if __name__ == "__main__":
    main()
