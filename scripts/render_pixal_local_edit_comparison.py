#!/usr/bin/env python3
"""Render color-preserving before/after turntables for local-edit audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load(path):
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    colors = np.asarray(cloud.colors, dtype=np.float64)
    if not len(points):
        raise ValueError(f"empty point cloud: {path}")
    if len(colors) != len(points):
        colors = np.full_like(points, .8)
    ids = np.linspace(0, len(points) - 1, min(len(points), 45000), dtype=np.int64)
    return points[ids], colors[ids]


def render(before_path, after_path, output, sample):
    before, before_color = load(before_path)
    after, after_color = load(after_path)
    joint = np.concatenate([before, after])
    center = np.median(joint, axis=0)
    radius = .56 * max(float(np.ptp(joint, axis=0).max()), 1e-8)
    figure = plt.figure(figsize=(16, 8), facecolor="#202124")
    for row, (label, points, colors) in enumerate((
            ("registered", before, before_color),
            ("partial-core local edit", after, after_color))):
        points = points - center
        for column, azimuth in enumerate((0, 90, 180, 270)):
            axis = figure.add_subplot(2, 4, row * 4 + column + 1,
                                      projection="3d", facecolor="#202124")
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=.11,
                         c=colors, linewidths=0, alpha=.86)
            axis.view_init(elev=15, azim=azimuth)
            axis.set_xlim(-radius, radius)
            axis.set_ylim(-radius, radius)
            axis.set_zlim(-radius, radius)
            axis.set_box_aspect((1, 1, 1))
            axis.set_axis_off()
            axis.set_title(f"{label} · {azimuth}°", color="white", fontsize=10)
    figure.suptitle(
        f"{sample}: partial gray, Pixal3D red", color="white", fontsize=14)
    figure.tight_layout(pad=.3)
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.before, args.after, args.output, args.sample)


if __name__ == "__main__":
    main()
