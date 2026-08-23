#!/usr/bin/env python3
"""Render GT-free v15-before / residual-after point-cloud overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load(path):
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)


def choose(points, count):
    if len(points) <= int(count):
        return points
    return points[np.linspace(0, len(points) - 1, int(count), dtype=np.int64)]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="v15 before / hierarchical residual after")
    args = parser.parse_args(argv)
    partial = choose(load(args.partial), 18000)
    before = choose(load(args.before), 26000)
    after = choose(load(args.after), 26000)
    center = np.median(np.concatenate((partial, before, after)), axis=0)
    covariance = np.cov((partial - center).T)
    _, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    partial = (partial - center) @ basis
    before = (before - center) @ basis
    after = (after - center) @ basis
    all_points = np.concatenate((partial, before, after))
    radius = .54 * max(float(np.ptp(all_points, axis=0).max()), 1e-8)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="#202124")
    views = ((0, 1), (0, 2), (1, 2))
    for row, (name, complete) in enumerate((("v15", before), ("after", after))):
        for column, (x, y) in enumerate(views):
            axis = axes[row, column]
            axis.set_facecolor("#202124")
            axis.scatter(partial[:, x], partial[:, y], s=.30, c="#bdbdbd",
                         alpha=.55, linewidths=0)
            axis.scatter(complete[:, x], complete[:, y], s=.22, c="#ff3030",
                         alpha=.62, linewidths=0)
            axis.set_xlim(-radius, radius); axis.set_ylim(-radius, radius)
            axis.set_aspect("equal"); axis.set_axis_off()
            axis.set_title(f"{name} · PCA {x}{y}", color="white", fontsize=10)
    figure.suptitle(args.title, color="white")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
