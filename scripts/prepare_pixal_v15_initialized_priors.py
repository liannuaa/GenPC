#!/usr/bin/env python3
"""Apply frozen v15 coarse transforms to newly generated Pixal3D priors.

The resulting layout intentionally matches the v15 runner contract, while the
subsequent bidirectional optimizer remains responsible for the final Sim(3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


DEFAULT_SAMPLES = (
    "01184", "05117", "05452", "06127", "06145",
    "06188", "06830", "07136", "07306", "09639",
)


def _load_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    colors = getattr(cloud.visual, "vertex_colors", None)
    if colors is not None:
        colors = np.asarray(colors)
        if len(colors) != len(points):
            colors = None
    return points, colors


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v15-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    args = parser.parse_args()

    for sample in args.samples:
        source_dir = args.root / sample
        baseline_dir = args.v15_root / sample
        output_dir = args.output_root / sample
        output_dir.mkdir(parents=True, exist_ok=True)

        transform = np.load(
            baseline_dir / f"{sample}_unified_registration_v14.npy"
        ).astype(np.float64)
        points, colors = _load_points(source_dir / "pixal3d_sampled_100k.ply")
        registered = _apply(points, transform)

        stem = output_dir / f"{sample}_unified_registration_v14"
        trimesh.PointCloud(registered, colors=colors).export(
            Path(f"{stem}_registered_100k.ply")
        )

        mesh = trimesh.load(source_dir / "pixal3d.glb", force="scene", process=False)
        mesh.apply_transform(transform)
        mesh.export(Path(f"{stem}_registered_mesh.glb"))
        np.save(Path(f"{stem}.npy"), transform)
        print(f"[Done] {sample}: {len(registered)} points")


if __name__ == "__main__":
    main()
