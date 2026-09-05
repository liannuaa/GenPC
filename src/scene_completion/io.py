"""Small colour-preserving point-cloud helpers for the scene wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def write_colored_points(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError(f"expected a non-empty [N,3] point array, got {values.shape}")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(values))
    if colors is not None:
        rgb = np.asarray(colors, dtype=np.float64)
        if rgb.shape != values.shape:
            raise ValueError("points and colors must both have shape [N,3]")
        cloud.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f"failed to write point cloud {path}")


def load_colored_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        raise ValueError(f"empty point cloud: {path}")
    colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
    return points, colors
