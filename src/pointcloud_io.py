"""Point-cloud I/O and JSON conversion used by the compact mainline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_points(path: Path) -> np.ndarray:
    path = Path(path)
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return points


def write_points(path: Path, points: np.ndarray, color=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if color is not None:
        cloud.paint_uniform_color(color)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f"Failed to write point cloud: {path}")


def write_compare(path: Path, partial: np.ndarray, generated: np.ndarray) -> None:
    partial_cloud = o3d.geometry.PointCloud()
    partial_cloud.points = o3d.utility.Vector3dVector(np.asarray(partial, dtype=np.float64))
    partial_cloud.paint_uniform_color([0.55, 0.55, 0.55])
    generated_cloud = o3d.geometry.PointCloud()
    generated_cloud.points = o3d.utility.Vector3dVector(np.asarray(generated, dtype=np.float64))
    generated_cloud.paint_uniform_color([0.9, 0.1, 0.1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), partial_cloud + generated_cloud):
        raise IOError(f"Failed to write comparison point cloud: {path}")
