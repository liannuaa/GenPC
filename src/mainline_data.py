"""Input loading and prompt labels for the compact mainline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


REDWOOD_LABELS = {
    "01184": "rubbish bin", "05117": "chair", "05452": "armchair",
    "06127": "a vase with leafy plant", "06145": "table", "06188": "motorcyle",
    "06830": "tricycle", "07136": "sofa", "07306": "trash container",
    "09639": "swivel chair",
}


def prompt_label(sample: str, cfg) -> str:
    overrides = getattr(cfg, "prompt_overrides", {}) or {}
    return str(overrides.get(str(sample), REDWOOD_LABELS.get(str(sample), sample)))


def load_partial(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if not len(points):
        raise ValueError(f"Point cloud is empty: {path}")
    if cloud.has_colors() and not np.allclose(np.asarray(cloud.colors), 0.0):
        colors = np.asarray(cloud.colors, dtype=np.float32)
    else:
        lower, upper = points.min(axis=0), points.max(axis=0)
        colors = np.clip((points - lower) / (upper - lower + 1e-8), 0.0, 1.0).astype(np.float32)
    return points, colors
