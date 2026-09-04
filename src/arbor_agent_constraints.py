"""Turn a partial scan into a coordinate-audited Arbor hull constraint."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ArborConstraintContract:
    raw_to_constraint: np.ndarray
    constraint_to_raw: np.ndarray
    source_point_count: int
    constraint_point_count: int
    support_radius: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["raw_to_constraint"] = self.raw_to_constraint.tolist()
        payload["constraint_to_raw"] = self.constraint_to_raw.tolist()
        return payload


def partial_to_arbor_hull(
    points: np.ndarray, *, max_points: int = 2048, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, ArborConstraintContract]:
    """Build a watertight union-of-octahedra hull in a normalized frame.

    Each observed point becomes a small watertight support primitive. This
    avoids inventing an unobserved back-side surface merely to make an open
    scan usable as a mesh constraint. The common global proper-Sim(3) solver
    later maps the generated prior back to the method frame.
    """
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) < 2:
        raise ValueError("points must be an Nx3-or-greater point cloud")
    xyz = xyz[:, :3]
    source_count = len(xyz)
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    if source_count > max_points:
        ids = np.random.default_rng(seed).choice(source_count, max_points, replace=False)
        xyz = xyz[np.sort(ids)]
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    scale = float((upper - lower).max())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("point cloud must have non-zero spatial extent")
    center = (lower + upper) / 2.
    raw_to_constraint = np.eye(4)
    raw_to_constraint[:3, :3] *= 1. / scale
    raw_to_constraint[:3, 3] = -center / scale
    constraint_to_raw = np.linalg.inv(raw_to_constraint)
    normalized = xyz @ raw_to_constraint[:3, :3].T + raw_to_constraint[:3, 3]
    distances, _ = cKDTree(normalized).query(normalized, k=2)
    radius = float(np.clip(1.5 * np.median(distances[:, 1]), .0025, .03))

    directions = np.array([
        [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
        [0., -1., 0.], [0., 0., 1.], [0., 0., -1.],
    ])
    base_faces = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=np.int64)
    vertices = (normalized[:, None, :] + radius * directions[None, :, :]).reshape(-1, 3)
    offsets = (6 * np.arange(len(normalized), dtype=np.int64))[:, None, None]
    faces = (base_faces[None, :, :] + offsets).reshape(-1, 3)
    contract = ArborConstraintContract(
        raw_to_constraint=raw_to_constraint,
        constraint_to_raw=constraint_to_raw,
        source_point_count=source_count,
        constraint_point_count=len(normalized),
        support_radius=radius,
    )
    return vertices, faces, contract
