"""Coordinate-safe SPAR3D preparation for the agent prior backend.

SPAR3D accepts a point-cloud condition but does not normalize a user supplied
condition internally.  This module makes that normalization explicit and
records the model's documented mesh-output rotation.  A downstream global
proper Sim(3) registration remains responsible for mapping a SPAR3D prior to
the saved-camera partial frame; no fixed dataset or Pixal convention leaks
into this backend adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Spar3DConditioning:
    """Reversible native conditioning frame and exported mesh convention."""

    raw_to_condition: np.ndarray
    condition_to_raw: np.ndarray
    mesh_output_rotation: np.ndarray
    source_point_count: int
    condition_point_count: int

    def to_dict(self) -> dict:
        value = asdict(self)
        for name in ("raw_to_condition", "condition_to_raw", "mesh_output_rotation"):
            value[name] = np.asarray(value[name]).tolist()
        return value


def _rigid_rotation_x(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1., 0., 0., 0.], [0., c, -s, 0.], [0., s, c, 0.], [0., 0., 0., 1.]])


def _rigid_rotation_y(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0., s, 0.], [0., 1., 0., 0.], [-s, 0., c, 0.], [0., 0., 0., 1.]])


def prepare_spar3d_condition(
    points: np.ndarray, *, max_points: int = 2048, seed: int = 0,
) -> tuple[np.ndarray, Spar3DConditioning]:
    """Center and isotropically scale an observed partial for SPAR3D.

    The mapping matches SPAR3D's ``normalize_pc_bbox`` convention, but is
    applied here because SPAR3D bypasses that function for caller-supplied
    point clouds. RGB is neutral: the semantic image supplies appearance while
    the observed scan supplies geometry.
    """
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) < 2:
        raise ValueError("points must be an Nx3-or-greater point cloud")
    xyz = xyz[:, :3]
    source_point_count = len(xyz)
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    if source_point_count > max_points:
        # The SPAR3D point tokenizer has global attention. A fixed-size,
        # deterministic sample bounds its quadratic memory without depending
        # on the scan order or object identity.
        keep = np.random.default_rng(seed).choice(source_point_count, size=max_points, replace=False)
        xyz = xyz[np.sort(keep)]
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    scale = float((upper - lower).max())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("point cloud must have non-zero spatial extent")
    center = (upper + lower) / 2.
    raw_to_condition = np.eye(4)
    raw_to_condition[:3, :3] *= 1. / scale
    raw_to_condition[:3, 3] = -center / scale
    condition_to_raw = np.linalg.inv(raw_to_condition)
    # Exact convention in SPAR3D's ``generate_mesh``: R_y(+90) R_x(-90).
    mesh_output_rotation = _rigid_rotation_y(90.) @ _rigid_rotation_x(-90.)
    conditioned_xyz = xyz @ raw_to_condition[:3, :3].T + raw_to_condition[:3, 3]
    conditioned = np.concatenate([conditioned_xyz, np.full((len(xyz), 3), .5)], axis=1)
    return conditioned.astype(np.float32), Spar3DConditioning(
        raw_to_condition=raw_to_condition,
        condition_to_raw=condition_to_raw,
        mesh_output_rotation=mesh_output_rotation,
        source_point_count=source_point_count,
        condition_point_count=len(xyz),
    )
