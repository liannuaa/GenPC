"""Coordinate contract for interchangeable image/text 3-D prior backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class NativePriorContract:
    """Explicit record of a backend's native-to-method coordinate mapping."""

    backend: str
    native_to_method: np.ndarray
    native_frame: str
    method_frame: str = "partial_saved_camera_method_frame"

    def to_dict(self) -> dict:
        output = asdict(self)
        output["native_to_method"] = np.asarray(self.native_to_method).tolist()
        return output


def validate_proper_sim3(transform: np.ndarray, *, atol: float = 1e-5) -> float:
    """Validate an adapter transform and return its isotropic scale."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("native_to_method must be a finite 4x4 transform")
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError("native_to_method must preserve orientation")
    scale = float(np.cbrt(determinant))
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("native_to_method may not contain anisotropic scale or shear")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("native_to_method must contain a proper rotation")
    return scale


def apply_native_to_method(points: np.ndarray, contract: NativePriorContract) -> np.ndarray:
    """Apply only an explicitly validated backend adapter transform."""
    transform = np.asarray(contract.native_to_method, dtype=np.float64)
    validate_proper_sim3(transform)
    values = np.asarray(points, dtype=np.float64)
    return values @ transform[:3, :3].T + transform[:3, 3]
