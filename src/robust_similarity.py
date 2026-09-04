"""Small, dependency-free proper Sim(3) estimation primitive."""

from __future__ import annotations

import numpy as np


def validate_proper_sim3(transform: np.ndarray, *, atol: float = 1e-5) -> float:
    """Validate a finite proper isotropic Sim(3) and return its scale."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError("transform must preserve orientation")
    scale = float(np.cbrt(determinant))
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("transform may not contain anisotropic scale or shear")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("transform must contain a proper rotation")
    return scale


def weighted_umeyama(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Estimate a weighted proper isotropic similarity from source to target."""
    source, target, weights = (np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64),
                               np.asarray(weights, dtype=np.float64))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source and target must be aligned (N, 3) arrays with N >= 3")
    weights = np.clip(weights, 0., None)
    weights /= max(float(weights.sum()), 1e-12)
    source_center = (weights[:, None] * source).sum(axis=0)
    target_center = (weights[:, None] * target).sum(axis=0)
    source_zero, target_zero = source - source_center, target - target_center
    covariance = (weights[:, None] * target_zero).T @ source_zero
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0.:
        sign[-1] = -1.
    rotation = u @ np.diag(sign) @ vt
    variance = float((weights * np.sum(source_zero * source_zero, axis=1)).sum())
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_center - scale * rotation @ source_center
    return transform
