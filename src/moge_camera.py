"""Camera model for a MoGe point map inferred from a semantic image."""

from __future__ import annotations

import numpy as np
import torch


class _IdentityCamera:
    def transform(self, points: torch.Tensor) -> torch.Tensor:
        return points


class MoGeProjector:
    """Perspective projector in MoGe's native semantic-image camera frame."""

    def __init__(self, normalized_intrinsics: np.ndarray, image_shape: tuple[int, int], *, device: str = "cpu"):
        intrinsic = np.asarray(normalized_intrinsics, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError("MoGe intrinsics must be 3x3")
        self.image_shape = tuple(map(int, image_shape))
        height, width = self.image_shape
        self.intrinsic = intrinsic.copy()
        self.intrinsic[0] *= width
        self.intrinsic[1] *= height
        self.intrinsic[2] = (0., 0., 1.)
        self.device = str(device)
        self.camera = _IdentityCamera()
        self.center_xy = np.array((width * .5, height * .5), dtype=np.float64)
        self.scale_xy = float(max(width, height))
        self.padding = 0.

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        depth = points[:, 2].copy()
        pixel = np.full((len(points), 2), np.nan, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1) & (depth > 1e-8)
        if np.any(valid):
            homogeneous = (self.intrinsic @ points[valid].T).T
            pixel[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
        return pixel, depth
