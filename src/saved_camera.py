"""Saved-camera projection and diagnostic overlays for the mainline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.zbuffer import zbuffer_depth_with_indices


class SavedCameraProjector:
    """Project raw-frame points with the saved ``src.depth_prompting`` camera."""

    def __init__(self, camera, center_xy, scale_xy, *, padding, image_shape, device="cpu"):
        self.camera = camera
        self.center_xy = np.asarray(center_xy, dtype=np.float64)
        self.scale_xy = float(scale_xy)
        self.padding = float(padding)
        self.image_shape = tuple(int(value) for value in image_shape)
        self.device = str(device)

    @classmethod
    def from_partial(cls, partial_points, camera_path, *, padding, image_shape, device="cpu"):
        camera = torch.load(str(camera_path), map_location=device, weights_only=False)
        points = torch.as_tensor(np.asarray(partial_points), dtype=torch.float32, device=device)
        with torch.no_grad():
            camera_points = camera.transform(points).detach().float().cpu().numpy()
        xy_min, xy_max = camera_points[:, :2].min(axis=0), camera_points[:, :2].max(axis=0)
        return cls(
            camera, (xy_min + xy_max) * 0.5,
            max(float((xy_max - xy_min).max()), 1e-8),
            padding=padding, image_shape=image_shape, device=device,
        )

    def project(self, points):
        points = np.asarray(points, dtype=np.float64)
        transformed = []
        with torch.no_grad():
            for start in range(0, len(points), 200_000):
                chunk = torch.as_tensor(points[start:start + 200_000], dtype=torch.float32, device=self.device)
                transformed.append(self.camera.transform(chunk).detach().float().cpu().numpy())
        camera_points = np.concatenate(transformed, axis=0)
        uv = (camera_points[:, :2] - self.center_xy) / self.scale_xy
        uv = uv * (1.0 - 2.0 * self.padding) + 0.5
        uv[:, 1] = 1.0 - uv[:, 1]
        height, width = self.image_shape
        return uv * np.array([width - 1, height - 1], dtype=np.float64), camera_points[:, 2]


def draw_projection_overlay(path: Path, image_path: Path, partial_points, generated_points, projector) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = projector.image_shape
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    partial_uv, partial_depth = projector.project(partial_points)
    generated_uv, generated_depth = projector.project(generated_points)
    _, partial_mask, _ = zbuffer_depth_with_indices(partial_uv, partial_depth, projector.image_shape, splat_radius=1)
    _, generated_mask, _ = zbuffer_depth_with_indices(generated_uv, generated_depth, projector.image_shape, splat_radius=1)
    overlay = image.copy()
    overlay[partial_mask] = (0.55 * overlay[partial_mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[generated_mask] = (0.55 * overlay[generated_mask] + 0.45 * np.array([0, 80, 255])).astype(np.uint8)
    overlap = partial_mask & generated_mask
    overlay[overlap] = (0.35 * overlay[overlap] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)
