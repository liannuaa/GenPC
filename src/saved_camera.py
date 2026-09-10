"""Saved-camera projection and diagnostic overlays for the mainline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.zbuffer import zbuffer_depth_with_indices


class SavedCameraProjector:
    """Project points with the exact camera model that produced Camera-1.

    Historical assets use the normalized Kaolin projection below.  Other
    datasets may carry a ``camera.json`` sidecar with an explicit pinhole
    intrinsic/extrinsic pair.  Reading that contract keeps registration
    dataset-agnostic and avoids mixing perspective and normalized UVs.
    """

    def __init__(self, camera, center_xy, scale_xy, *, padding, image_shape, device="cpu",
                 pinhole_intrinsic=None, pinhole_world_to_camera=None,
                 pinhole_image_shape=None):
        self.camera = camera
        self.center_xy = np.asarray(center_xy, dtype=np.float64)
        self.scale_xy = float(scale_xy)
        self.padding = float(padding)
        self.image_shape = tuple(int(value) for value in image_shape)
        self.device = str(device)
        self.pinhole_intrinsic = (
            None if pinhole_intrinsic is None
            else np.asarray(pinhole_intrinsic, dtype=np.float64)
        )
        self.pinhole_world_to_camera = (
            None if pinhole_world_to_camera is None
            else np.asarray(pinhole_world_to_camera, dtype=np.float64)
        )
        self.pinhole_image_shape = (
            None if pinhole_image_shape is None
            else tuple(int(value) for value in pinhole_image_shape)
        )
        self.projection_model = (
            "pinhole_sidecar" if self.pinhole_intrinsic is not None
            else "normalized_saved_camera"
        )

    @classmethod
    def from_partial(cls, partial_points, camera_path, *, padding, image_shape, device="cpu"):
        camera_path = Path(camera_path)
        camera = torch.load(str(camera_path), map_location=device, weights_only=False)
        sidecar = camera_path.with_name("camera.json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            intrinsic = np.asarray(metadata.get("intrinsic"), dtype=np.float64)
            world_to_camera = np.asarray(
                metadata.get("extrinsic_world_to_camera"), dtype=np.float64,
            )
            image_size = metadata.get("image_size")
            if (
                intrinsic.shape == (3, 3)
                and world_to_camera.shape == (4, 4)
                and image_size is not None
            ):
                native_shape = (
                    (int(image_size), int(image_size))
                    if np.isscalar(image_size)
                    else tuple(int(value) for value in image_size)
                )
                if len(native_shape) != 2:
                    raise ValueError("camera.json image_size must define height and width")
                return cls(
                    camera, (0.0, 0.0), 1.0,
                    padding=padding, image_shape=image_shape, device=device,
                    pinhole_intrinsic=intrinsic,
                    pinhole_world_to_camera=world_to_camera,
                    pinhole_image_shape=native_shape,
                )
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
        if self.pinhole_intrinsic is not None:
            rotation = self.pinhole_world_to_camera[:3, :3]
            translation = self.pinhole_world_to_camera[:3, 3]
            camera_points = points @ rotation.T + translation
            depth = camera_points[:, 2]
            safe_depth = np.where(np.abs(depth) > 1e-12, depth, np.nan)
            intrinsic = self.pinhole_intrinsic
            pixel = np.empty((len(points), 2), dtype=np.float64)
            pixel[:, 0] = intrinsic[0, 0] * camera_points[:, 0] / safe_depth + intrinsic[0, 2]
            pixel[:, 1] = intrinsic[1, 1] * camera_points[:, 1] / safe_depth + intrinsic[1, 2]
            native_height, native_width = self.pinhole_image_shape
            height, width = self.image_shape
            pixel[:, 0] *= max(width - 1, 1) / max(native_width - 1, 1)
            pixel[:, 1] *= max(height - 1, 1) / max(native_height - 1, 1)
            return pixel, depth
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


def world_to_camera_axes(camera) -> np.ndarray:
    """Return the orthonormal Camera-1 row axes used by a saved Kaolin camera.

    For world-space row vectors, ``world @ axes.T`` expresses displacements in
    the saved camera frame.  Translation cancels for local scale estimates,
    so this is exactly the frame in which image-plane and depth constraints
    are observed.
    """
    rotation = getattr(camera, "R", None)
    if rotation is None:
        raise ValueError("saved camera does not expose an extrinsic rotation R")
    if hasattr(rotation, "detach"):
        rotation = rotation.detach().float().cpu().numpy()
    axes = np.asarray(rotation, dtype=np.float64)
    if axes.ndim == 3 and axes.shape[0] == 1:
        axes = axes[0]
    if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3), rtol=1e-4, atol=1e-5):
        raise ValueError("saved camera rotation is not a valid orthonormal (3, 3) frame")
    return axes


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
