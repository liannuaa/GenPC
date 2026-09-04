"""Cheap saved-camera gate for text-conditioned 2-D edit targets.

An agent's image action must retain the projection that defined the registered
partial.  This gate operates before an expensive image-to-3D/mesh-edit action:
it vetoes a target that changes foreground size, position, or silhouette too
much.  It does not choose a target and has no category, sample, or GT input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ImageActionEvidence:
    iou: float
    source_coverage: float
    target_leakage: float
    bbox_scale_ratio: float
    centroid_shift_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def white_background_mask(image: np.ndarray, *, threshold: int = 248) -> np.ndarray:
    """Recover a conservative foreground mask from a white-canvas edit image."""
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError("expected an RGB image")
    return np.min(rgb[..., :3], axis=-1) < int(threshold)


def _bbox(mask: np.ndarray) -> tuple[np.ndarray, float]:
    yy, xx = np.nonzero(mask)
    if len(xx) < 16:
        raise ValueError("foreground mask is empty or degenerate")
    lo = np.asarray((xx.min(), yy.min()), dtype=np.float64)
    hi = np.asarray((xx.max(), yy.max()), dtype=np.float64)
    extent = np.maximum(hi - lo + 1., 1.)
    return .5 * (lo + hi), float(np.linalg.norm(extent))


def measure_camera_locked_image_action(source: np.ndarray, target: np.ndarray) -> ImageActionEvidence:
    source_mask, target_mask = white_background_mask(source), white_background_mask(target)
    if source_mask.shape != target_mask.shape:
        raise ValueError("source and target images must share a saved-camera resolution")
    intersection = float(np.logical_and(source_mask, target_mask).sum())
    source_count = float(source_mask.sum())
    target_count = float(target_mask.sum())
    union = float(np.logical_or(source_mask, target_mask).sum())
    source_center, source_extent = _bbox(source_mask)
    target_center, target_extent = _bbox(target_mask)
    image_diagonal = max(float(np.linalg.norm(source_mask.shape[::-1])), 1.)
    return ImageActionEvidence(
        iou=intersection / max(union, 1.),
        source_coverage=intersection / max(source_count, 1.),
        target_leakage=(target_count - intersection) / max(target_count, 1.),
        bbox_scale_ratio=target_extent / max(source_extent, 1e-12),
        centroid_shift_ratio=float(np.linalg.norm(target_center - source_center) / image_diagonal),
    )


def accept_camera_locked_image_action(
    evidence: ImageActionEvidence,
    *,
    minimum_iou: float = .88,
    minimum_source_coverage: float = .90,
    maximum_target_leakage: float = .12,
    minimum_bbox_scale_ratio: float = .90,
    maximum_bbox_scale_ratio: float = 1.10,
    maximum_centroid_shift_ratio: float = .035,
) -> bool:
    """Shared no-harm gate for a text-to-image target in the saved view."""
    return bool(
        evidence.iou >= float(minimum_iou)
        and evidence.source_coverage >= float(minimum_source_coverage)
        and evidence.target_leakage <= float(maximum_target_leakage)
        and float(minimum_bbox_scale_ratio) <= evidence.bbox_scale_ratio <= float(maximum_bbox_scale_ratio)
        and evidence.centroid_shift_ratio <= float(maximum_centroid_shift_ratio)
    )
