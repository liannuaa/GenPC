"""All-support residual Sim(3) capture for a regenerated complete prior.

The visible part of a partial scan may contain small but decisive structures
that are removed by a fixed trimmed-ICP fraction.  This module instead scores
all observed samples with a bounded robust distance and a high-quantile term.
The complete Camera-1 silhouette supplies the complementary image constraint.
No category, part label, ground truth, or preferred world axis is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch


@dataclass(frozen=True)
class AllSupportSim3Config:
    rotation_degrees: float = 22.0
    log_scale: float = math.log(1.14)
    translation_ratio: float = 0.18
    partial_samples: int = 8_192
    prior_samples: int = 24_000
    population: int = 7
    iterations: int = 22
    seed: int = 6145
    silhouette_weight: float = 0.0
    visible_weight: float = 0.55


def _camera_affine(projector) -> tuple[np.ndarray, np.ndarray]:
    probes = torch.as_tensor(
        np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0),
        dtype=torch.float32,
        device=projector.device,
    )
    with torch.no_grad():
        mapped = projector.camera.transform(probes).detach().float().cpu().numpy()
    return (mapped[1:] - mapped[0]).astype(np.float64), mapped[0].astype(np.float64)


def _foreground_statistics(image_path, projector) -> dict[str, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.any(image < 245, axis=2).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) < 64:
        raise ValueError(f"Camera-1 condition has no reliable foreground: {image_path}")
    height, width = mask.shape
    uv = np.stack((xs / max(width - 1, 1), ys / max(height - 1, 1)), axis=1)
    lower, median, upper = np.quantile(uv, (0.01, 0.50, 0.99), axis=0)
    centred = uv - uv.mean(axis=0)
    covariance = centred.T @ centred / max(len(centred), 1)
    return {
        "lower": lower,
        "median": median,
        "upper": upper,
        "covariance": covariance,
    }


def _project_normalized(points: np.ndarray, projector, linear: np.ndarray, offset: np.ndarray) -> np.ndarray:
    camera = points @ linear + offset
    uv = (camera[:, :2] - projector.center_xy) / projector.scale_xy
    uv = uv * (1.0 - 2.0 * projector.padding) + 0.5
    uv[:, 1] = 1.0 - uv[:, 1]
    valid = np.isfinite(uv).all(axis=1)
    valid &= np.isfinite(camera[:, 2])
    valid &= (uv[:, 0] > -0.25) & (uv[:, 0] < 1.25)
    valid &= (uv[:, 1] > -0.25) & (uv[:, 1] < 1.25)
    return uv[valid]


def _silhouette_moment_error(uv: np.ndarray, target: dict[str, np.ndarray]) -> float:
    if len(uv) < 64:
        return 3.0
    lower, median, upper = np.quantile(uv, (0.01, 0.50, 0.99), axis=0)
    extent = np.maximum(upper - lower, 1e-6)
    target_extent = np.maximum(target["upper"] - target["lower"], 1e-6)
    centred = uv - uv.mean(axis=0)
    covariance = centred.T @ centred / max(len(centred), 1)
    covariance_scale = max(float(np.trace(target["covariance"])), 1e-6)
    return float(
        1.8 * np.linalg.norm(median - target["median"])
        + 0.9 * np.mean(np.abs(np.log(extent / target_extent)))
        + 0.30 * np.linalg.norm(covariance - target["covariance"]) / covariance_scale
    )


def _pivoted_delta(parameters: np.ndarray, pivot: np.ndarray, diagonal: float) -> np.ndarray:
    rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
    scale = float(np.exp(parameters[3]))
    translation = np.asarray(parameters[4:7], dtype=np.float64) * float(diagonal)
    linear = scale * rotation
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = pivot + translation - linear @ pivot
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def capture_all_support_sim3(
    initial_prior: np.ndarray,
    partial: np.ndarray,
    projector,
    camera1_condition,
    *,
    config: AllSupportSim3Config = AllSupportSim3Config(),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a residual proper Sim(3) selected without ground truth."""
    prior = np.asarray(initial_prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if min(len(prior), len(partial)) < 96:
        raise ValueError("prior and partial must each contain at least 96 points")
    rng = np.random.default_rng(int(config.seed))
    prior_ids = rng.choice(len(prior), min(int(config.prior_samples), len(prior)), replace=False)
    partial_ids = rng.choice(len(partial), min(int(config.partial_samples), len(partial)), replace=False)
    source = prior[prior_ids]
    observed = partial[partial_ids]
    tree = cKDTree(source)
    pivot = np.median(source, axis=0)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    linear, offset = _camera_affine(projector)
    target = _foreground_statistics(camera1_condition, projector)
    from src.bidirectional_cycle_registration import prepare_visible_target, visible_score
    target_cache = prepare_visible_target(observed, projector)

    def components(parameters: np.ndarray) -> tuple[float, float, float, float, float]:
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        scale = float(np.exp(parameters[3]))
        translation = np.asarray(parameters[4:7]) * diagonal
        # Query a fixed source tree by applying the inverse residual Sim(3) to
        # the observations. This makes broad search inexpensive and exact for
        # isotropic scale.
        inverse_observed = pivot + (observed - pivot - translation) @ rotation / scale
        distances = tree.query(inverse_observed, k=1, workers=-1)[0] * scale / diagonal
        robust_mean = float(np.mean(np.minimum(distances, 0.18)))
        high_support = float(np.quantile(distances, 0.90))
        median = float(np.median(distances))
        delta = _pivoted_delta(parameters, pivot, diagonal)
        moved_source = apply_transform(source, delta)
        uv = _project_normalized(moved_source, projector, linear, offset)
        silhouette = _silhouette_moment_error(uv, target)
        visible = visible_score(
            observed, moved_source, projector, diagonal,
            pixel_radius=5.0, target_cache=target_cache,
        )
        objective = (
            robust_mean + 0.40 * high_support + 0.12 * median
            + float(config.silhouette_weight) * silhouette
            + float(config.visible_weight) * float(visible["objective"])
        )
        return float(objective), robust_mean, high_support, silhouette, float(visible["objective"])

    angle = math.radians(float(config.rotation_degrees))
    bounds = (
        (-angle, angle), (-angle, angle), (-angle, angle),
        (-float(config.log_scale), float(config.log_scale)),
        (-float(config.translation_ratio), float(config.translation_ratio)),
        (-float(config.translation_ratio), float(config.translation_ratio)),
        (-float(config.translation_ratio), float(config.translation_ratio)),
    )
    zero = np.zeros(7, dtype=np.float64)
    before = components(zero)
    result = differential_evolution(
        lambda parameters: components(parameters)[0],
        bounds,
        seed=int(config.seed),
        popsize=int(config.population),
        maxiter=int(config.iterations),
        tol=2e-4,
        polish=True,
        updating="immediate",
        workers=1,
    )
    residual = _pivoted_delta(result.x, pivot, diagonal)
    registered = apply_transform(prior, residual)
    after = components(result.x)
    rotation_degrees = float(np.degrees(np.linalg.norm(result.x[:3])))
    return registered, residual, {
        "method": "all_partial_support_plus_camera1_silhouette_residual_sim3",
        "strict_zero_shot": True,
        "ground_truth_used": False,
        "category_or_part_rules_used": False,
        "before": {
            "objective": before[0], "robust_mean_3d": before[1],
            "partial_p90_3d": before[2], "silhouette_moment_error": before[3],
            "camera1_visible_objective": before[4],
        },
        "after": {
            "objective": after[0], "robust_mean_3d": after[1],
            "partial_p90_3d": after[2], "silhouette_moment_error": after[3],
            "camera1_visible_objective": after[4],
        },
        "parameters": {
            "rotation_vector": result.x[:3].tolist(),
            "rotation_degrees": rotation_degrees,
            "scale": float(np.exp(result.x[3])),
            "translation": (result.x[4:7] * diagonal).tolist(),
            "partial_diagonal": diagonal,
        },
        "residual_transform": residual.tolist(),
        "optimization": {
            "success": bool(result.success), "message": str(result.message),
            "evaluations": int(result.nfev), "iterations": int(result.nit),
        },
        "config": {
            key: getattr(config, key) for key in config.__dataclass_fields__
        },
    }
