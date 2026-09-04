"""GPU test-time proper-Sim(3) refinement from a saved partial view.

The module is intentionally small: correspondences are recomputed in 3-D but
the update is optimized through their saved-camera reprojection as well as
their surface residual.  It never moves points independently, so the complete
Pixal prior stays a complete object.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from scripts import run_pixal_pca_depth_sim3_ttt_v3 as depth_v3
from src.ray_consistent_registration import soft_ray_correspondences


def _rotation(rotvec: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues map with a stable identity neighbourhood."""
    x, y, z = rotvec.unbind()
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return torch.matrix_exp(skew)


def _project(points: torch.Tensor, projector) -> torch.Tensor:
    camera = projector.camera.transform(points)
    center = torch.as_tensor(projector.center_xy, dtype=points.dtype, device=points.device)
    uv = (camera[:, :2] - center) / float(projector.scale_xy)
    uv = uv * (1. - 2. * float(projector.padding)) + .5
    uv[:, 1] = 1. - uv[:, 1]
    height, width = projector.image_shape
    return uv * torch.tensor([width - 1., height - 1.], dtype=points.dtype, device=points.device)


def _soft_screen(points: torch.Tensor, projector, *, size: int, z_min: float,
                 z_span: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable bilinear point-splat occupancy and visible depth."""
    camera = projector.camera.transform(points)
    center = torch.as_tensor(projector.center_xy, dtype=points.dtype, device=points.device)
    uv = (camera[:, :2] - center) / float(projector.scale_xy)
    uv = uv * (1. - 2. * float(projector.padding)) + .5
    uv = torch.stack((uv[:, 0], 1. - uv[:, 1]), dim=1) * float(size - 1)
    base = torch.floor(uv).long(); fraction = uv - base.to(uv.dtype)
    depth = camera[:, 2]
    visibility = torch.exp(-8. * torch.clamp((depth - z_min) / z_span, min=-.2, max=1.5))
    weights = torch.zeros(size * size, dtype=points.dtype, device=points.device)
    depth_sum = torch.zeros_like(weights)
    for dx, dy, bilinear in ((0, 0, (1-fraction[:, 0])*(1-fraction[:, 1])),
                             (1, 0, fraction[:, 0]*(1-fraction[:, 1])),
                             (0, 1, (1-fraction[:, 0])*fraction[:, 1]),
                             (1, 1, fraction[:, 0]*fraction[:, 1])):
        x, y = base[:, 0] + dx, base[:, 1] + dy
        valid = ((x >= 0) & (x < size) & (y >= 0) & (y < size)
                 & torch.isfinite(depth) & (depth > 1e-8))
        flat = y[valid] * size + x[valid]
        value = bilinear[valid] * visibility[valid]
        weights.scatter_add_(0, flat, value)
        depth_sum.scatter_add_(0, flat, value * depth[valid])
    occupancy = 1. - torch.exp(-1.6 * weights)
    return occupancy.reshape(size, size), (depth_sum / weights.clamp_min(1e-8)).reshape(size, size)


def optimize_saved_view_sim3(
    complete: np.ndarray, partial: np.ndarray, projector, *, diagonal: float,
    pixel_schedule=(8., 5., 3.), steps: int = 48, max_pairs: int = 8000,
    max_rotation_deg: float = 2., scale_bounds=(.985, 1.015),
    max_translation_ratio: float = .012, seed: int = 6145, device: str = "cuda",
    screen_size: int = 96, screen_points: int = 12000, screen_weight: float = .8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit one bounded proper Sim(3) using frozen saved-view correspondences."""
    complete = np.asarray(complete, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    diagonal = max(float(diagonal), 1e-8)
    rng = np.random.default_rng(int(seed))
    current = complete.copy()
    total = np.eye(4, dtype=np.float64)
    trace = []
    torch_device = torch.device(device)
    target_depth, target_mask = depth_v3.render_depth_mask(projector, partial, int(screen_size), splat=1)
    depth_values = target_depth[target_mask]
    z_min = float(np.quantile(depth_values, .01))
    z_span = max(float(np.quantile(depth_values, .99) - z_min), 1e-8)
    target_mask_t = torch.as_tensor(target_mask.astype(np.float32), device=torch_device)
    target_depth_t = torch.as_tensor(target_depth, dtype=torch.float32, device=torch_device)

    for stage, radius in enumerate(pixel_schedule):
        pairs = soft_ray_correspondences(
            partial, current, projector, pixel_radius=float(radius), trim_quantile=.75,
            max_distance_ratio=.14, bbox_diagonal=diagonal)
        partial_ids = np.asarray(pairs["partial_ids"], dtype=np.int64)
        complete_ids = np.asarray(pairs["generated_ids"], dtype=np.int64)
        if len(partial_ids) < 96:
            trace.append({"radius": float(radius), "accepted": False, "reason": "insufficient_pairs"})
            continue
        if len(partial_ids) > int(max_pairs):
            chosen = np.sort(rng.choice(len(partial_ids), size=int(max_pairs), replace=False))
            partial_ids, complete_ids = partial_ids[chosen], complete_ids[chosen]

        center = np.median(current, axis=0)
        screen_ids = (rng.choice(len(current), size=min(int(screen_points), len(current)), replace=False)
                      if len(current) > int(screen_points) else np.arange(len(current)))
        source = torch.as_tensor(current[complete_ids], dtype=torch.float32, device=torch_device)
        screen_source = torch.as_tensor(current[screen_ids], dtype=torch.float32, device=torch_device)
        target = torch.as_tensor(partial[partial_ids], dtype=torch.float32, device=torch_device)
        target_uv = _project(target, projector).detach()
        center_t = torch.as_tensor(center, dtype=torch.float32, device=torch_device)
        log_scale = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device=torch_device))
        rotvec = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=torch_device))
        translation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=torch_device))
        optimizer = torch.optim.Adam((log_scale, rotvec, translation), lr=.012)
        best = None
        for iteration in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            rotation = _rotation(rotvec)
            moved = torch.exp(log_scale) * ((source - center_t) @ rotation.T) + center_t + translation * diagonal
            surface = F.smooth_l1_loss((moved - target) / diagonal, torch.zeros_like(moved), beta=.02)
            reprojection = F.smooth_l1_loss(_project(moved, projector) / 512., target_uv / 512., beta=.01)
            moved_screen = torch.exp(log_scale) * ((screen_source - center_t) @ rotation.T) + center_t + translation * diagonal
            occupancy, rendered_depth = _soft_screen(moved_screen, projector, size=int(screen_size), z_min=z_min, z_span=z_span)
            silhouette = F.l1_loss(occupancy, target_mask_t)
            overlap = occupancy.detach().clamp(min=.05) * target_mask_t
            screen_depth = (torch.abs(rendered_depth - target_depth_t).clamp(max=.25*z_span) / z_span * overlap).sum() / overlap.sum().clamp_min(1.)
            regularizer = .02 * log_scale.square().sum() + .01 * rotvec.square().sum() + .01 * translation.square().sum()
            loss = 3. * surface + reprojection + float(screen_weight) * (silhouette + .75 * screen_depth) + regularizer
            loss.backward(); optimizer.step()
            with torch.no_grad():
                log_scale.clamp_(math.log(float(scale_bounds[0])), math.log(float(scale_bounds[1])))
                angle = torch.linalg.norm(rotvec); maximum = math.radians(float(max_rotation_deg))
                if angle > maximum:
                    rotvec.mul_(maximum / angle)
                norm = torch.linalg.norm(translation)
                if norm > float(max_translation_ratio):
                    translation.mul_(float(max_translation_ratio) / norm)
                value = float(loss.detach())
                if best is None or value < best["loss"]:
                    best = {"loss": value, "iteration": int(iteration),
                            "log_scale": float(log_scale.detach()),
                            "rotvec": rotvec.detach().cpu().numpy().copy(),
                            "translation": translation.detach().cpu().numpy().copy(),
                            "surface": float(surface.detach()), "reprojection": float(reprojection.detach())}
        scale = float(np.exp(best["log_scale"]))
        rotation = _rotation(torch.as_tensor(best["rotvec"], dtype=torch.float32, device=torch_device)).detach().cpu().numpy()
        shift = np.asarray(best["translation"], dtype=np.float64) * diagonal
        linear = scale * rotation
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = linear
        delta[:3, 3] = center + shift - linear @ center
        current = current @ linear.T + delta[:3, 3]
        total = delta @ total
        trace.append({"radius": float(radius), "accepted": True, "pairs": int(len(partial_ids)),
                      "best": {**best, "rotvec": best["rotvec"].tolist(), "translation": best["translation"].tolist()},
                      "delta_scale": scale})
    return current, total, {"method": "gpu_saved_view_screen_surface_proper_sim3_tto", "stages": trace}
