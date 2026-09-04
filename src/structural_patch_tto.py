"""Structure-aware proper Sim(3) test-time refinement.

The action is category-free: local planar/low-curvature patches are inferred
from neighbourhood covariance, then matched only through the existing frozen
saved camera.  It is a registration action, not a non-rigid edit.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from src.ray_consistent_registration import apply_transform, soft_ray_correspondences
from src.saved_view_tto import _project, _rotation


def estimate_normals_planarity(points: np.ndarray, *, neighbours: int = 20,
                               batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Return unsigned local normals and PCA planarity in the source frame."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("points must have shape (N, 3) with N >= 4")
    count = min(max(int(neighbours), 4), len(points))
    tree = cKDTree(points)
    normals = np.empty_like(points)
    planarity = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), int(batch_size)):
        stop = min(start + int(batch_size), len(points))
        _, ids = tree.query(points[start:stop], k=count, workers=-1)
        local = points[ids] - points[ids].mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", local, local) / max(count - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        normals[start:stop] = vectors[:, :, 0]
        planarity[start:stop] = (values[:, 1] - values[:, 0]) / np.maximum(values[:, 2], 1e-12)
    # Normal sign is irrelevant in the loss; normalize for numerical safety.
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals, np.clip(planarity, 0., 1.)


def optimize_structural_patch_sim3(
    complete: np.ndarray, partial: np.ndarray, projector, *, diagonal: float,
    pixel_schedule=(5., 3.), steps: int = 48, max_pairs: int = 6000,
    max_rotation_deg: float = 1.5, scale_bounds=(.99, 1.01),
    max_translation_ratio: float = .008, planarity_quantile: float = .60,
    seed: int = 6145, device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use visible planar/low-curvature patches to refine a bounded proper Sim(3)."""
    complete = np.asarray(complete, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    diagonal = max(float(diagonal), 1e-8)
    partial_normal, partial_planarity = estimate_normals_planarity(partial)
    current = complete.copy()
    current_normal, current_planarity = estimate_normals_planarity(current)
    total = np.eye(4, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    trace = []
    for radius in pixel_schedule:
        pairs = soft_ray_correspondences(
            partial, current, projector, pixel_radius=float(radius), trim_quantile=.75,
            max_distance_ratio=.14, bbox_diagonal=diagonal)
        pids = np.asarray(pairs["partial_ids"], dtype=np.int64)
        cids = np.asarray(pairs["generated_ids"], dtype=np.int64)
        if len(pids) < 96:
            trace.append({"radius": float(radius), "accepted": False, "reason": "insufficient_pairs"})
            continue
        p_limit = float(np.quantile(partial_planarity[pids], float(planarity_quantile)))
        c_limit = float(np.quantile(current_planarity[cids], float(planarity_quantile)))
        compatible = ((partial_planarity[pids] >= p_limit) & (current_planarity[cids] >= c_limit)
                      & (np.abs(np.sum(partial_normal[pids] * current_normal[cids], axis=1)) >= .45))
        pids, cids = pids[compatible], cids[compatible]
        if len(pids) < 96:
            trace.append({"radius": float(radius), "accepted": False, "reason": "insufficient_structural_pairs"})
            continue
        if len(pids) > int(max_pairs):
            selected = np.sort(rng.choice(len(pids), int(max_pairs), replace=False))
            pids, cids = pids[selected], cids[selected]
        centre = np.median(current, axis=0)
        source = torch.as_tensor(current[cids], dtype=torch.float32, device=device)
        source_normal = torch.as_tensor(current_normal[cids], dtype=torch.float32, device=device)
        target = torch.as_tensor(partial[pids], dtype=torch.float32, device=device)
        target_normal = torch.as_tensor(partial_normal[pids], dtype=torch.float32, device=device)
        target_uv = _project(target, projector).detach()
        centre_t = torch.as_tensor(centre, dtype=torch.float32, device=device)
        log_scale = torch.nn.Parameter(torch.zeros(1, device=device))
        rotvec = torch.nn.Parameter(torch.zeros(3, device=device))
        translation = torch.nn.Parameter(torch.zeros(3, device=device))
        optimizer = torch.optim.Adam((log_scale, rotvec, translation), lr=.01)
        best = None
        for iteration in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            rotation = _rotation(rotvec)
            moved = torch.exp(log_scale) * ((source - centre_t) @ rotation.T) + centre_t + translation * diagonal
            moved_normal = source_normal @ rotation.T
            plane = F.smooth_l1_loss(
                ((moved - target) * target_normal).sum(dim=1) / diagonal,
                torch.zeros(len(target), device=device), beta=.015)
            point = F.smooth_l1_loss((moved - target) / diagonal, torch.zeros_like(moved), beta=.02)
            normal = (1. - torch.abs((moved_normal * target_normal).sum(dim=1)).clamp(max=1.)).mean()
            reprojection = F.smooth_l1_loss(_project(moved, projector) / 512., target_uv / 512., beta=.01)
            regularizer = .03 * log_scale.square().sum() + .02 * rotvec.square().sum() + .02 * translation.square().sum()
            loss = 4. * plane + .25 * point + .15 * normal + reprojection + regularizer
            loss.backward(); optimizer.step()
            with torch.no_grad():
                log_scale.clamp_(math.log(float(scale_bounds[0])), math.log(float(scale_bounds[1])))
                angle = torch.linalg.norm(rotvec); maximum = math.radians(float(max_rotation_deg))
                if angle > maximum:
                    rotvec.mul_(maximum / angle)
                length = torch.linalg.norm(translation)
                if length > float(max_translation_ratio):
                    translation.mul_(float(max_translation_ratio) / length)
                if best is None or float(loss) < best["loss"]:
                    best = {"loss": float(loss), "iteration": int(iteration),
                            "log_scale": float(log_scale), "rotvec": rotvec.detach().cpu().numpy().copy(),
                            "translation": translation.detach().cpu().numpy().copy(),
                            "plane": float(plane), "normal": float(normal)}
        scale = float(np.exp(best["log_scale"]))
        rotation = _rotation(torch.as_tensor(best["rotvec"], dtype=torch.float32, device=device)).detach().cpu().numpy()
        shift = np.asarray(best["translation"], dtype=np.float64) * diagonal
        linear = scale * rotation
        delta = np.eye(4, dtype=np.float64); delta[:3, :3] = linear; delta[:3, 3] = centre + shift - linear @ centre
        current = apply_transform(current, delta)
        current_normal = current_normal @ rotation.T
        total = delta @ total
        trace.append({"radius": float(radius), "accepted": True, "pairs": int(len(pids)),
                      "best": {**best, "rotvec": best["rotvec"].tolist(), "translation": best["translation"].tolist()}})
    return current, total, {"method": "saved_view_structural_patch_proper_sim3", "stages": trace}
