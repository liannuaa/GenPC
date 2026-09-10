"""Camera-1 rendered 2D-to-visible-3D Sim(3) capture for new 3D priors."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as functional

from src.bidirectional_cycle_registration import prepare_visible_target, visible_score


def _camera_matrix(projector) -> tuple[np.ndarray, np.ndarray]:
    # Kaolin's ``camera.transform`` includes the projection convention used
    # when depth.png was produced; public R/t alone do not reproduce it.
    # Recover the exact row-vector affine map from origin and basis probes.
    probe = torch.as_tensor(
        np.concatenate([np.zeros((1, 3)), np.eye(3)], axis=0),
        dtype=torch.float32, device=projector.device,
    )
    with torch.no_grad():
        mapped = projector.camera.transform(probe).detach().float().cpu().numpy().astype(np.float64)
    offset = mapped[0]
    linear = mapped[1:] - offset
    if abs(float(np.linalg.det(linear))) < 1e-10:
        raise ValueError("saved Camera-1 transform is singular")
    return linear, offset


def _target_camera_statistics(partial: np.ndarray, projector) -> dict[str, np.ndarray | float]:
    linear, offset = _camera_matrix(projector)
    camera_points = np.asarray(partial, dtype=np.float64) @ linear + offset
    lower, upper = np.quantile(camera_points, (0.02, 0.98), axis=0)
    return {
        "centre": np.median(camera_points, axis=0),
        "extent": np.maximum(upper - lower, 1e-8),
        "linear": linear,
        "offset": offset,
    }


def _coarse_target_mask(partial: np.ndarray, projector, resolution: int, device: str) -> torch.Tensor:
    uv, _ = projector.project(partial)
    height, width = projector.image_shape
    uv = uv / np.array([width - 1, height - 1], dtype=np.float64)
    pixels = np.floor(uv * resolution).astype(np.int64)
    valid = np.all((pixels >= 0) & (pixels < resolution), axis=1)
    flat = pixels[valid, 1] * resolution + pixels[valid, 0]
    mask = torch.zeros(resolution * resolution, dtype=torch.float32, device=device)
    mask[torch.as_tensor(flat, dtype=torch.long, device=device)] = 1.0
    return functional.max_pool2d(mask.view(1, 1, resolution, resolution), 5, 1, 2).view(-1)


def _semantic_target(
    image_path,
    partial: np.ndarray,
    projector,
    resolution: int,
    device: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Convert a white-background Camera-1 condition into metric screen evidence."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    foreground = np.any(rgb < 245, axis=2)
    foreground_t = torch.as_tensor(foreground, dtype=torch.float32)[None, None]
    foreground_t = functional.max_pool2d(foreground_t, 3, 1, 1)
    ys, xs = np.nonzero(foreground_t[0, 0].numpy() > 0.5)
    if len(xs) < 64:
        raise ValueError(f"no reliable foreground in Camera-1 condition: {image_path}")
    height, width = foreground.shape
    lower_uv = np.array([xs.min() / width, ys.min() / height], dtype=np.float64)
    upper_uv = np.array([(xs.max() + 1) / width, (ys.max() + 1) / height], dtype=np.float64)
    centre_uv = 0.5 * (lower_uv + upper_uv)
    extent_uv = np.maximum(upper_uv - lower_uv, 1e-6)

    partial_stats = _target_camera_statistics(partial, projector)
    centre_camera = np.asarray(partial_stats["centre"], dtype=np.float64).copy()
    centre_camera[0] = projector.center_xy[0] + (
        (centre_uv[0] - 0.5) * projector.scale_xy / (1.0 - 2.0 * projector.padding)
    )
    centre_camera[1] = projector.center_xy[1] + (
        ((1.0 - centre_uv[1]) - 0.5) * projector.scale_xy / (1.0 - 2.0 * projector.padding)
    )
    extent_camera = np.asarray(partial_stats["extent"], dtype=np.float64).copy()
    extent_camera[:2] = extent_uv * projector.scale_xy / (1.0 - 2.0 * projector.padding)
    resized = functional.interpolate(
        foreground_t, size=(resolution, resolution), mode="nearest",
    ).to(device=device).reshape(-1)
    return resized, centre_camera, extent_camera


def _random_rotations(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(int(count), random_state=rng).as_matrix()
    rotations = np.concatenate([np.eye(3, dtype=np.float64)[None], rotations], axis=0)
    return rotations


def camera1_rendered_sim3_capture(
    source: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    camera1_condition=None,
    seed: int = 6145,
    rotation_candidates: int = 4_096,
    source_samples: int = 8_192,
    render_resolution: int = 128,
    shortlist: int = 48,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """Capture a regenerated prior directly in the partial Camera-1 frame.

    Each SO(3) hypothesis receives an analytic isotropic scale and translation
    from robust Camera-1 projected extents and centres.  A low-resolution soft
    raster shortlist is then reranked with the existing full-resolution
    silhouette, visible-depth and pixel-indexed 3D evidence.  No assumption of
    full-shape equality with the previous prior is made.
    """
    source = np.asarray(source, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if min(len(source), len(partial)) < 64:
        raise ValueError("source and partial need at least 64 points")
    if rotation_candidates < 24 or source_samples < 512 or shortlist < 1:
        raise ValueError("insufficient Camera-1 capture candidates or samples")
    if render_resolution < 32:
        raise ValueError("render resolution must be at least 32")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(len(source), min(int(source_samples), len(source)), replace=False)
    sampled = source[sample_ids]
    source_centre = np.median(sampled, axis=0)
    centred = sampled - source_centre
    target = _target_camera_statistics(partial, projector)
    camera_linear = np.asarray(target["linear"])
    camera_offset = np.asarray(target["offset"])
    target_centre = np.asarray(target["centre"])
    target_extent = np.asarray(target["extent"])
    if camera1_condition is None:
        target_mask = _coarse_target_mask(partial, projector, render_resolution, device)
        target_kind = "partial_projection"
    else:
        target_mask, target_centre, target_extent = _semantic_target(
            camera1_condition, partial, projector, render_resolution, device,
        )
        target_kind = "complete_camera1_condition"
    target_count = target_mask.sum()
    rotations = _random_rotations(rotation_candidates, seed)

    points_t = torch.as_tensor(centred, dtype=torch.float32, device=device)
    camera_linear_t = torch.as_tensor(camera_linear, dtype=torch.float32, device=device)
    target_centre_t = torch.as_tensor(target_centre, dtype=torch.float32, device=device)
    target_extent_t = torch.as_tensor(target_extent, dtype=torch.float32, device=device)
    centre_xy_t = torch.as_tensor(projector.center_xy, dtype=torch.float32, device=device)
    records: list[dict] = []
    batch_size = 24 if device.startswith("cuda") else 4
    for start in range(0, len(rotations), batch_size):
        rotation_batch = torch.as_tensor(
            rotations[start:start + batch_size], dtype=torch.float32, device=device,
        )
        world_relative = torch.einsum("nj,bkj->bnk", points_t, rotation_batch)
        camera_relative = torch.einsum("bnj,jk->bnk", world_relative, camera_linear_t)
        camera_median = camera_relative.median(dim=1).values
        quantiles = torch.quantile(camera_relative, torch.tensor([0.02, 0.98], device=device), dim=1)
        source_extent = (quantiles[1] - quantiles[0]).clamp_min(1e-6)
        width_scale = target_extent_t[0] / source_extent[:, 0]
        height_scale = target_extent_t[1] / source_extent[:, 1]
        geometric_scale = torch.sqrt(width_scale * height_scale)
        scale_sets = torch.stack([width_scale, geometric_scale, height_scale], dim=1)
        # A partial silhouette can omit one side.  Bound the three analytic
        # choices around their median rather than around any category size.
        median_scale = scale_sets.median(dim=1, keepdim=True).values
        scale_sets = torch.clamp(scale_sets, median_scale * 0.72, median_scale * 1.38)
        for scale_index in range(scale_sets.shape[1]):
            scales = scale_sets[:, scale_index]
            camera_points = (
                (camera_relative - camera_median[:, None, :]) * scales[:, None, None]
                + target_centre_t[None, None, :]
            )
            uv = (camera_points[..., :2] - centre_xy_t) / float(projector.scale_xy)
            uv = uv * (1.0 - 2.0 * float(projector.padding)) + 0.5
            uv[..., 1] = 1.0 - uv[..., 1]
            pixels = torch.floor(uv * int(render_resolution)).long()
            valid = (
                (pixels[..., 0] >= 0) & (pixels[..., 0] < render_resolution)
                & (pixels[..., 1] >= 0) & (pixels[..., 1] < render_resolution)
            )
            flat = pixels[..., 1] * render_resolution + pixels[..., 0]
            flat = torch.where(valid, flat, torch.zeros_like(flat))
            masks = torch.zeros(
                (len(rotation_batch), render_resolution * render_resolution),
                dtype=torch.float32, device=device,
            )
            masks.scatter_(1, flat, valid.float())
            masks = functional.max_pool2d(
                masks.view(-1, 1, render_resolution, render_resolution), 3, 1, 1,
            ).view(len(rotation_batch), -1)
            intersection = (masks * target_mask).sum(dim=1)
            source_count = masks.sum(dim=1).clamp_min(1.0)
            union = source_count + target_count - intersection
            iou = intersection / union.clamp_min(1.0)
            coverage = intersection / target_count.clamp_min(1.0)
            leakage = (source_count - intersection) / source_count
            scaled_depth_extent = source_extent[:, 2] * scales
            depth_extent_error = torch.abs(torch.log(scaled_depth_extent / target_extent_t[2])).clamp_max(2.0)
            energy = (
                0.68 * (1.0 - iou)
                + 0.14 * (1.0 - coverage)
                + 0.14 * leakage
                + 0.04 * depth_extent_error
            )
            for batch_index in range(len(rotation_batch)):
                records.append({
                    "rotation_index": int(start + batch_index),
                    "scale_source": ("width", "geometric", "height")[scale_index],
                    "scale": float(scales[batch_index].item()),
                    "camera_relative_median": camera_median[batch_index].detach().cpu().numpy(),
                    "coarse_energy": float(energy[batch_index].item()),
                    "coarse_iou": float(iou[batch_index].item()),
                    "coarse_coverage": float(coverage[batch_index].item()),
                    "coarse_leakage": float(leakage[batch_index].item()),
                    "coarse_depth_extent_error": float(depth_extent_error[batch_index].item()),
                })
    records.sort(key=lambda item: item["coarse_energy"])
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    target_cache = prepare_visible_target(partial, projector)
    best_coarse = max(records[0]["coarse_iou"], 1e-8)
    for item in records[: min(shortlist, len(records))]:
        rotation = rotations[item["rotation_index"]]
        scale = float(item["scale"])
        camera_median = np.asarray(item["camera_relative_median"])
        target_world_centre = (
            target_centre - camera_offset - scale * camera_median
        ) @ np.linalg.inv(camera_linear)
        transformed = (source - source_centre) @ rotation.T * scale + target_world_centre
        score = visible_score(
            partial, transformed, projector, diagonal,
            pixel_radius=5.0, target_cache=target_cache,
        )
        item["full_objective"] = float(score["objective"])
        item["full_geometric"] = float(score["geometric"]["objective"])
        item["full_projection"] = {key: float(value) for key, value in score["projection"].items()}
        item["joint_objective"] = float(
            0.72 * item["coarse_energy"]
            + 0.28 * item["full_objective"]
            + 0.20 * max(0.0, best_coarse - item["coarse_iou"])
        )
    selected = min(records[: min(shortlist, len(records))], key=lambda item: item["joint_objective"])
    rotation = rotations[selected["rotation_index"]]
    scale = float(selected["scale"])
    camera_median = np.asarray(selected["camera_relative_median"])
    target_world_centre = (
        target_centre - camera_offset - scale * camera_median
    ) @ np.linalg.inv(camera_linear)
    result = (source - source_centre) @ rotation.T * scale + target_world_centre
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_world_centre - scale * rotation @ source_centre
    return result, {
        "method": "camera1_rendered_2d_to_visible3d_global_proper_sim3",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "previous_complete_prior_used": False,
        "camera1_target": target_kind,
        "camera1_condition": None if camera1_condition is None else str(camera1_condition),
        "rotation_candidates": int(len(rotations)),
        "scale_candidates_per_rotation": 3,
        "source_samples": int(len(sampled)),
        "render_resolution": int(render_resolution),
        "shortlist": int(min(shortlist, len(records))),
        "selected": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in selected.items()
        },
        "transform": transform.tolist(),
        "coarse_top": [
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in item.items()
            }
            for item in records[: min(16, len(records))]
        ],
    }
