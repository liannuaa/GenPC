"""Offline-only point-cloud completion evaluation.

This module is intentionally separate from generation, registration, and
Gaussian editing so ground truth cannot influence inference.

The MVP protocol uses the benchmark's native 2,048-point complete target and
an exactly 16,384-point prediction.  Unlike the legacy Redwood helper below,
it reports squared Chamfer distance (CD-L2) and F-score without EMD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from fpsample import fps_sampling

from loss_functions import chamfer_3DDist
from utils.loss_util import Completionloss


def _points(path: Path) -> np.ndarray:
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if not len(points):
        raise ValueError(f"Point cloud is empty: {path}")
    return points


def fscore_from_squared_distances(
    prediction_to_target: np.ndarray | torch.Tensor,
    target_to_prediction: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.01,
) -> tuple[float, float, float]:
    """Return precision, recall, and F-score for a Euclidean threshold.

    Chamfer's CUDA extension returns *squared* nearest-neighbour distances,
    while the standard MVP F-score@1% threshold is expressed in Euclidean
    coordinates.  Keeping this conversion here avoids a silent 100x error.
    """
    if threshold <= 0.0:
        raise ValueError("F-score threshold must be positive")
    threshold_squared = float(threshold) ** 2
    pred_dist = np.asarray(prediction_to_target.detach().cpu() if isinstance(prediction_to_target, torch.Tensor) else prediction_to_target)
    target_dist = np.asarray(target_to_prediction.detach().cpu() if isinstance(target_to_prediction, torch.Tensor) else target_to_prediction)
    if pred_dist.size == 0 or target_dist.size == 0:
        raise ValueError("F-score requires non-empty nearest-neighbour distances")
    precision = float(np.mean(pred_dist <= threshold_squared))
    recall = float(np.mean(target_dist <= threshold_squared))
    fscore = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, float(fscore)


def _fps_exact(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministically farthest-point sample a cloud to exactly ``count``."""
    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) point array, got {cloud.shape}")
    if count <= 0:
        raise ValueError("sample count must be positive")
    if len(cloud) < count:
        raise ValueError(f"Prediction has {len(cloud)} points, fewer than required {count}")
    if len(cloud) == count:
        return cloud
    # A fixed start index makes the paper-facing protocol reproducible.
    indices = fps_sampling(cloud, int(count), start_idx=0)
    return cloud[np.asarray(indices, dtype=np.int64)]


def evaluate_mvp_cd_l2_fscore(
    prediction: Path,
    complete_target: np.ndarray,
    *,
    prediction_count: int = 16384,
    fscore_threshold: float = 0.01,
    device: str = "cuda",
) -> dict[str, float | int]:
    """Evaluate one MVP prediction under the 16k CD-L2/F-score protocol.

    ``complete_target`` must be read from MVP's official H5 file.  It is kept
    at its native cardinality (currently 2,048) instead of being synthetically
    upsampled.  This function is offline-only and must never be imported by an
    inference or agent-decision path.
    """
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("MVP CD-L2/F-score evaluation requires the CUDA Chamfer extension")
    prediction_points = _points(Path(prediction))
    sampled_prediction = _fps_exact(prediction_points, int(prediction_count))
    target = np.asarray(complete_target, dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != 3 or not len(target):
        raise ValueError(f"Expected a non-empty MVP complete target of shape (N, 3), got {target.shape}")

    pred_tensor = torch.from_numpy(sampled_prediction).unsqueeze(0).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(target).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_to_target, target_to_pred, _, _ = chamfer_3DDist()(pred_tensor, target_tensor)
    cd_l2 = float(pred_to_target.mean().item() + target_to_pred.mean().item())
    precision, recall, fscore = fscore_from_squared_distances(
        pred_to_target.squeeze(0), target_to_pred.squeeze(0), threshold=float(fscore_threshold),
    )
    return {
        "prediction_points_before_sampling": int(len(prediction_points)),
        "prediction_points": int(len(sampled_prediction)),
        "complete_points": int(len(target)),
        "cd_l2": cd_l2,
        "cd_l2_x1e4": cd_l2 * 1.0e4,
        "fscore_1pct": fscore,
        "precision_1pct": precision,
        "recall_1pct": recall,
    }


def evaluate_cd_emd(prediction: Path, ground_truth: Path, *, count: int = 16384, seed: int = 6145) -> tuple[float, float]:
    """Measure a completed point cloud after inference has finished."""
    pred, gt = _points(prediction), _points(ground_truth)
    count = min(int(count), len(pred), len(gt))
    if count <= 0:
        raise ValueError("Cannot evaluate an empty point cloud")
    rng = np.random.default_rng(int(seed) % (2**32))
    pred_idx = fps_sampling(pred, count, start_idx=int(rng.integers(0, len(pred))))
    gt_idx = fps_sampling(gt, count, start_idx=int(rng.integers(0, len(gt))))
    pred_tensor = torch.from_numpy(pred[pred_idx]).unsqueeze(0).float().cuda()
    gt_tensor = torch.from_numpy(gt[gt_idx]).unsqueeze(0).float().cuda()
    cd = Completionloss(loss_func="cd_l1").get_loss(gen=pred_tensor, gt=gt_tensor)
    emd = Completionloss(loss_func="emd").get_loss(gen=pred_tensor, gt=gt_tensor)
    return float(cd.item()), float(emd.item())
