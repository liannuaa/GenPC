"""Offline-only Redwood CD-L1 and EMD evaluation.

This module is intentionally separate from generation, registration, and
Gaussian editing so ground truth cannot influence inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from fpsample import fps_sampling

from utils.loss_util import Completionloss


def _points(path: Path) -> np.ndarray:
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if not len(points):
        raise ValueError(f"Point cloud is empty: {path}")
    return points


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
