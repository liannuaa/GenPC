import contextlib
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDATOR_ROOT = PROJECT_ROOT / "third_party" / "OverlapPredator"
DEFAULT_INDOOR_CONFIG = PREDATOR_ROOT / "configs" / "test" / "indoor.yaml"
DEFAULT_INDOOR_WEIGHTS = PREDATOR_ROOT / "weights" / "indoor.pth"


def _ensure_predator_paths():
    root = str(PREDATOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _unit_sphere(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    scale = float(np.max(np.linalg.norm(points - center, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return ((points - center) / scale).astype(np.float32), center, scale


def _build_architecture(config):
    architecture = ["simple", "resnetb"]
    for _ in range(int(config.num_layers) - 1):
        architecture += ["resnetb_strided", "resnetb", "resnetb"]
    for _ in range(int(config.num_layers) - 2):
        architecture += ["nearest_upsample", "unary"]
    architecture += ["nearest_upsample", "last_unary"]
    config.architecture = architecture


class _PairDataset(Dataset):
    def __init__(self, config, src_points, ref_points):
        self.config = config
        self.src_points = np.asarray(src_points, dtype=np.float32)
        self.ref_points = np.asarray(ref_points, dtype=np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        src_feats = np.ones_like(self.src_points[:, :1], dtype=np.float32)
        ref_feats = np.ones_like(self.ref_points[:, :1], dtype=np.float32)
        return (
            self.src_points,
            self.ref_points,
            src_feats,
            ref_feats,
            np.eye(3, dtype=np.float32),
            np.ones((3, 1), dtype=np.float32),
            torch.ones(1, 2).long(),
            self.src_points,
            self.ref_points,
            torch.ones(1),
        )


def _to_device(inputs, device):
    out = {}
    for key, value in inputs.items():
        if isinstance(value, list):
            out[key] = [item.to(device) for item in value]
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


@contextlib.contextmanager
def _pushd(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _as_o3d_feature(features):
    reg = o3d.pipelines.registration
    out = reg.Feature()
    out.data = np.asarray(features, dtype=np.float64).T
    return out


def _as_o3d_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def _run_feature_ransac(src_points, ref_points, src_feats, ref_feats, distance_threshold, ransac_n):
    reg = o3d.pipelines.registration
    src_pcd = _as_o3d_pcd(src_points)
    ref_pcd = _as_o3d_pcd(ref_points)
    src_feature = _as_o3d_feature(src_feats)
    ref_feature = _as_o3d_feature(ref_feats)
    checkers = [
        reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        reg.CorrespondenceCheckerBasedOnDistance(float(distance_threshold)),
    ]
    criteria = reg.RANSACConvergenceCriteria(50000, 1000)
    estimation = reg.TransformationEstimationPointToPoint(False)
    try:
        result = reg.registration_ransac_based_on_feature_matching(
            src_pcd,
            ref_pcd,
            src_feature,
            ref_feature,
            False,
            float(distance_threshold),
            estimation,
            int(ransac_n),
            checkers,
            criteria,
        )
    except TypeError:
        result = reg.registration_ransac_based_on_feature_matching(
            src_pcd,
            ref_pcd,
            src_feature,
            ref_feature,
            float(distance_threshold),
            estimation,
            int(ransac_n),
            checkers,
            criteria,
        )
    return result


def _sample_by_scores(points, features, scores, count, seed):
    points = torch.as_tensor(points).detach().cpu().numpy()
    features = torch.as_tensor(features).detach().cpu().numpy()
    scores = torch.as_tensor(scores).detach().cpu().numpy().reshape(-1)
    count = min(int(count), len(points))
    if len(points) <= count:
        return points, features
    scores = np.maximum(scores, 0.0)
    if float(scores.sum()) <= 1e-12:
        probs = None
    else:
        probs = scores / scores.sum()
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(points), size=count, replace=False, p=probs)
    return points[indices], features[indices]


def _load_model(config_path, weights_path, device):
    _ensure_predator_paths()
    from lib.utils import load_config
    from models.architectures import KPFCNN

    config = edict(load_config(str(config_path)))
    config.gpu_mode = str(device).startswith("cuda")
    config.device = torch.device(device)
    _build_architecture(config)
    model = KPFCNN(config).to(config.device)
    state = torch.load(str(weights_path), map_location=config.device, weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return config, model


def estimate_predator_transform(
    ref_pcd,
    src_pcd,
    config_path=DEFAULT_INDOOR_CONFIG,
    weights_path=DEFAULT_INDOOR_WEIGHTS,
    device="cuda",
    n_points=1000,
    distance_threshold=0.05,
    ransac_n=3,
    seed=7351,
    neighborhood_limits=(64, 64, 64, 64),
):
    """Estimate a source-to-reference transform with PREDATOR features.

    Source and reference are independently normalized to a unit sphere before
    PREDATOR inference, then the RANSAC transform is converted back to the
    original coordinates. This keeps the feature extractor in its expected
    metric range while allowing one uniform scale between the two clouds.
    """
    if not PREDATOR_ROOT.exists():
        raise FileNotFoundError(f"OverlapPredator repo not found: {PREDATOR_ROOT}")
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"PREDATOR weights not found: {weights_path}")

    ref_points = np.asarray(ref_pcd.points, dtype=np.float64)
    src_points = np.asarray(src_pcd.points, dtype=np.float64)
    ref_norm, ref_center, ref_scale = _unit_sphere(ref_points)
    src_norm, src_center, src_scale = _unit_sphere(src_points)

    _ensure_predator_paths()
    from datasets.dataloader import collate_fn_descriptor

    config, model = _load_model(config_path, weights_path, device)
    dataset = _PairDataset(config, src_norm, ref_norm)
    with _pushd(PREDATOR_ROOT):
        inputs = collate_fn_descriptor(
            [dataset[0]],
            config=config,
            neighborhood_limits=list(neighborhood_limits),
        )
    inputs = _to_device(inputs, config.device)

    with torch.no_grad():
        feats, scores_overlap, scores_saliency = model(inputs)

    points = inputs["points"][0].detach().cpu()
    len_src = int(inputs["stack_lengths"][0][0].item())
    src_features = torch.nn.functional.normalize(feats[:len_src].detach().cpu(), p=2, dim=1)
    ref_features = torch.nn.functional.normalize(feats[len_src:].detach().cpu(), p=2, dim=1)
    src_scores = (scores_overlap[:len_src] * scores_saliency[:len_src]).detach().cpu()
    ref_scores = (scores_overlap[len_src:] * scores_saliency[len_src:]).detach().cpu()
    src_feat_points = points[:len_src]
    ref_feat_points = points[len_src:]

    sampled_src, sampled_src_features = _sample_by_scores(
        src_feat_points, src_features, src_scores, n_points, seed
    )
    sampled_ref, sampled_ref_features = _sample_by_scores(
        ref_feat_points, ref_features, ref_scores, n_points, seed + 1
    )

    result = _run_feature_ransac(
        sampled_src,
        sampled_ref,
        sampled_src_features,
        sampled_ref_features,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
    )
    transform_norm = np.asarray(result.transformation, dtype=np.float64)
    rotation = transform_norm[:3, :3]
    translation = transform_norm[:3, 3]
    uniform_scale = ref_scale / max(src_scale, 1e-12)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = uniform_scale * rotation
    transform[:3, 3] = ref_center + ref_scale * translation - uniform_scale * rotation @ src_center

    info = {
        "backend": "predator_indoor",
        "config": str(config_path),
        "weights": str(weights_path),
        "n_points": int(n_points),
        "distance_threshold": float(distance_threshold),
        "ransac_n": int(ransac_n),
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "ref_center": ref_center.tolist(),
        "src_center": src_center.tolist(),
        "ref_scale": float(ref_scale),
        "src_scale": float(src_scale),
        "uniform_scale": float(uniform_scale),
        "sampled_src_points": int(len(sampled_src)),
        "sampled_ref_points": int(len(sampled_ref)),
    }
    return transform, info


def transformed_point_cloud(pcd, transform):
    output = deepcopy(pcd)
    output.transform(transform)
    return output
