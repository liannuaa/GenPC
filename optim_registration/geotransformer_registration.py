import importlib
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOTRANSFORMER_ROOT = PROJECT_ROOT / "third_party" / "GeoTransformer"
MODELNET_EXP_DIR = (
    GEOTRANSFORMER_ROOT
    / "experiments"
    / "geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn"
)
DEFAULT_MODELNET_WEIGHTS = GEOTRANSFORMER_ROOT / "weights" / "geotransformer-modelnet.pth.tar"


def _ensure_geotransformer_paths():
    for path in (GEOTRANSFORMER_ROOT, MODELNET_EXP_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _normalize_points(points):
    center = points.mean(axis=0)
    scale = np.max(np.linalg.norm(points - center, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return ((points - center) / scale).astype(np.float32), center.astype(np.float64), float(scale)


def _sample_points(points, num_points, seed):
    rng = np.random.default_rng(seed)
    count = len(points)
    if count == 0:
        raise ValueError("GeoTransformer input point cloud is empty.")
    if count >= num_points:
        indices = rng.choice(count, size=num_points, replace=False)
    else:
        indices = rng.choice(count, size=num_points, replace=True)
    return points[indices]


def _load_model(weights_path, device):
    _ensure_geotransformer_paths()
    config_module = importlib.import_module("config")
    model_module = importlib.import_module("model")

    cfg = config_module.make_cfg()
    model = model_module.create_model(cfg).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, model


def _to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_device(value, device) for value in data]
    return data


def estimate_geotransformer_transform(
    ref_pcd,
    src_pcd,
    weights_path=DEFAULT_MODELNET_WEIGHTS,
    device="cuda",
    num_points=717,
    seed=7351,
):
    """Estimate a source-to-reference transform using the ModelNet GeoTransformer.

    The ModelNet pipeline normalizes object points to a unit sphere and samples 717
    points. We normalize source and reference independently, run the model in that
    normalized space, then convert the result back to this project's coordinates.
    Independent normalization lets the rigid model absorb a uniform scale mismatch.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"GeoTransformer weights not found: {weights_path}")
    if not GEOTRANSFORMER_ROOT.exists():
        raise FileNotFoundError(f"GeoTransformer repo not found: {GEOTRANSFORMER_ROOT}")

    ref_points = np.asarray(ref_pcd.points, dtype=np.float64)
    src_points = np.asarray(src_pcd.points, dtype=np.float64)
    ref_norm, ref_center, ref_scale = _normalize_points(ref_points)
    src_norm, src_center, src_scale = _normalize_points(src_points)

    ref_sample = _sample_points(ref_norm, int(num_points), int(seed)).astype(np.float32)
    src_sample = _sample_points(src_norm, int(num_points), int(seed) + 1).astype(np.float32)

    _ensure_geotransformer_paths()
    data_module = importlib.import_module("geotransformer.utils.data")
    torch_module = importlib.import_module("geotransformer.utils.torch")

    cfg, model = _load_model(str(weights_path), device)
    neighbor_limits = list(getattr(cfg, "neighbor_limits", [55, 55, 55]))
    if len(neighbor_limits) != int(cfg.backbone.num_stages):
        neighbor_limits = [55] * int(cfg.backbone.num_stages)

    data_dict = {
        "ref_points": ref_sample,
        "src_points": src_sample,
        "ref_feats": np.ones_like(ref_sample[:, :1], dtype=np.float32),
        "src_feats": np.ones_like(src_sample[:, :1], dtype=np.float32),
        "transform": np.eye(4, dtype=np.float32),
    }
    data_dict = data_module.registration_collate_fn_stack_mode(
        [data_dict],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
    )
    data_dict = _to_device(data_dict, device)

    with torch.no_grad():
        output_dict = model(data_dict)

    output_dict = torch_module.release_cuda(output_dict)
    transform_norm = np.asarray(output_dict["estimated_transform"], dtype=np.float64)
    corr_scores = np.asarray(output_dict.get("corr_scores", []))

    rotation = transform_norm[:3, :3]
    translation = transform_norm[:3, 3]
    scale = ref_scale / src_scale
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = ref_center + ref_scale * translation - scale * rotation @ src_center

    info = {
        "backend": "geotransformer_modelnet",
        "num_points": int(num_points),
        "num_correspondences": int(corr_scores.shape[0]) if corr_scores.ndim > 0 else 0,
        "ref_center": ref_center.tolist(),
        "src_center": src_center.tolist(),
        "ref_scale": ref_scale,
        "src_scale": src_scale,
        "uniform_scale": float(scale),
    }
    return transform, info


def transformed_point_cloud(pcd, transform):
    output = o3d.geometry.PointCloud(pcd)
    output.transform(transform)
    return output
