import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optim_registration.geotransformer_registration import estimate_geotransformer_transform


DEFAULT_SAMPLE_DIR = (
    PROJECT_ROOT
    / "workspace"
    / "scansalon_hunyuan3d_omni_car__132_geotransformer_only"
    / "car__132"
)
DEFAULT_MOGEV2_MODEL = PROJECT_ROOT / "models" / "moge-2-vitl"


def pcd_to_numpy(pcd):
    return np.asarray(pcd.points, dtype=np.float64)


def numpy_to_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def build_foreground_mask(points, mask, image_rgb, white_threshold=248, image_mask=None):
    valid = np.asarray(mask, dtype=bool).copy()
    valid &= np.isfinite(points).all(axis=-1)
    valid &= np.linalg.norm(points, axis=-1) > 1e-8
    if image_mask is not None:
        valid &= np.asarray(image_mask, dtype=bool)
    image_rgb = np.asarray(image_rgb)
    near_white = (image_rgb >= int(white_threshold)).all(axis=-1)
    valid &= ~near_white
    return valid


def image_foreground_mask(image, white_threshold=248, alpha_threshold=8):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image, got shape {image.shape}")
    rgb = image[..., :3]
    near_white = (rgb >= int(white_threshold)).all(axis=-1)
    foreground = ~near_white
    if image.shape[-1] == 4:
        foreground &= image[..., 3] >= int(alpha_threshold)
    return foreground


def crop_to_foreground(image, foreground, padding=8):
    image = np.asarray(image)
    foreground = np.asarray(foreground, dtype=bool)
    if not foreground.any():
        return image, foreground, (0, 0, image.shape[1], image.shape[0])

    ys, xs = np.where(foreground)
    pad = max(0, int(padding))
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad + 1)
    y1 = min(image.shape[0], int(ys.max()) + pad + 1)
    return image[y0:y1, x0:x1], foreground[y0:y1, x0:x1], (x0, y0, x1, y1)


def image_to_rgb_for_moge(image):
    image = np.asarray(image)
    rgb = image[..., :3]
    if image.shape[-1] == 4:
        alpha = (image[..., 3:4].astype(np.float32) / 255.0).clip(0, 1)
        rgb = rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return np.rint(rgb).astype(np.uint8)
    return rgb


def maybe_subsample(points, colors, max_points, seed):
    if max_points is None or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(points), size=int(max_points), replace=False)
    return points[indices], colors[indices]


def bbox_diagonal(pcd):
    points = pcd_to_numpy(pcd)
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def pcd_center(pcd):
    points = pcd_to_numpy(pcd)
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")
    return points.mean(axis=0)


def initial_similarity_transform(source, target):
    source_diag = max(bbox_diagonal(source), 1e-8)
    target_diag = max(bbox_diagonal(target), 1e-8)
    scale = target_diag / source_diag
    source_center = pcd_center(source)
    target_center = pcd_center(target)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3) * scale
    transform[:3, 3] = target_center - scale * source_center
    return transform, float(scale)


def copy_pcd(pcd):
    return o3d.geometry.PointCloud(pcd)


def downsample_for_icp(pcd, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return copy_pcd(pcd)
    down = pcd.voxel_down_sample(float(voxel_size))
    if len(down.points) == 0:
        return copy_pcd(pcd)
    return down


def run_similarity_icp(source, target, max_correspondence_distance=0.05, voxel_size=0.01):
    init_transform, init_scale = initial_similarity_transform(source, target)
    source_icp = downsample_for_icp(source, voxel_size)
    target_icp = downsample_for_icp(target, voxel_size)
    result = o3d.pipelines.registration.registration_icp(
        source_icp,
        target_icp,
        float(max_correspondence_distance),
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    aligned = copy_pcd(source)
    aligned.transform(result.transformation)
    info = {
        "initial_scale": init_scale,
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "max_correspondence_distance": float(max_correspondence_distance),
        "voxel_size": None if voxel_size is None else float(voxel_size),
    }
    return aligned, np.asarray(result.transformation, dtype=np.float64), info


def nearest_neighbor_stats(source, target):
    source_points = pcd_to_numpy(source)
    target_points = pcd_to_numpy(target)
    if len(source_points) == 0 or len(target_points) == 0:
        return {"mean": None, "median": None, "p95": None}
    distances = cKDTree(target_points).query(source_points, k=1)[0]
    return {
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
    }


def make_visualization_cloud(partial, moge_aligned, omni_aligned):
    partial_points = pcd_to_numpy(partial)
    moge_points = pcd_to_numpy(moge_aligned)
    omni_points = pcd_to_numpy(omni_aligned)

    points = np.concatenate([partial_points, moge_points, omni_points], axis=0)
    partial_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(partial_points), 1))
    moge_colors = np.tile(np.array([[0.0, 0.2, 1.0]]), (len(moge_points), 1))
    omni_colors = np.full((len(omni_points), 3), 0.68, dtype=np.float64)
    colors = np.concatenate([partial_colors, moge_colors, omni_colors], axis=0)
    return numpy_to_pcd(points, colors)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_mogev2_points(
    image_path,
    pretrained,
    device,
    fp16,
    white_threshold,
    max_points,
    seed,
    crop_foreground=True,
    crop_padding=8,
    alpha_threshold=8,
):
    try:
        import cv2
        import torch
        from moge.model.v2 import MoGeModel
    except Exception as exc:
        raise RuntimeError(
            "MoGeV2 is not available. Install it in the genpc environment with: "
            "/opt/data/private/cr/miniconda3/envs/genpc/bin/python -m pip install "
            "git+https://github.com/microsoft/MoGe.git"
        ) from exc

    pretrained_path = Path(pretrained)
    if pretrained_path.exists() and pretrained_path.is_dir():
        pretrained = str(pretrained_path / "model.pt")

    image_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if image_raw.ndim == 2:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
    elif image_raw.shape[-1] == 4:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGRA2RGBA)
    else:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    original_shape = list(image_raw.shape[:2])
    source_foreground = image_foreground_mask(
        image_raw,
        white_threshold=white_threshold,
        alpha_threshold=alpha_threshold,
    )
    crop_box = (0, 0, image_raw.shape[1], image_raw.shape[0])
    if crop_foreground:
        image_raw, source_foreground, crop_box = crop_to_foreground(
            image_raw, source_foreground, padding=crop_padding
        )

    image_rgb = image_to_rgb_for_moge(image_raw)
    image_tensor = torch.tensor(
        image_rgb / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    model = MoGeModel.from_pretrained(pretrained).to(device).eval()
    with torch.no_grad():
        if fp16 and str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.infer(image_tensor)
        else:
            output = model.infer(image_tensor)

    points = output["points"].detach().float().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()
    valid = build_foreground_mask(
        points,
        mask,
        image_rgb,
        white_threshold,
        image_mask=source_foreground,
    )
    flat_points = points[valid].astype(np.float64)
    flat_colors = (image_rgb[valid].astype(np.float64) / 255.0).clip(0, 1)
    flat_points, flat_colors = maybe_subsample(flat_points, flat_colors, max_points, seed)
    if len(flat_points) == 0:
        raise ValueError("MoGeV2 produced no valid foreground points after filtering.")
    return numpy_to_pcd(flat_points, flat_colors), {
        "pretrained": pretrained,
        "image_path": str(image_path),
        "valid_points": int(len(flat_points)),
        "white_threshold": int(white_threshold),
        "alpha_threshold": int(alpha_threshold),
        "crop_foreground": bool(crop_foreground),
        "crop_padding": int(crop_padding),
        "crop_box_xyxy": list(crop_box),
        "original_hw": original_shape,
        "inference_hw": list(image_rgb.shape[:2]),
        "max_points": None if max_points is None else int(max_points),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--sample_id", default="car__132")
    parser.add_argument("--image_name", default="img_sam.png")
    parser.add_argument(
        "--moge_pretrained",
        default=str(DEFAULT_MOGEV2_MODEL),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--white_threshold", type=int, default=248)
    parser.add_argument("--alpha_threshold", type=int, default=8)
    parser.add_argument("--crop_foreground", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop_padding", type=int, default=8)
    parser.add_argument("--moge_max_points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--icp_max_correspondence_distance", type=float, default=0.05)
    parser.add_argument("--icp_voxel_size", type=float, default=0.01)
    parser.add_argument(
        "--geotransformer_weights",
        default="third_party/GeoTransformer/weights/geotransformer-modelnet.pth.tar",
    )
    parser.add_argument("--geotransformer_num_points", type=int, default=717)
    return parser.parse_args()


def main():
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    sample_id = args.sample_id
    image_path = sample_dir / args.image_name
    partial_path = sample_dir / f"{sample_id}_hunyuan3d_omni_point_control_processed.ply"
    omni_path = sample_dir / f"{sample_id}_hunyuan3d_omni.ply"

    for path in (image_path, partial_path, omni_path):
        if not path.exists():
            raise FileNotFoundError(path)

    moge_path = sample_dir / f"{sample_id}_mogev2_points.ply"
    moge_info_path = sample_dir / f"{sample_id}_mogev2_points_info.json"
    icp_path = sample_dir / f"{sample_id}_mogev2_icp_to_processed_partial.ply"
    icp_transform_path = sample_dir / f"{sample_id}_mogev2_icp_to_processed_partial_transform.npy"
    icp_info_path = sample_dir / f"{sample_id}_mogev2_icp_to_processed_partial_info.json"
    omni_registered_path = sample_dir / f"{sample_id}_omni_geotransformer_to_mogev2_icp.ply"
    geo_transform_path = sample_dir / f"{sample_id}_omni_geotransformer_to_mogev2_icp_transform.npy"
    geo_info_path = sample_dir / f"{sample_id}_omni_geotransformer_to_mogev2_icp_info.json"
    vis_path = sample_dir / f"{sample_id}_mogev2_icp_geotransformer_vis.ply"

    partial = o3d.io.read_point_cloud(str(partial_path))
    omni = o3d.io.read_point_cloud(str(omni_path))
    moge, moge_info = load_mogev2_points(
        image_path=image_path,
        pretrained=args.moge_pretrained,
        device=args.device,
        fp16=args.fp16,
        white_threshold=args.white_threshold,
        max_points=args.moge_max_points,
        seed=args.seed,
        crop_foreground=args.crop_foreground,
        crop_padding=args.crop_padding,
        alpha_threshold=args.alpha_threshold,
    )
    o3d.io.write_point_cloud(str(moge_path), moge)
    save_json(moge_info_path, moge_info)

    moge_icp, icp_transform, icp_info = run_similarity_icp(
        moge,
        partial,
        max_correspondence_distance=args.icp_max_correspondence_distance,
        voxel_size=args.icp_voxel_size,
    )
    o3d.io.write_point_cloud(str(icp_path), moge_icp)
    np.save(icp_transform_path, icp_transform)
    icp_info["partial_to_moge_icp_nn"] = nearest_neighbor_stats(partial, moge_icp)
    icp_info["moge_icp_to_partial_nn"] = nearest_neighbor_stats(moge_icp, partial)
    save_json(icp_info_path, {**icp_info, "transform": icp_transform.tolist()})

    geo_transform, geo_info = estimate_geotransformer_transform(
        moge_icp,
        omni,
        weights_path=args.geotransformer_weights,
        device=args.device,
        num_points=args.geotransformer_num_points,
        seed=args.seed,
    )
    omni_registered = copy_pcd(omni)
    omni_registered.transform(geo_transform)
    o3d.io.write_point_cloud(str(omni_registered_path), omni_registered)
    np.save(geo_transform_path, geo_transform)
    geo_info["moge_icp_to_omni_registered_nn"] = nearest_neighbor_stats(
        moge_icp, omni_registered
    )
    geo_info["omni_registered_to_moge_icp_nn"] = nearest_neighbor_stats(
        omni_registered, moge_icp
    )
    save_json(geo_info_path, {**geo_info, "transform": geo_transform.tolist()})

    vis = make_visualization_cloud(partial, moge_icp, omni_registered)
    o3d.io.write_point_cloud(str(vis_path), vis)

    print(f"MoGeV2 points: {len(moge.points)} -> {moge_path}")
    print(
        "ICP "
        f"initial_scale:{icp_info['initial_scale']:.6f}, "
        f"fitness:{icp_info['fitness']:.6f}, "
        f"inlier_rmse:{icp_info['inlier_rmse']:.6f}"
    )
    print(
        "GeoTransformer "
        f"corr:{geo_info['num_correspondences']}, "
        f"scale:{geo_info['uniform_scale']:.6f}"
    )
    print(f"ICP aligned MoGeV2: {icp_path}")
    print(f"GeoTransformer registered omni: {omni_registered_path}")
    print(f"Visualization: {vis_path}")
    print("partial -> moge_icp nn:", icp_info["partial_to_moge_icp_nn"])
    print("moge_icp -> omni_registered nn:", geo_info["moge_icp_to_omni_registered_nn"])


if __name__ == "__main__":
    main()
