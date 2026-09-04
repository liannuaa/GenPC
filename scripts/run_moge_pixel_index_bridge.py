import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "workspace" / "scansalon_zup_side_512" / "car__132"
DEFAULT_MOGE_MODEL = PROJECT_ROOT / "models" / "moge-2-vitl"


@dataclass
class PixelIndexResult:
    partial_to_moge: np.ndarray
    pixel_distances: np.ndarray
    matched_partial_indices: np.ndarray
    matched_moge_indices: np.ndarray


@dataclass
class MogeObjectPoints:
    points: np.ndarray
    colors: np.ndarray
    pixel_xy: np.ndarray
    original_indices: np.ndarray


def point_uv_to_pixel_xy(point_uv, image_size, *, flip_y=False):
    uv = np.asarray(point_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"point_uv must have shape [N, 2], got {uv.shape}")
    uv = uv.copy()
    if bool(flip_y):
        # DepthPrompting canonical UV uses a bottom-left origin, while the
        # semantic PNG and MoGe arrays use top-left image coordinates.
        uv[:, 1] = 1.0 - uv[:, 1]
    pixel_xy = np.rint(uv * (int(image_size) - 1)).astype(np.int64)
    valid = np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0)
    valid &= (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
    return pixel_xy, valid


def build_partial_to_moge_index(
    point_uv,
    image_size,
    moge_pixel_xy,
    max_pixel_distance,
    *,
    flip_y=False,
):
    pixel_xy, valid_partial = point_uv_to_pixel_xy(point_uv, image_size, flip_y=flip_y)
    moge_pixel_xy = np.asarray(moge_pixel_xy, dtype=np.float64)
    if moge_pixel_xy.ndim != 2 or moge_pixel_xy.shape[1] != 2:
        raise ValueError(f"moge_pixel_xy must have shape [M, 2], got {moge_pixel_xy.shape}")

    partial_to_moge = np.full(len(pixel_xy), -1, dtype=np.int64)
    pixel_distances = np.full(len(pixel_xy), np.inf, dtype=np.float64)
    if len(moge_pixel_xy) == 0 or not valid_partial.any():
        return PixelIndexResult(
            partial_to_moge=partial_to_moge,
            pixel_distances=pixel_distances,
            matched_partial_indices=np.empty(0, dtype=np.int64),
            matched_moge_indices=np.empty(0, dtype=np.int64),
        )

    valid_indices = np.where(valid_partial)[0]
    distances, nearest = cKDTree(moge_pixel_xy).query(pixel_xy[valid_indices], k=1)
    keep = distances <= float(max_pixel_distance)
    matched_partial = valid_indices[keep].astype(np.int64)
    matched_moge = nearest[keep].astype(np.int64)
    partial_to_moge[matched_partial] = matched_moge
    pixel_distances[matched_partial] = distances[keep]
    return PixelIndexResult(
        partial_to_moge=partial_to_moge,
        pixel_distances=pixel_distances,
        matched_partial_indices=matched_partial,
        matched_moge_indices=matched_moge,
    )


def colors_for_moge_hits(num_points, hit_indices):
    colors = np.full((int(num_points), 3), 0.55, dtype=np.float64)
    hit_indices = np.asarray(hit_indices, dtype=np.int64)
    hit_indices = hit_indices[(hit_indices >= 0) & (hit_indices < int(num_points))]
    if len(hit_indices):
        colors[np.unique(hit_indices)] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return colors


def prepare_object_mask(object_mask, alpha_threshold, erode_pixels=0):
    object_mask = np.asarray(object_mask)
    if object_mask.ndim == 3:
        object_mask = object_mask[..., -1]
    if object_mask.ndim != 2:
        raise ValueError(f"object_mask must be 2D or RGBA-like, got {object_mask.shape}")

    binary = (object_mask >= int(alpha_threshold)).astype(np.uint8) * 255
    erode_pixels = int(erode_pixels)
    if erode_pixels > 0:
        import cv2

        kernel_size = erode_pixels * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
    return binary


def filter_moge_points_by_object_mask(
    points,
    colors,
    pixel_xy,
    object_mask,
    alpha_threshold,
    erode_pixels=0,
):
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    object_mask = prepare_object_mask(object_mask, alpha_threshold, erode_pixels)
    if len(points) != len(colors) or len(points) != len(pixel_xy):
        raise ValueError(
            "points, colors, and pixel_xy must have the same first dimension: "
            f"{len(points)}, {len(colors)}, {len(pixel_xy)}"
        )

    xy = np.rint(pixel_xy).astype(np.int64)
    height, width = object_mask.shape
    in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    keep = np.zeros(len(xy), dtype=bool)
    valid_indices = np.where(in_bounds)[0]
    keep[valid_indices] = object_mask[xy[valid_indices, 1], xy[valid_indices, 0]] > 0
    original_indices = np.where(keep)[0].astype(np.int64)
    return MogeObjectPoints(
        points=points[keep],
        colors=colors[keep],
        pixel_xy=pixel_xy[keep],
        original_indices=original_indices,
    )


def numpy_to_pcd(points, colors=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def write_pcd(path, points, colors=None):
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), numpy_to_pcd(points, colors))


def load_image_rgb(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)[..., :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_alpha_mask(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return image
    if image.shape[-1] == 4:
        return image[..., 3]
    raise ValueError(f"Expected RMBG output with alpha channel, got shape {image.shape}")


def save_mask_png(path, mask):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8))


def run_rmbg_mask(image_path, output_rgba_path, model_path):
    from tools.RMBG import RMBG_pred

    RMBG_pred(str(image_path), str(output_rgba_path), model_path=str(model_path))
    return load_alpha_mask(output_rgba_path)


def _serialize_output_value(value):
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if np.isscalar(value):
        return value.item()
    return str(type(value).__name__)


def run_moge_with_pixels(image_path, pretrained, device, fp16):
    import torch
    from moge.model.v2 import MoGeModel

    pretrained = Path(pretrained)
    if pretrained.is_dir():
        pretrained = pretrained / "model.pt"

    image_rgb = load_image_rgb(image_path)
    image_tensor = torch.tensor(
        image_rgb / 255.0,
        dtype=torch.float32,
        device=device,
    ).permute(2, 0, 1)

    model = MoGeModel.from_pretrained(str(pretrained)).to(device).eval()
    with torch.no_grad():
        if fp16 and str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.infer(image_tensor)
        else:
            output = model.infer(image_tensor)

    points = output["points"].detach().float().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy().astype(bool)
    valid = mask.copy()
    valid &= np.isfinite(points).all(axis=-1)
    valid &= np.linalg.norm(points, axis=-1) > 1e-8

    ys, xs = np.where(valid)
    pixel_xy = np.stack([xs, ys], axis=1).astype(np.float64)
    flat_points = points[valid].astype(np.float64)
    flat_colors = (image_rgb[valid].astype(np.float64) / 255.0).clip(0, 1)
    info = {
        "pretrained": str(pretrained),
        "image_path": str(image_path),
        "image_hw": list(image_rgb.shape[:2]),
        "valid_points": int(len(flat_points)),
        "output_keys": {
            key: _serialize_output_value(value)
            for key, value in output.items()
            if key not in {"points", "mask"}
        },
        "camera2_frame": "moge_camera_coordinates_identity",
    }
    return flat_points, flat_colors, pixel_xy, info


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def run(args):
    sample_dir = Path(args.sample_dir)
    image_path = sample_dir / args.image_name
    point_uv_path = sample_dir / args.point_uv_name
    prefix = sample_dir / args.output_prefix

    point_uv = np.load(point_uv_path)
    image_size = int(args.image_size)
    full_moge_points, full_moge_colors, full_moge_pixel_xy, moge_info = run_moge_with_pixels(
        image_path=image_path,
        pretrained=args.moge_model,
        device=args.device,
        fp16=bool(args.fp16),
    )
    if moge_info["image_hw"] != [image_size, image_size]:
        raise ValueError(
            f"Expected {image_size}x{image_size} semantic image, got {moge_info['image_hw']}"
        )

    rmbg_rgba_path = prefix.with_name(prefix.name + "_rmbg.png")
    object_mask_path = prefix.with_name(prefix.name + "_object_mask.png")
    if bool(args.use_rmbg_mask):
        object_mask = run_rmbg_mask(
            image_path=image_path,
            output_rgba_path=rmbg_rgba_path,
            model_path=args.rmbg_model,
        )
    else:
        object_mask = np.full((image_size, image_size), 255, dtype=np.uint8)
    object_mask = prepare_object_mask(
        object_mask,
        alpha_threshold=args.object_alpha_threshold,
        erode_pixels=args.object_mask_erode_pixels,
    )
    save_mask_png(object_mask_path, object_mask)
    object_moge = filter_moge_points_by_object_mask(
        points=full_moge_points,
        colors=full_moge_colors,
        pixel_xy=full_moge_pixel_xy,
        object_mask=object_mask,
        alpha_threshold=args.object_alpha_threshold,
        erode_pixels=0,
    )

    index_result = build_partial_to_moge_index(
        point_uv=point_uv,
        image_size=image_size,
        moge_pixel_xy=object_moge.pixel_xy,
        max_pixel_distance=args.max_pixel_distance,
        flip_y=bool(args.flip_point_uv_y),
    )
    partial_to_full_moge = np.full(len(index_result.partial_to_moge), -1, dtype=np.int64)
    matched_mask = index_result.partial_to_moge >= 0
    partial_to_full_moge[matched_mask] = object_moge.original_indices[
        index_result.partial_to_moge[matched_mask]
    ]
    matches = np.column_stack(
        [
            index_result.matched_partial_indices,
            index_result.matched_moge_indices,
            object_moge.original_indices[index_result.matched_moge_indices],
            index_result.pixel_distances[index_result.matched_partial_indices],
        ]
    )

    moge_points_path = prefix.with_name(prefix.name + "_moge_points.ply")
    red_hits_path = prefix.with_name(prefix.name + "_moge_partial_hits_red.ply")
    index_path = prefix.with_name(prefix.name + "_partial_to_moge_index.npy")
    full_index_path = prefix.with_name(prefix.name + "_partial_to_full_moge_index.npy")
    distance_path = prefix.with_name(prefix.name + "_partial_to_moge_pixel_distances.npy")
    matches_path = prefix.with_name(prefix.name + "_partial_to_moge_matches.npy")
    pixel_path = prefix.with_name(prefix.name + "_moge_pixels.npy")
    full_pixel_path = prefix.with_name(prefix.name + "_full_moge_pixels.npy")
    object_original_index_path = prefix.with_name(prefix.name + "_moge_object_original_indices.npy")
    info_path = prefix.with_name(prefix.name + "_info.json")

    write_pcd(moge_points_path, object_moge.points, object_moge.colors)
    hit_colors = colors_for_moge_hits(len(object_moge.points), index_result.matched_moge_indices)
    write_pcd(red_hits_path, object_moge.points, hit_colors)
    np.save(index_path, index_result.partial_to_moge)
    np.save(full_index_path, partial_to_full_moge)
    np.save(distance_path, index_result.pixel_distances)
    np.save(matches_path, matches)
    np.save(pixel_path, object_moge.pixel_xy)
    np.save(full_pixel_path, full_moge_pixel_xy)
    np.save(object_original_index_path, object_moge.original_indices)

    info = {
        "sample_dir": str(sample_dir),
        "image": str(image_path),
        "point_uv": str(point_uv_path),
        "image_size": image_size,
        "partial_points": int(len(point_uv)),
        "full_moge_points": int(len(full_moge_points)),
        "moge_points": int(len(object_moge.points)),
        "removed_background_points": int(len(full_moge_points) - len(object_moge.points)),
        "matched_partial_points": int(len(index_result.matched_partial_indices)),
        "match_ratio": float(len(index_result.matched_partial_indices) / max(len(point_uv), 1)),
        "max_pixel_distance": float(args.max_pixel_distance),
        "partial_uv_bottom_left_to_image_top_left": bool(args.flip_point_uv_y),
        "use_rmbg_mask": bool(args.use_rmbg_mask),
        "object_alpha_threshold": int(args.object_alpha_threshold),
        "object_mask_erode_pixels": int(args.object_mask_erode_pixels),
        "moge": moge_info,
        "outputs": {
            "moge_points": str(moge_points_path),
            "red_hits": str(red_hits_path),
            "partial_to_moge_index": str(index_path),
            "partial_to_full_moge_index": str(full_index_path),
            "partial_to_moge_pixel_distances": str(distance_path),
            "partial_to_moge_matches": str(matches_path),
            "moge_pixels": str(pixel_path),
            "full_moge_pixels": str(full_pixel_path),
            "moge_object_original_indices": str(object_original_index_path),
            "rmbg_rgba": str(rmbg_rgba_path) if bool(args.use_rmbg_mask) else None,
            "object_mask": str(object_mask_path),
        },
    }
    save_json(info_path, info)
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--image_name", default="qwen_edit_2511_car_completion_from_depth.png")
    parser.add_argument("--point_uv_name", default="point_uv.npy")
    parser.add_argument("--moge_model", default=str(DEFAULT_MOGE_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_pixel_distance", type=float, default=2.0)
    parser.add_argument("--flip_point_uv_y", action=argparse.BooleanOptionalAction, default=False,
                        help="Convert bottom-left canonical partial UV to top-left semantic/MoGe pixels.")
    parser.add_argument("--rmbg_model", default=str(PROJECT_ROOT / "models" / "RMBG-2.0"))
    parser.add_argument("--object_alpha_threshold", type=int, default=128)
    parser.add_argument("--object_mask_erode_pixels", type=int, default=2)
    parser.add_argument("--use_rmbg_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_prefix", default="car__132_moge_pixel_bridge")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
