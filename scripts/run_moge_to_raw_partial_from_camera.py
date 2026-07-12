import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_moge_pixel_index_bridge import (
    build_partial_to_moge_index,
    colors_for_moge_hits,
    filter_moge_points_by_object_mask,
    prepare_object_mask,
    run_moge_with_pixels,
    run_rmbg_mask,
    save_mask_png,
    write_pcd,
)
from scripts.run_moge_to_partial_from_index import (
    apply_transform,
    make_compare_cloud,
    pcd_points,
    ransac_similarity,
    save_json,
)


DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "workspace" / "scansalon_zup_side_512" / "car__132"
DEFAULT_PARTIAL = PROJECT_ROOT / "workspace" / "scansalon" / "_inputs_denoised" / "car" / "car__132.ply"
DEFAULT_MOGE_MODEL = PROJECT_ROOT / "models" / "moge-2-vitl"
DEFAULT_RMBG_MODEL = PROJECT_ROOT / "models" / "RMBG-2.0"


def camera_xy_to_uv(camera_xy, padding):
    camera_xy = np.asarray(camera_xy, dtype=np.float64)
    if camera_xy.ndim != 2 or camera_xy.shape[1] != 2:
        raise ValueError(f"camera_xy must have shape [N, 2], got {camera_xy.shape}")
    xy_min = camera_xy.min(axis=0, keepdims=True)
    xy_max = camera_xy.max(axis=0, keepdims=True)
    center = (xy_min + xy_max) * 0.5
    scale = max(float((xy_max - xy_min).max()), 1e-8)
    return ((camera_xy - center) / scale) * (1.0 - 2.0 * float(padding)) + 0.5


def depth_image_uv_from_projection_uv(projection_uv):
    image_uv = np.asarray(projection_uv, dtype=np.float64).copy()
    if image_uv.ndim != 2 or image_uv.shape[1] != 2:
        raise ValueError(f"projection_uv must have shape [N, 2], got {image_uv.shape}")
    # DepthPrompting.paintPixels() flips the tensor vertically before saving
    # depth.png. Qwen/MoGe operate on that saved image, so pixel matching must
    # use the post-flip image-space v coordinate.
    image_uv[:, 1] = 1.0 - image_uv[:, 1]
    return image_uv


def project_partial_with_saved_camera(partial_points, camera_path, padding, device):
    import torch

    camera = torch.load(str(camera_path), map_location=device, weights_only=False)
    points = torch.as_tensor(partial_points, dtype=torch.float32, device=device)
    with torch.no_grad():
        transformed = camera.transform(points).detach().float().cpu().numpy()
    projection_uv = camera_xy_to_uv(transformed[:, :2], padding=padding)
    image_uv = depth_image_uv_from_projection_uv(projection_uv)
    return image_uv, transformed


def write_compare(path, partial_pcd, aligned_points, colors=None):
    import open3d as o3d

    tmp_path = Path(path).with_suffix(".aligned_tmp.ply")
    write_pcd(tmp_path, aligned_points, colors)
    aligned = o3d.io.read_point_cloud(str(tmp_path))
    o3d.io.write_point_cloud(str(path), make_compare_cloud(partial_pcd, aligned))
    tmp_path.unlink(missing_ok=True)


def run(args):
    sample_dir = Path(args.sample_dir)
    image_path = sample_dir / args.image_name
    camera_path = sample_dir / args.camera_name
    partial_path = Path(args.partial_path)
    prefix = sample_dir / args.output_prefix

    partial_pcd, partial_points = pcd_points(partial_path)
    point_uv, camera_points = project_partial_with_saved_camera(
        partial_points=partial_points,
        camera_path=camera_path,
        padding=args.padding,
        device=args.device,
    )
    np.save(prefix.with_name(prefix.name + "_raw_partial_point_uv.npy"), point_uv)
    np.save(prefix.with_name(prefix.name + "_raw_partial_camera_points.npy"), camera_points)

    full_moge_points, full_moge_colors, full_moge_pixels, moge_info = run_moge_with_pixels(
        image_path=image_path,
        pretrained=args.moge_model,
        device=args.device,
        fp16=bool(args.fp16),
    )
    image_size = int(args.image_size)
    if moge_info["image_hw"] != [image_size, image_size]:
        raise ValueError(f"Expected {image_size}x{image_size}, got {moge_info['image_hw']}")

    rmbg_path = prefix.with_name(prefix.name + "_rmbg.png")
    object_mask_path = prefix.with_name(prefix.name + "_object_mask.png")
    object_mask = run_rmbg_mask(image_path, rmbg_path, args.rmbg_model)
    object_mask = prepare_object_mask(
        object_mask,
        alpha_threshold=args.object_alpha_threshold,
        erode_pixels=args.object_mask_erode_pixels,
    )
    save_mask_png(object_mask_path, object_mask)
    object_moge = filter_moge_points_by_object_mask(
        points=full_moge_points,
        colors=full_moge_colors,
        pixel_xy=full_moge_pixels,
        object_mask=object_mask,
        alpha_threshold=args.object_alpha_threshold,
        erode_pixels=0,
    )

    index_result = build_partial_to_moge_index(
        point_uv=point_uv,
        image_size=image_size,
        moge_pixel_xy=object_moge.pixel_xy,
        max_pixel_distance=args.max_pixel_distance,
    )
    valid = index_result.partial_to_moge >= 0
    source = object_moge.points[index_result.partial_to_moge[valid]]
    target = partial_points[valid]
    if len(source) > int(args.max_correspondences):
        rng = np.random.default_rng(int(args.seed))
        ids = rng.choice(len(source), size=int(args.max_correspondences), replace=False)
        source = source[ids]
        target = target[ids]

    ransac = ransac_similarity(
        source,
        target,
        iterations=args.ransac_iterations,
        threshold=args.ransac_threshold,
        seed=args.seed,
    )
    transform = ransac["transform"]
    aligned_moge = apply_transform(object_moge.points, transform)

    object_ply = prefix.with_name(prefix.name + "_moge_object_only.ply")
    hit_ply = prefix.with_name(prefix.name + "_moge_object_partial_hits_red.ply")
    aligned_ply = prefix.with_name(prefix.name + "_moge_aligned_to_raw_partial.ply")
    compare_ply = prefix.with_name(prefix.name + "_raw_partial_gray_moge_red_aligned.ply")
    transform_path = prefix.with_name(prefix.name + "_moge_to_raw_partial_transform.npy")
    index_path = prefix.with_name(prefix.name + "_partial_to_moge_index.npy")
    info_path = prefix.with_name(prefix.name + "_info.json")

    write_pcd(object_ply, object_moge.points, object_moge.colors)
    write_pcd(
        hit_ply,
        object_moge.points,
        colors_for_moge_hits(len(object_moge.points), index_result.matched_moge_indices),
    )
    write_pcd(aligned_ply, aligned_moge, object_moge.colors)
    write_compare(compare_ply, partial_pcd, aligned_moge, object_moge.colors)
    np.save(transform_path, transform)
    np.save(index_path, index_result.partial_to_moge)

    info = {
        "partial": str(partial_path),
        "partial_points": int(len(partial_points)),
        "camera": str(camera_path),
        "image": str(image_path),
        "image_size": image_size,
        "full_moge_points": int(len(full_moge_points)),
        "moge_object_points": int(len(object_moge.points)),
        "object_alpha_threshold": int(args.object_alpha_threshold),
        "object_mask_erode_pixels": int(args.object_mask_erode_pixels),
        "matched_partial_points": int(valid.sum()),
        "match_ratio": float(valid.sum() / max(len(partial_points), 1)),
        "used_correspondences": int(len(source)),
        "ransac_inliers": int(ransac["score"]),
        "ransac_inlier_ratio": float(ransac["score"] / max(len(source), 1)),
        "ransac_median_error": float(ransac["error"]),
        "ransac_mean_error": float(ransac["mean_error"]),
        "ransac_p95_error": float(ransac["p95_error"]),
        "projection": {
            "method": "saved_camera_transform_then_depthprompting_rescale_xy",
            "padding": float(args.padding),
            "axis_note": "partial xyz is projected by camera.pth; no depth-view PLY axes are used as target",
        },
        "moge": moge_info,
        "transform": transform.tolist(),
        "outputs": {
            "moge_object": str(object_ply),
            "moge_hits": str(hit_ply),
            "aligned_moge": str(aligned_ply),
            "compare": str(compare_ply),
            "transform": str(transform_path),
            "partial_to_moge_index": str(index_path),
            "object_mask": str(object_mask_path),
            "rmbg": str(rmbg_path),
        },
    }
    save_json(info_path, info)
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--partial_path", default=str(DEFAULT_PARTIAL))
    parser.add_argument("--image_name", default="qwen_edit_2511_car_completion_from_depth.png")
    parser.add_argument("--camera_name", default="camera.pth")
    parser.add_argument("--moge_model", default=str(DEFAULT_MOGE_MODEL))
    parser.add_argument("--rmbg_model", default=str(DEFAULT_RMBG_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--max_pixel_distance", type=float, default=2.0)
    parser.add_argument("--object_alpha_threshold", type=int, default=128)
    parser.add_argument("--object_mask_erode_pixels", type=int, default=2)
    parser.add_argument("--ransac_iterations", type=int, default=5000)
    parser.add_argument("--ransac_threshold", type=float, default=0.08)
    parser.add_argument("--max_correspondences", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--output_prefix", default="car__132_moge_to_raw_partial")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
