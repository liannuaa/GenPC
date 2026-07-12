import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEREG_ROOT = PROJECT_ROOT / "third_party" / "FreeReg"
LEGACY_FREEREG_ROOT = PROJECT_ROOT.parent / "FreeReg"
DEFAULT_FALLBACK_IR_3D = (0.10, 0.20)
DEFAULT_MAX_COMPLETE_TO_IMAGE_TRANSLATION = 50.0


def add_freereg_to_path(freereg_root):
    freereg_root = Path(freereg_root).resolve()
    if not freereg_root.exists() and LEGACY_FREEREG_ROOT.exists():
        freereg_root = LEGACY_FREEREG_ROOT.resolve()

    depthpro_src = freereg_root / "tools" / "DepthPro" / "src"
    for path in (str(depthpro_src), str(freereg_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return freereg_root


def load_object_mask(mask_path, image_shape, threshold, erode_pixels):
    mask = Image.open(mask_path).convert("L")
    width, height = image_shape[1], image_shape[0]
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    mask_np = np.asarray(mask) >= int(threshold)
    if erode_pixels > 0:
        kernel_size = int(erode_pixels) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_np = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask_np


def image_to_masked_depthpro_points(pipe, image, object_mask, max_points, seed):
    from Utils.utils import edge_filter

    height, width = image.shape[:2]
    pipe.H = height
    pipe.W = width
    depth, _, _, intrinsic = pipe.depthpro(image)
    sky = depth > 199.0
    xyz = pipe.projector.proj_depth(depth, intrinsic, depth_unit=1.0)
    edge = edge_filter(depth, sky, times=0.05).reshape(-1)
    valid = (~edge) & (~sky.reshape(-1)) & object_mask.reshape(-1)
    xyz = xyz[valid]
    if len(xyz) == 0:
        raise RuntimeError("Object mask removed all DepthPro points.")
    if len(xyz) > max_points:
        rng = np.random.default_rng(int(seed))
        xyz = xyz[rng.permutation(len(xyz))[: int(max_points)]]
    return xyz, intrinsic, {"height": int(height), "width": int(width), "valid_object_points": int(len(xyz))}


def write_colored_pcd(path, points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.paint_uniform_color(color)
    o3d.io.write_point_cloud(str(path), pcd)
    return pcd


def sim3_matrix(scale, transform):
    matrix = np.asarray(transform, dtype=np.float64).copy()
    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[:3, :3] *= float(scale)
    return matrix @ scale_matrix


def apply_matrix(points, transform):
    points = np.asarray(points, dtype=np.float64)
    hom = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ hom.T).T[:, :3]


def parse_float_list(value):
    if value is None:
        return []
    return [float(item) for item in str(value).split(",") if item.strip()]


def build_ir_3d_candidates(auto_ir_3d, explicit_ir_3d, fallback_ir_3d):
    if explicit_ir_3d is not None:
        return [{"label": "explicit", "ir_3d": float(explicit_ir_3d)}]

    candidates = [{"label": "auto", "ir_3d": float(auto_ir_3d)}]
    for value in fallback_ir_3d:
        value = float(value)
        if not any(np.isclose(value, item["ir_3d"]) for item in candidates):
            candidates.append({"label": f"fallback_{value:g}", "ir_3d": value})
    return candidates


def transform_translation_norm(transform):
    transform = np.asarray(transform, dtype=np.float64)
    return float(np.linalg.norm(transform[:3, 3]))


def estimate_freereg_sim3(
    pipe,
    image_kpt_uvs,
    image_kpts,
    complete_kpts,
    matches,
    ir_3d,
    min_hypotheses,
    max_complete_to_image_translation,
):
    solver = pipe.solver
    solver.ird_3d = float(ir_3d)
    solver.ird_2d = max(10, (pipe.H + pipe.W) / 200.0) if pipe.ir_2d is None else pipe.ir_2d

    matched_image_uvs = image_kpt_uvs[matches[:, 0]]
    matched_image_kpts = image_kpts[matches[:, 0]]
    matched_complete_kpts = complete_kpts[matches[:, 1]]
    scales, hypotheses = solver.gen_hypos(
        matched_image_kpts,
        matched_complete_kpts,
        solver.iters,
        solver.ird_3d,
        np_per_hypo=solver.np_per_hypo,
    )
    hypothesis_count = int(len(scales))
    diagnostics = {
        "ir_3d": float(solver.ird_3d),
        "ir_2d": float(solver.ird_2d),
        "hypotheses": hypothesis_count,
        "valid": False,
    }
    if hypothesis_count < int(min_hypotheses):
        diagnostics["reject_reason"] = "not_enough_hypotheses"
        return diagnostics

    scale, image_to_complete_rigid = solver.ransac(
        matched_image_uvs,
        matched_image_kpts,
        matched_complete_kpts,
        scales,
        hypotheses,
        thres2d=solver.ird_2d,
        thres3d=solver.ird_3d,
    )
    image_to_complete = sim3_matrix(scale, image_to_complete_rigid)
    complete_to_image = np.linalg.inv(image_to_complete)
    complete_to_image_translation = transform_translation_norm(complete_to_image)
    diagnostics.update(
        {
            "valid": bool(np.isfinite(image_to_complete).all()),
            "freereg_scale": float(scale),
            "image_to_complete_translation_norm": transform_translation_norm(image_to_complete),
            "complete_to_image_translation_norm": complete_to_image_translation,
            "image_to_complete_rigid": image_to_complete_rigid,
            "image_to_complete": image_to_complete,
            "complete_to_image": complete_to_image,
        }
    )
    if not diagnostics["valid"]:
        diagnostics["reject_reason"] = "nonfinite_transform"
    elif complete_to_image_translation > float(max_complete_to_image_translation):
        diagnostics["valid"] = False
        diagnostics["reject_reason"] = "complete_to_image_translation_too_large"
    return diagnostics


def json_ready_candidate(candidate):
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"image_to_complete_rigid", "image_to_complete", "complete_to_image"}
    }


def run(args):
    freereg_root = add_freereg_to_path(args.freereg_root)
    from demo import Pipe

    sample_dir = Path(args.sample_dir)
    image_path = sample_dir / args.image_name
    complete_path = sample_dir / args.complete_name
    mask_path = sample_dir / args.object_mask_name
    out_prefix = sample_dir / args.output_prefix

    np.random.seed(int(args.seed))
    pipe = Pipe(args.nkpts, args.vs, args.w_2d, args.ir_2d, args.ir_3d)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    object_mask = load_object_mask(
        mask_path,
        image.shape,
        threshold=args.mask_threshold,
        erode_pixels=args.extra_erode_pixels,
    )
    image_pc, intrinsic, depthpro_info = image_to_masked_depthpro_points(
        pipe,
        image,
        object_mask,
        max_points=args.max_depthpro_points,
        seed=args.seed,
    )

    complete_pcd = o3d.io.read_point_cloud(str(complete_path))
    complete_points = np.asarray(complete_pcd.points, dtype=np.float64)
    if len(complete_points) == 0:
        raise RuntimeError(f"Empty complete point cloud: {complete_path}")

    image_pc, complete_for_reg = pipe._coarse_align(image_pc, complete_points.copy())
    pipe._determine_vs(image_pc, complete_for_reg)
    print(
        f"[FreeReg masked] image_object_points={len(image_pc)} "
        f"complete={len(complete_for_reg)} vs={pipe.vs:.8f}"
    )

    image_kpts, image_feats = pipe._extract_yoho(image_pc, pipe.nkpts)
    image_kpt_uvs, _ = pipe.projector.proj_3to2(image_kpts, intrinsic, np.eye(4))
    complete_kpts, complete_feats = pipe._extract_yoho(complete_for_reg, pipe.nkpts)
    matches = pipe._match(image_feats, complete_feats).astype(np.int16)
    print(f"[FreeReg masked] matches={len(matches)}")
    if len(matches) < 4:
        raise RuntimeError("Not enough FreeReg descriptor matches.")

    pipe.solver.set_intrinsic(intrinsic)
    auto_ir_3d = pipe.vs * 5
    candidates = build_ir_3d_candidates(
        auto_ir_3d=auto_ir_3d,
        explicit_ir_3d=args.ir_3d,
        fallback_ir_3d=parse_float_list(args.fallback_ir_3d),
    )
    candidate_rng_state = np.random.get_state()
    candidate_results = []
    selected = None
    for candidate in candidates:
        np.random.set_state(candidate_rng_state)
        result = estimate_freereg_sim3(
            pipe,
            image_kpt_uvs,
            image_kpts,
            complete_kpts,
            matches,
            ir_3d=candidate["ir_3d"],
            min_hypotheses=args.min_hypotheses,
            max_complete_to_image_translation=args.max_complete_to_image_translation,
        )
        result["label"] = candidate["label"]
        candidate_results.append(result)
        if result["valid"]:
            selected = result
            break
    if selected is None:
        raise RuntimeError(
            "FreeReg failed to produce a valid non-random transform. "
            f"Candidates: {[json_ready_candidate(item) for item in candidate_results]}"
        )

    freereg_scale = selected["freereg_scale"]
    image_to_complete_rigid = selected["image_to_complete_rigid"]
    image_to_complete = selected["image_to_complete"]
    complete_to_image = selected["complete_to_image"]
    registered_complete_points = apply_matrix(complete_for_reg, complete_to_image)

    image_points_path = Path(str(out_prefix) + "_object_depthpro_points.ply")
    registered_path = Path(str(out_prefix) + "_complete_registered_to_object_depthpro.ply")
    fused_path = Path(str(out_prefix) + "_gray_object_depthpro_blue_complete_fused.ply")
    info_path = Path(str(out_prefix) + "_info.json")

    image_pcd = write_colored_pcd(image_points_path, image_pc, [0.55, 0.55, 0.55])
    registered_pcd = write_colored_pcd(registered_path, registered_complete_points, [0.0, 0.25, 1.0])
    o3d.io.write_point_cloud(str(fused_path), image_pcd + registered_pcd)

    info = {
        "method": "original_F-FreeReg_DepthPro_YOHO_Kabsch_object_masked_adaptive_ir3d",
        "freereg_root": str(freereg_root),
        "image": str(image_path),
        "complete_point_cloud": str(complete_path),
        "object_mask": str(mask_path),
        "mask_threshold": int(args.mask_threshold),
        "extra_erode_pixels": int(args.extra_erode_pixels),
        "max_depthpro_points": int(args.max_depthpro_points),
        "nkpts": int(pipe.nkpts),
        "w_2d": float(pipe.w_2d),
        "vs": float(pipe.vs),
        "auto_ir_3d": float(auto_ir_3d),
        "selected_ir_3d": float(selected["ir_3d"]),
        "selected_ir_3d_label": selected["label"],
        "min_hypotheses": int(args.min_hypotheses),
        "max_complete_to_image_translation": float(args.max_complete_to_image_translation),
        "freereg_candidates": [json_ready_candidate(item) for item in candidate_results],
        "depthpro": depthpro_info,
        "complete_points": int(len(complete_for_reg)),
        "image_keypoints": int(len(image_kpts)),
        "complete_keypoints": int(len(complete_kpts)),
        "matches": int(len(matches)),
        "fixed_uv": True,
        "freereg_scale": float(freereg_scale),
        "intrinsic": intrinsic.tolist(),
        "image_to_complete_rigid": image_to_complete_rigid.tolist(),
        "image_to_complete": image_to_complete.tolist(),
        "complete_to_image": complete_to_image.tolist(),
        "outputs": {
            "object_depthpro_points": str(image_points_path),
            "registered_complete": str(registered_path),
            "fused": str(fused_path),
        },
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--image-name", default="img.png")
    parser.add_argument("--complete-name", default=None)
    parser.add_argument("--object-mask-name", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--freereg-root", default=str(DEFAULT_FREEREG_ROOT))
    parser.add_argument("--nkpts", type=int, default=5000)
    parser.add_argument("--vs", type=float, default=None)
    parser.add_argument("--w_2d", type=float, default=0.5)
    parser.add_argument("--ir_2d", type=int, default=None)
    parser.add_argument("--ir_3d", type=float, default=None)
    parser.add_argument(
        "--fallback-ir-3d",
        default=",".join(str(value) for value in DEFAULT_FALLBACK_IR_3D),
        help="Comma-separated fallback 3D inlier thresholds used when auto ir_3d has too few hypotheses.",
    )
    parser.add_argument("--min-hypotheses", type=int, default=2)
    parser.add_argument(
        "--max-complete-to-image-translation",
        type=float,
        default=DEFAULT_MAX_COMPLETE_TO_IMAGE_TRANSLATION,
    )
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--extra-erode-pixels", type=int, default=0)
    parser.add_argument("--max-depthpro-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1184)
    args = parser.parse_args()
    sample_dir = Path(args.sample_dir)
    if args.complete_name is None:
        args.complete_name = f"{sample_dir.name}_hunyuan2.1.ply"
    if args.output_prefix is None:
        args.output_prefix = f"{sample_dir.name}_freereg_original_depthpro_fixeduv_sim3"
    return args


if __name__ == "__main__":
    run(parse_args())
