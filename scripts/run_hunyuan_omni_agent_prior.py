#!/usr/bin/env python3
"""Generate an agent prior with Hunyuan3D-Omni image + partial-point control.

Omni's native point-control frame is explicitly recorded.  The following
generic proper-Sim(3) registration is responsible for mapping it to the
partial frame; this runner never assumes a Pixal or Hunyuan canonical axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import torch
import trimesh
import cv2
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "third_party" / "Hunyuan3D-Omni"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Hunyuan3D-Omni"


def load_points(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"empty partial point cloud: {path}")
    return points


def normalize_point_control(points: np.ndarray, *, max_points: int, scale: float) -> tuple[np.ndarray, dict]:
    if len(points) > max_points:
        ids = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[ids]
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = .5 * (lo + hi)
    extent = max(float((hi - lo).max()), 1e-8)
    normalized = (points - center) / extent * (2. * float(scale))
    return normalized.astype(np.float32), {
        "native_frame": "partial_bbox_normalized",
        "input_center": center.tolist(), "input_max_extent": extent,
        "target_half_extent": float(scale), "points": int(len(normalized)),
    }


def semantic_rgba_condition(image_path: Path, destination: Path) -> dict:
    """Give Omni an explicit object alpha rather than treating white canvas as shape.

    Hunyuan3D-Omni's preprocessor assigns full alpha to RGB files.  Our semantic
    images intentionally use a white background, so derive a conservative
    foreground alpha from luminance/chroma and let Omni apply its documented
    object recentering to the true silhouette.
    """
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    foreground = (gray < 245) | (chroma > 12)
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    foreground = cv2.dilate(foreground, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if int(foreground.sum()) < 128:
        raise ValueError(f"could not extract semantic foreground from {image_path}")
    rgba = np.dstack((rgb, foreground.astype(np.uint8) * 255))
    Image.fromarray(rgba, mode="RGBA").save(destination)
    ys, xs = np.nonzero(foreground)
    return {"alpha_source": "semantic_luminance_chroma", "foreground_pixels": int(foreground.sum()),
            "foreground_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--octree-resolution", type=int, default=512)
    parser.add_argument("--max-control-points", type=int, default=81920)
    parser.add_argument("--control-scale", type=float, default=.98)
    parser.add_argument("--sample-count", type=int, default=100000)
    parser.add_argument("--fast-decode", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=16,
                        help="Parallel CPU threads for the octree/mesh decode stage.")
    args = parser.parse_args()
    if not args.image.exists() or not args.partial.exists() or not args.model.exists():
        raise FileNotFoundError("image, partial and local Omni model must exist")
    if not (args.source_root / "hy3dshape").exists():
        raise FileNotFoundError(f"Hunyuan3D-Omni source missing: {args.source_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    if str(args.source_root) not in sys.path:
        sys.path.insert(0, str(args.source_root))
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

    partial = load_points(args.partial)
    control, control_info = normalize_point_control(
        partial, max_points=args.max_control_points, scale=args.control_scale
    )
    control_path = args.output_dir / "partial_point_control_native.ply"
    native_cloud = o3d.geometry.PointCloud()
    native_cloud.points = o3d.utility.Vector3dVector(control.astype(np.float64))
    o3d.io.write_point_cloud(str(control_path), native_cloud, write_ascii=False)
    condition_image = args.output_dir / "semantic_condition_rgba.png"
    alpha_info = semantic_rgba_condition(args.image, condition_image)

    started = time.time()
    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        str(args.model), variant="ema", device="cuda", dtype=torch.float16,
        fast_decode=bool(args.fast_decode),
    )
    generator = torch.Generator("cuda").manual_seed(args.seed)
    result = pipeline(
        image=str(condition_image),
        point=torch.from_numpy(control).unsqueeze(0).to(device="cuda", dtype=torch.float16),
        num_inference_steps=args.steps, octree_resolution=args.octree_resolution,
        mc_level=0., guidance_scale=args.guidance, generator=generator,
        fast_decode=bool(args.fast_decode),
    )
    mesh = DegenerateFaceRemover()(FloaterRemover()(result["shapes"][0][0]))
    mesh_path = args.output_dir / "hunyuan_omni_native.glb"
    mesh.export(mesh_path)
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(
        loaded, count=args.sample_count, seed=np.random.default_rng(args.seed)
    )
    points_path = args.output_dir / "hunyuan_omni_native_sampled_100k.ply"
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(points_path), cloud, write_ascii=False)
    metadata = {
        "backend": "Hunyuan3D-Omni", "strict_zero_shot": True,
        "conditioning": {"image": str(args.image.resolve()), "rgba_condition": str(condition_image.resolve()),
                           "partial": str(args.partial.resolve()), "point_control": str(control_path.resolve()),
                           **control_info, **alpha_info},
        "model": str(args.model.resolve()), "source": str(args.source_root.resolve()),
        "seed": args.seed, "steps": args.steps, "guidance": args.guidance,
        "octree_resolution": args.octree_resolution, "fast_decode": bool(args.fast_decode),
        "native_to_method": "unknown_proper_sim3_solved_by_common_global_registration",
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "hunyuan_omni_agent_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"mesh": str(mesh_path), "points": str(points_path),
                      "seconds": round(metadata["elapsed_seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
