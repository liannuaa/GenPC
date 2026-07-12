import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

from utils.runtime import model_path, sample_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OMNI_REPO_ROOT = PROJECT_ROOT / "third_party" / "Hunyuan3D-Omni"

_omni_pipeline = None
_omni_pipeline_key = None


def release_hunyuan3d_omni_cache():
    global _omni_pipeline
    global _omni_pipeline_key
    _omni_pipeline = None
    _omni_pipeline_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _ensure_omni_imports():
    if not OMNI_REPO_ROOT.exists():
        raise FileNotFoundError(f"Hunyuan3D-Omni source not found: {OMNI_REPO_ROOT}")
    if str(OMNI_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(OMNI_REPO_ROOT))

    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

    return Hunyuan3DOmniSiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover


def _load_pipeline(cfg):
    global _omni_pipeline
    global _omni_pipeline_key

    pipeline_cls, _, _ = _ensure_omni_imports()
    omni_model_path = model_path(cfg, "hunyuan_omni_model_path", "Hunyuan3D-Omni")
    key = (
        str(omni_model_path),
        str(getattr(cfg, "device", "cuda")),
        getattr(cfg, "hunyuan_omni_variant", None),
        bool(getattr(cfg, "hunyuan_omni_fast_decode", False)),
    )
    if _omni_pipeline is not None and _omni_pipeline_key == key:
        return _omni_pipeline

    print(f"Loading Hunyuan3D-Omni from {omni_model_path}")
    _omni_pipeline = pipeline_cls.from_pretrained(
        str(omni_model_path),
        variant=getattr(cfg, "hunyuan_omni_variant", None),
        device=getattr(cfg, "device", "cuda"),
        dtype=torch.float16,
        fast_decode=bool(getattr(cfg, "hunyuan_omni_fast_decode", False)),
    )
    _omni_pipeline_key = key
    return _omni_pipeline


def _load_point_control(
    path,
    device,
    dtype,
    max_points=None,
    normalize=True,
    scale=0.98,
    normalize_mode="bbox",
):
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if len(colors) != len(points):
        colors = None
    if points.size == 0:
        raise ValueError(f"Empty point control: {path}")
    if max_points and len(points) > int(max_points):
        rng = np.random.default_rng(0)
        indices = rng.choice(len(points), size=int(max_points), replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    if normalize and normalize_mode == "bbox":
        center = (points.max(axis=0) + points.min(axis=0)) * 0.5
        extent = float((points.max(axis=0) - points.min(axis=0)).max())
        if extent > 1e-8:
            points = (points - center) / extent * (2.0 * float(scale))
    elif normalize and normalize_mode == "none":
        points = points * float(scale)
    elif normalize:
        raise ValueError(f"Unknown Hunyuan3D-Omni point normalize mode: {normalize_mode}")
    point_tensor = torch.from_numpy(points).unsqueeze(0).to(device=device, dtype=dtype)
    return point_tensor, points, colors


def _export_processed_point_control(points, colors, ply_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)


def _export_sampled_point_cloud(sampled_point, ply_path):
    points = sampled_point.detach().float().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)


def _export_mesh_points(glb_path, ply_path, num_points):
    mesh = trimesh.load(str(glb_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    points = mesh.sample(int(num_points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)


def hunyuan3d_omni(cfg, flag, img):
    pipeline = _load_pipeline(cfg)
    _, floater_remover_cls, degenerate_face_remover_cls = _ensure_omni_imports()

    model_name = getattr(cfg, "generative_model", "hunyuan3d_omni")
    output_dir = Path(cfg.output_path) / str(flag)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = sample_file(cfg, flag, "img_sam.png")
    point_path = sample_file(cfg, flag, getattr(cfg, "hunyuan_omni_point_file", "color_point.ply"))
    if not image_path.exists():
        raise FileNotFoundError(f"Hunyuan3D-Omni image not found: {image_path}")
    if not point_path.exists():
        raise FileNotFoundError(f"Hunyuan3D-Omni point control not found: {point_path}")

    point, processed_points, processed_colors = _load_point_control(
        point_path,
        pipeline.device,
        pipeline.dtype,
        max_points=getattr(cfg, "hunyuan_omni_point_max_points", None),
        normalize=bool(getattr(cfg, "hunyuan_omni_normalize_point_control", True)),
        scale=float(getattr(cfg, "hunyuan_omni_point_scale", 0.98)),
        normalize_mode=getattr(cfg, "hunyuan_omni_point_normalize", "bbox"),
    )
    _export_processed_point_control(
        processed_points,
        processed_colors,
        sample_file(cfg, flag, f"{flag}_{model_name}_point_control_processed.ply"),
    )

    seed = getattr(cfg, "hunyuan_seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator(str(pipeline.device)).manual_seed(int(seed))

    steps = int(getattr(cfg, "hunyuan_omni_steps", getattr(cfg, "hunyuan_shape_steps", 50)))
    octree_resolution = int(getattr(cfg, "hunyuan_omni_octree_resolution", 512))
    guidance_scale = float(getattr(cfg, "hunyuan_omni_guidance_scale", 4.5))
    num_chunks = int(getattr(cfg, "hunyuan_omni_num_chunks", getattr(cfg, "hunyuan_num_chunks", 8000)))

    print(
        "Running Hunyuan3D-Omni point-control generation "
        f"({steps} steps, octree={octree_resolution}, guidance={guidance_scale})..."
    )
    start_time = time.time()
    result = pipeline(
        image=str(image_path),
        point=point,
        num_inference_steps=steps,
        octree_resolution=octree_resolution,
        mc_level=float(getattr(cfg, "hunyuan_omni_mc_level", 0.0)),
        guidance_scale=guidance_scale,
        num_chunks=num_chunks,
        generator=generator,
        fast_decode=bool(getattr(cfg, "hunyuan_omni_fast_decode", False)),
    )

    mesh = result["shapes"][0][0]
    if bool(getattr(cfg, "hunyuan_omni_postprocess", True)):
        mesh = floater_remover_cls()(mesh)
        mesh = degenerate_face_remover_cls()(mesh)

    final_glb_path = sample_file(cfg, flag, f"{flag}_{model_name}.glb")
    mesh.export(str(final_glb_path))
    sampled_point = result["sampled_point"][0]
    _export_sampled_point_cloud(
        sampled_point,
        sample_file(cfg, flag, f"{flag}_{model_name}_condition_sampled.ply"),
    )
    _export_mesh_points(
        final_glb_path,
        sample_file(cfg, flag, f"{flag}_{model_name}.ply"),
        int(getattr(cfg, "hunyuan_point_sample_num", 100000)),
    )
    print(f"Hunyuan3D-Omni generation finished in {int(time.time() - start_time)}s")
    print(f"Saved Hunyuan3D-Omni output to {final_glb_path}")
