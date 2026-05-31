import os
import sys
import time
from pathlib import Path

import open3d as o3d
import torch
from PIL import Image

from utils.dataUtils import glb2point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUNYUAN_REPO_ROOT = PROJECT_ROOT / "models" / "Hunyuan3D-2"
if str(HUNYUAN_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(HUNYUAN_REPO_ROOT))

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


_shape_pipeline = None
_paint_pipeline = None
_shape_pipeline_key = None
_paint_pipeline_key = None
_background_remover = None


def _get_background_remover():
    global _background_remover
    if _background_remover is None:
        _background_remover = BackgroundRemover()
    return _background_remover


def _get_model_root(cfg):
    return Path(getattr(cfg, "hunyuan_model_path", "models/Hunyuan3D-2-ms")).resolve()


def _load_shape_pipeline(cfg):
    global _shape_pipeline
    global _shape_pipeline_key

    model_root = _get_model_root(cfg)
    subfolder = getattr(cfg, "hunyuan_shape_subfolder", "hunyuan3d-dit-v2-0")
    key = (str(model_root), subfolder, cfg.device)
    if _shape_pipeline is not None and _shape_pipeline_key == key:
        return _shape_pipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(model_root),
        subfolder=subfolder,
        variant=getattr(cfg, "hunyuan_variant", "fp16"),
        device=cfg.device,
        dtype=torch.float16,
    )
    if getattr(cfg, "hunyuan_enable_flashvdm", True):
        pipeline.enable_flashvdm(mc_algo="mc")

    _shape_pipeline = pipeline
    _shape_pipeline_key = key
    return _shape_pipeline


def _load_paint_pipeline(cfg):
    global _paint_pipeline
    global _paint_pipeline_key

    model_root = _get_model_root(cfg)
    subfolder = getattr(cfg, "hunyuan_paint_subfolder", "hunyuan3d-paint-v2-0")
    delight_root = model_root / "hunyuan3d-delight-v2-0"
    delight_index = delight_root / "model_index.json"
    if not delight_index.exists():
        raise FileNotFoundError(
            f"Missing {delight_index}. The local Hunyuan delight model is incomplete, "
            "so paint generation cannot run from this checkout."
        )
    key = (str(model_root), subfolder)
    if _paint_pipeline is not None and _paint_pipeline_key == key:
        return _paint_pipeline

    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        str(model_root),
        subfolder=subfolder,
    )
    _paint_pipeline = pipeline
    _paint_pipeline_key = key
    return _paint_pipeline


def _prepare_input_image(img):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = img

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    if image.mode == "RGB":
        image = _get_background_remover()(image)
    elif image.mode == "RGBA":
        alpha = image.getchannel("A")
        if alpha.getextrema() == (255, 255):
            image = _get_background_remover()(image)

    return image.convert("RGBA")


def _export_point_cloud(glb_path, ply_path, num_points):
    pcd = glb2point(glb_path, down_sample=None, num_points=num_points)
    o3d.io.write_point_cloud(ply_path, pcd)


def hunyuan3d_2(cfg, flag, img):
    output_dir = Path(cfg.output_path) / flag
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = getattr(cfg, "generative_model", "hunyuan2.0")
    image = _prepare_input_image(img)

    shape_steps = getattr(cfg, "hunyuan_shape_steps", 50)
    octree_resolution = getattr(cfg, "hunyuan_octree_resolution", 380)
    num_chunks = getattr(cfg, "hunyuan_num_chunks", 20000)
    point_sample_num = getattr(cfg, "hunyuan_point_sample_num", 100000)
    seed = getattr(cfg, "hunyuan_seed", 12345)
    enable_paint = getattr(cfg, "hunyuan_paint", False)

    print("Running Hunyuan3D-2.0 shape generation...")
    start_time = time.time()
    shape_pipeline = _load_shape_pipeline(cfg)
    mesh = shape_pipeline(
        image=image,
        num_inference_steps=shape_steps,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        generator=torch.manual_seed(seed),
        output_type="trimesh",
    )[0]
    shape_elapsed = time.time() - start_time

    shape_glb_path = output_dir / f"{flag}_{model_name}_shape.glb"
    mesh.export(shape_glb_path)
    print(f"Shape generation finished in {int(shape_elapsed)}s")

    final_mesh = mesh
    if enable_paint:
        try:
            print("Running Hunyuan3D-2.0 paint generation...")
            paint_start = time.time()
            paint_pipeline = _load_paint_pipeline(cfg)
            painted_mesh = paint_pipeline(mesh, image=image)
            paint_elapsed = time.time() - paint_start
            paint_glb_path = output_dir / f"{flag}_{model_name}_paint.glb"
            painted_mesh.export(paint_glb_path)
            print(f"Paint generation finished in {int(paint_elapsed)}s")
            final_mesh = painted_mesh
        except Exception as exc:
            print(f"Hunyuan paint failed, fallback to shape-only mesh: {exc}")

    final_glb_path = output_dir / f"{flag}_{model_name}.glb"
    final_mesh.export(final_glb_path)
    _export_point_cloud(
        str(final_glb_path),
        str(output_dir / f"{flag}_{model_name}.ply"),
        point_sample_num,
    )
    print(f"Saved Hunyuan3D output to {final_glb_path}")
