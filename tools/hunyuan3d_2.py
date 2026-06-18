import os
import sys
import time
from pathlib import Path

import open3d as o3d
import torch
from PIL import Image

from utils.dataUtils import glb2point
from utils.runtime import cfg_path, model_path, sample_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_shape_pipeline = None
_shape_pipeline_key = None
_background_remover = None
_pipeline_cls = None
_background_remover_cls = None


def _hunyuan_repo_root(cfg):
    env_path = os.environ.get("HUNYUAN3D_21_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return cfg_path(
        cfg,
        "paths",
        "hunyuan_repo_root",
        legacy_key="hunyuan_repo_root",
        default=PROJECT_ROOT.parent / "Hunyuan3D-2.1",
    )


def _ensure_hunyuan_imports(cfg):
    global _pipeline_cls, _background_remover_cls
    if _pipeline_cls is not None and _background_remover_cls is not None:
        return _pipeline_cls, _background_remover_cls

    shape_root = _hunyuan_repo_root(cfg) / "hy3dshape"
    if not shape_root.exists():
        raise FileNotFoundError(
            f"Hunyuan3D-2.1 code not found at {shape_root}. "
            "Set paths.hunyuan_repo_root or HUNYUAN3D_21_ROOT."
        )
    if str(shape_root) not in sys.path:
        sys.path.insert(0, str(shape_root))

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    _pipeline_cls = Hunyuan3DDiTFlowMatchingPipeline
    _background_remover_cls = BackgroundRemover
    return _pipeline_cls, _background_remover_cls


def _get_background_remover(cfg):
    global _background_remover
    if _background_remover is None:
        _, background_remover_cls = _ensure_hunyuan_imports(cfg)
        _background_remover = background_remover_cls()
    return _background_remover


def _get_model_root(cfg):
    return model_path(cfg, "hunyuan_model_path", "Hunyuan3D-2.1")


def _load_shape_pipeline(cfg):
    global _shape_pipeline
    global _shape_pipeline_key

    model_root = _get_model_root(cfg)
    subfolder = getattr(cfg, "hunyuan_shape_subfolder", "hunyuan3d-dit-v2-1")
    key = (str(model_root), subfolder, cfg.device, getattr(cfg, "hunyuan_variant", "fp16"))
    if _shape_pipeline is not None and _shape_pipeline_key == key:
        return _shape_pipeline

    pipeline_cls, _ = _ensure_hunyuan_imports(cfg)
    pipeline = pipeline_cls.from_pretrained(
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


def _prepare_input_image(cfg, img):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = img

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    if image.mode == "RGB":
        image = _get_background_remover(cfg)(image)
    elif image.mode == "RGBA":
        alpha = image.getchannel("A")
        if alpha.getextrema() == (255, 255):
            image = _get_background_remover(cfg)(image)

    return image.convert("RGBA")


def _export_point_cloud(glb_path, ply_path, num_points):
    pcd = glb2point(glb_path, down_sample=None, num_points=num_points)
    o3d.io.write_point_cloud(ply_path, pcd)


def hunyuan3d_2(cfg, flag, img):
    output_dir = Path(cfg.output_path) / flag
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = getattr(cfg, "generative_model", "hunyuan2.1")
    image = _prepare_input_image(cfg, img)

    shape_steps = getattr(cfg, "hunyuan_shape_steps", 50)
    octree_resolution = getattr(cfg, "hunyuan_octree_resolution", 384)
    num_chunks = getattr(cfg, "hunyuan_num_chunks", 8000)
    point_sample_num = getattr(cfg, "hunyuan_point_sample_num", 100000)
    seed = getattr(cfg, "hunyuan_seed", 12345)

    print("Running Hunyuan3D-2.1 shape generation...")
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

    shape_glb_path = sample_file(cfg, flag, f"{flag}_{model_name}_shape.glb")
    mesh.export(shape_glb_path)
    print(f"Shape generation finished in {int(shape_elapsed)}s")

    final_glb_path = sample_file(cfg, flag, f"{flag}_{model_name}.glb")
    mesh.export(final_glb_path)
    _export_point_cloud(
        str(final_glb_path),
        str(sample_file(cfg, flag, f"{flag}_{model_name}.ply")),
        point_sample_num,
    )
    print(f"Saved Hunyuan3D output to {final_glb_path}")
