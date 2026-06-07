import os
import sys
import time
from pathlib import Path

import open3d as o3d
import torch
from PIL import Image

from utils.dataUtils import glb2point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUNYUAN_REPO_ROOT = Path(os.environ.get("HUNYUAN3D_21_ROOT", "/home/chenrui/Hunyuan3D-2.1"))
HUNYUAN_SHAPE_ROOT = HUNYUAN_REPO_ROOT / "hy3dshape"
if str(HUNYUAN_SHAPE_ROOT) not in sys.path:
    sys.path.insert(0, str(HUNYUAN_SHAPE_ROOT))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover


_shape_pipeline = None
_shape_pipeline_key = None
_background_remover = None


def _get_background_remover():
    global _background_remover
    if _background_remover is None:
        _background_remover = BackgroundRemover()
    return _background_remover


def _get_model_root(cfg):
    return Path(getattr(cfg, "hunyuan_model_path", "models/Hunyuan3D-2.1")).resolve()


def _load_shape_pipeline(cfg):
    global _shape_pipeline
    global _shape_pipeline_key

    model_root = _get_model_root(cfg)
    subfolder = getattr(cfg, "hunyuan_shape_subfolder", "hunyuan3d-dit-v2-1")
    key = (str(model_root), subfolder, cfg.device, getattr(cfg, "hunyuan_variant", "fp16"))
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

    shape_glb_path = output_dir / f"{flag}_{model_name}_shape.glb"
    mesh.export(shape_glb_path)
    print(f"Shape generation finished in {int(shape_elapsed)}s")

    final_glb_path = output_dir / f"{flag}_{model_name}.glb"
    mesh.export(final_glb_path)
    _export_point_cloud(
        str(final_glb_path),
        str(output_dir / f"{flag}_{model_name}.ply"),
        point_sample_num,
    )
    print(f"Saved Hunyuan3D output to {final_glb_path}")


if __name__ == "__main__":
    class Config:
        output_path = "workspace"
        device = "cuda"
        generative_model = "hunyuan2.1"
        hunyuan_model_path = "models/Hunyuan3D-2.1"
        hunyuan_shape_subfolder = "hunyuan3d-dit-v2-1"
        hunyuan_enable_flashvdm = True
        hunyuan_shape_steps = 50
        hunyuan_octree_resolution = 384
        hunyuan_num_chunks = 8000
        hunyuan_point_sample_num = 100000
        hunyuan_seed = 12345

    hunyuan3d_2(
        Config(),
        flag="hunyuan21_example",
        img="/home/chenrui/Hunyuan3D-2.1/assets/demo.png",
    )
