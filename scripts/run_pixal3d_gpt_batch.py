#!/usr/bin/env python3
"""Run local Pixal3D on the accepted GPT semantic-image batch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image
import torch
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# A worktree intentionally does not duplicate multi-GB source checkouts.  The
# external source is read-only; generated assets still belong to ``--root``.
PIXAL3D_ROOT = Path(os.environ.get(
    "PIXAL3D_SOURCE", str(SHARED_ROOT / "models" / "Pixal3D"))).resolve()
if not PIXAL3D_ROOT.exists():
    raise FileNotFoundError(
        f"Pixal3D source is missing: {PIXAL3D_ROOT}. Set PIXAL3D_SOURCE.")
if str(PIXAL3D_ROOT) not in sys.path:
    sys.path.insert(0, str(PIXAL3D_ROOT))

import o_voxel
from moge.model.v2 import MoGeModel
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)
from src.mainline_paths import REDWOOD10_SAMPLE_IDS

SAMPLER_PARAMS = {
    "sparse_structure": {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    },
    "shape": {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    },
    "texture": {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    },
}


def prepare_local_pipeline_config(model_path: Path, rmbg_path: Path) -> str:
    source = model_path / "pipeline.json"
    target = model_path / "pipeline.local.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["name"] = "Pixal3DImageTo3DPipeline"
    config["args"]["rembg_model"]["args"]["model_name"] = str(rmbg_path.resolve())
    encoded = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if not target.exists() or target.read_text(encoding="utf-8") != encoded:
        target.write_text(encoded, encoding="utf-8")
    return target.name


def build_image_conditioner(
    dino_path: Path,
    image_size: int,
    grid_resolution: int,
    use_naf: bool = False,
    naf_target_size: int | None = None,
) -> DinoV3ProjFeatureExtractor:
    model = DinoV3ProjFeatureExtractor(
        model_name=str(dino_path.resolve()),
        image_size=image_size,
        grid_resolution=grid_resolution,
        use_naf_upsample=use_naf,
        naf_target_size=naf_target_size,
    )
    model.eval()
    return model


def init_pipeline(model_path: Path, dino_path: Path, rmbg_path: Path) -> Pixal3DImageTo3DPipeline:
    config_name = prepare_local_pipeline_config(model_path, rmbg_path)
    print(f"[Pipeline] Loading local checkpoints from {model_path}", flush=True)
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(str(model_path), config_file=config_name)
    pipeline.image_cond_model_ss = build_image_conditioner(dino_path, 512, 16)
    pipeline.image_cond_model_shape_512 = build_image_conditioner(dino_path, 512, 32, True, 512)
    pipeline.image_cond_model_shape_1024 = build_image_conditioner(dino_path, 1024, 64, True, 512)
    pipeline.image_cond_model_tex_1024 = build_image_conditioner(dino_path, 1024, 64, True, 1024)
    pipeline._device = torch.device("cuda")
    pipeline.low_vram = True
    for attr in (
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        getattr(pipeline, attr)._load_naf()
    print("[Pipeline] Local Pixal3D and NAF initialization passed", flush=True)
    return pipeline


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    return focal_length * resolution / 32.0


def camera_distance(camera_angle_x: float, image_resolution: int, mesh_scale: float = 1.0) -> float:
    # Official front-view point and coordinate convention from Pixal3D inference.py.
    xw = -1.0 / mesh_scale / 2.0
    xt = -image_resolution / 2.0
    return compute_f_pixels(camera_angle_x, image_resolution) * xw / xt


@torch.no_grad()
def estimate_camera(image_path: Path, model: MoGeModel, image_resolution: int = 512) -> dict[str, float]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
    output = model.infer(tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx = float(intrinsics[0, 0]) * image.shape[1]
    angle = 2.0 * math.atan(image.shape[1] / (2.0 * fx))
    return {
        "camera_angle_x": angle,
        "distance": camera_distance(angle, image_resolution),
        "mesh_scale": 1.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def save_native_moge_observation(
    image_path: Path,
    model: MoGeModel,
    cache_path: Path,
) -> Path:
    """Persist the exact FP16 MoGe observation used by native registration.

    Pixal camera estimation intentionally remains float32.  The native
    registration historically uses an FP16 MoGe pass, so this second pass is
    retained verbatim but is saved here to prevent a later model reload and
    duplicate inference in the registration runner.
    """
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.infer(tensor)
    points = output["points"].detach().float().cpu().numpy()
    valid = output["mask"].detach().cpu().numpy().astype(bool)
    valid &= np.isfinite(points).all(axis=-1)
    valid &= np.linalg.norm(points, axis=-1) > 1e-8
    ys, xs = np.where(valid)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema_version=np.asarray(1, dtype=np.int64),
        input_sha256=np.asarray(_sha256(image_path)),
        image_hw=np.asarray(image_rgb.shape[:2], dtype=np.int64),
        intrinsics=output["intrinsics"].squeeze().detach().float().cpu().numpy().astype(np.float64),
        points=points[valid].astype(np.float64),
        colors=(image_rgb[valid].astype(np.float64) / 255.0).clip(0.0, 1.0),
        pixel_xy=np.stack((xs, ys), axis=1).astype(np.float64),
    )
    return cache_path


def scene_geometry(scene: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(scene, trimesh.Trimesh):
        return scene
    geometry = scene.to_geometry()
    if not isinstance(geometry, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh after scene conversion, got {type(geometry)!r}")
    return geometry


def sample_surface_to_ply(scene: trimesh.Scene | trimesh.Trimesh, output: Path, count: int, seed: int) -> dict[str, object]:
    mesh = scene_geometry(scene)
    np.random.seed(seed)
    points, face_index = trimesh.sample.sample_surface(mesh, count)
    colors = None
    try:
        vertex_colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.float32)
        colors = vertex_colors[np.asarray(mesh.faces)[face_index]].mean(axis=1).astype(np.uint8)
    except Exception as exc:
        print(f"[PLY] Texture-to-point color conversion skipped: {exc}", flush=True)
    trimesh.points.PointCloud(points, colors=colors).export(output)
    return {
        "point_count": int(len(points)),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "bounds": np.asarray(mesh.bounds).tolist(),
    }


def export_result(
    pipeline: Pixal3DImageTo3DPipeline,
    mesh: object,
    resolution: int,
    glb_path: Path,
    ply_path: Path,
    point_count: int,
    seed: int,
    decimation_target: int,
    texture_size: int,
) -> dict[str, object]:
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    rotation = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    glb.apply_transform(rotation)
    glb.export(glb_path, extension_webp=True)
    return sample_surface_to_ply(glb, ply_path, point_count, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "gpt_version")
    parser.add_argument("--input-root", type=Path, default=None,
                        help="Semantic-image root; defaults to --root.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Pixal asset root; defaults to --root.")
    parser.add_argument("--input-name", default="gpt_image.png",
                        help="Per-sample semantic image filename.")
    parser.add_argument("--model", type=Path, default=SHARED_ROOT / "models" / "Pixal3D-weights")
    parser.add_argument("--dino", type=Path, default=SHARED_ROOT / "models" / "dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--moge", type=Path, default=SHARED_ROOT / "models" / "moge-2-vitl")
    parser.add_argument("--rmbg", type=Path, default=SHARED_ROOT / "models" / "RMBG-2.0")
    parser.add_argument("--ids", nargs="+", default=list(REDWOOD10_SAMPLE_IDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--point-count", type=int, default=100_000)
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_root = args.input_root or args.root
    output_root = args.output_root or args.root

    for required in (args.model, args.dino, args.moge, args.rmbg):
        if not required.exists():
            raise FileNotFoundError(required)

    moge_checkpoint = args.moge / "model.pt" if args.moge.is_dir() else args.moge
    if not moge_checkpoint.exists():
        raise FileNotFoundError(moge_checkpoint)

    # Resume must be cheap.  Loading Pixal, DINO and MoGe only to discover
    # that every requested GLB/PLY already exists wastes several GPU-minutes
    # without changing an output.  ``--overwrite`` remains the explicit way
    # to regenerate a completed prior after changing its image or parameters.
    pending_ids = []
    for sample_id in args.ids:
        output_dir = output_root / sample_id
        glb_path = output_dir / "pixal3d.glb"
        ply_path = output_dir / "pixal3d_sampled_100k.ply"
        if glb_path.exists() and ply_path.exists() and not args.overwrite:
            print(f"[Skip] {sample_id}: outputs already exist", flush=True)
        else:
            pending_ids.append(sample_id)
    if not pending_ids:
        print("[Done] No Pixal3D priors require generation.", flush=True)
        return

    pipeline = init_pipeline(args.model, args.dino, args.rmbg)

    prepared: dict[str, tuple[Path, dict[str, float]]] = {}
    preprocessed_paths: dict[str, Path] = {}
    native_moge_caches: dict[str, Path] = {}
    for sample_id in pending_ids:
        input_dir = input_root / sample_id
        output_dir = output_root / sample_id
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = input_dir / args.input_name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        processed_path = output_dir / "pixal3d_input.png"
        processed = pipeline.preprocess_image(Image.open(image_path))
        processed.save(processed_path)
        preprocessed_paths[sample_id] = processed_path

    print(f"[MoGe] Loading local camera model from {moge_checkpoint}", flush=True)
    moge = MoGeModel.from_pretrained(str(moge_checkpoint.resolve())).cuda().eval()
    for sample_id, processed_path in preprocessed_paths.items():
        prepared[sample_id] = (processed_path, estimate_camera(processed_path, moge))
        native_moge_caches[sample_id] = save_native_moge_observation(
            processed_path, moge,
            output_root / sample_id / "pixal_moge_fp16_observation.npz",
        )
        print(f"[Camera] {sample_id}: {prepared[sample_id][1]}", flush=True)
    moge.cpu()
    del moge
    gc.collect()
    torch.cuda.empty_cache()

    for sample_id in pending_ids:
        input_dir = input_root / sample_id
        sample_dir = output_root / sample_id
        glb_path = sample_dir / "pixal3d.glb"
        ply_path = sample_dir / "pixal3d_sampled_100k.ply"
        metadata_path = sample_dir / "pixal3d_metadata.json"
        processed_path, camera = prepared[sample_id]
        image = Image.open(processed_path).convert("RGBA")
        started = time.time()
        print(f"[Generate] {sample_id} at {args.resolution}", flush=True)
        meshes, (_, _, actual_resolution) = pipeline.run(
            image,
            camera_params=camera,
            seed=args.seed,
            sparse_structure_sampler_params=SAMPLER_PARAMS["sparse_structure"],
            shape_slat_sampler_params=SAMPLER_PARAMS["shape"],
            tex_slat_sampler_params=SAMPLER_PARAMS["texture"],
            preprocess_image=False,
            return_latent=True,
            pipeline_type=f"{args.resolution}_cascade",
            max_num_tokens=49_152,
        )
        geometry = export_result(
            pipeline,
            meshes[0],
            actual_resolution,
            glb_path,
            ply_path,
            args.point_count,
            args.seed,
            args.decimation_target,
            args.texture_size,
        )
        metadata = {
            "sample_id": sample_id,
            "input": str((input_dir / args.input_name).resolve()),
            "preprocessed_input": str(processed_path.resolve()),
            "model": str(args.model.resolve()),
            "dino": str(args.dino.resolve()),
            "moge": str(moge_checkpoint.resolve()),
            "rmbg": str(args.rmbg.resolve()),
            "attention_backend": os.environ["ATTN_BACKEND"],
            "seed": args.seed,
            "requested_resolution": args.resolution,
            "actual_resolution": int(actual_resolution),
            "decimation_target": args.decimation_target,
            "texture_size": args.texture_size,
            "sampler_params": SAMPLER_PARAMS,
            "camera": camera,
            "native_moge_observation_cache": str(native_moge_caches[sample_id].resolve()),
            "native_moge_observation_contract": "FP16 MoGe on pixal3d_input.png; consumed by native registration",
            "glb": str(glb_path.resolve()),
            "sampled_ply": str(ply_path.resolve()),
            "elapsed_seconds": time.time() - started,
            **geometry,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[Done] {sample_id}: {glb_path} and {ply_path}", flush=True)
        del meshes
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
