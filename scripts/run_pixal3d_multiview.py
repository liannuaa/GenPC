#!/usr/bin/env python3
"""Run calibrated Pixal3D-MV and export the mainline asset contract."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
PIXAL3D_ROOT = Path(
    os.environ.get("PIXAL3D_SOURCE", str(SHARED_ROOT / "models" / "Pixal3D"))
).resolve()
for path_entry in (ROOT, PIXAL3D_ROOT):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from inference_mv import check_main_view, load_rgba, load_views, make_rembg
from pixal3d.pipelines import Pixal3DMVImageTo3DPipeline
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjMultiViewFeatureExtractor,
)
from scripts.run_pixal3d_gpt_batch import (
    SAMPLER_PARAMS,
    export_result,
    load_naf_without_src_collision,
)


def prepare_local_mv_config(model_path: Path, rmbg_path: Path) -> str:
    """Pin the MV pipeline to local weights and the local matting model."""
    source = model_path / "pipeline_mv.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing; download the Pixal3D multi-view checkpoint update"
        )
    target = model_path / "pipeline_mv.local.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["name"] = "Pixal3DMVImageTo3DPipeline"
    config["args"]["rembg_model"]["args"]["model_name"] = str(rmbg_path.resolve())
    encoded = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if not target.exists() or target.read_text(encoding="utf-8") != encoded:
        target.write_text(encoded, encoding="utf-8")
    return target.name


def build_conditioner(
    dino_path: Path,
    *,
    image_size: int,
    grid_resolution: int,
    use_naf: bool = False,
    naf_target_size: int | None = None,
) -> DinoV3ProjMultiViewFeatureExtractor:
    model = DinoV3ProjMultiViewFeatureExtractor(
        model_name=str(dino_path.resolve()),
        image_size=image_size,
        grid_resolution=grid_resolution,
        use_naf_upsample=use_naf,
        naf_target_size=naf_target_size,
        multiview_fusion="average",
    )
    model.eval()
    return model


def init_pipeline(
    model_path: Path,
    dino_path: Path,
    rmbg_path: Path,
) -> Pixal3DMVImageTo3DPipeline:
    config_name = prepare_local_mv_config(model_path, rmbg_path)
    print(f"[Pipeline] Loading calibrated Pixal3D-MV from {model_path}", flush=True)
    pipeline = Pixal3DMVImageTo3DPipeline.from_pretrained(
        str(model_path), config_file=config_name
    )
    pipeline.image_cond_model_ss = build_conditioner(
        dino_path, image_size=512, grid_resolution=16
    )
    pipeline.image_cond_model_shape_512 = build_conditioner(
        dino_path,
        image_size=512,
        grid_resolution=32,
        use_naf=True,
        naf_target_size=512,
    )
    pipeline.image_cond_model_shape_1024 = build_conditioner(
        dino_path,
        image_size=1024,
        grid_resolution=64,
        use_naf=True,
        naf_target_size=512,
    )
    pipeline.image_cond_model_tex_1024 = build_conditioner(
        dino_path,
        image_size=1024,
        grid_resolution=64,
        use_naf=True,
        naf_target_size=1024,
    )
    pipeline._device = torch.device("cuda")
    pipeline.low_vram = True
    for attr in (
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        load_naf_without_src_collision(getattr(pipeline, attr))
    print("[Pipeline] 24-GB low-VRAM initialization passed", flush=True)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=SHARED_ROOT / "models" / "Pixal3D-weights",
    )
    parser.add_argument(
        "--dino",
        type=Path,
        default=SHARED_ROOT / "models" / "dinov3-vitl16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--rmbg",
        type=Path,
        default=SHARED_ROOT / "models" / "RMBG-2.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--num-views", type=int, choices=(3, 4), default=4)
    parser.add_argument("--point-count", type=int, default=100_000)
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for required in (PIXAL3D_ROOT, args.model, args.dino, args.rmbg):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.resolution != 1024:
        print(
            "[Warning] 1536 may exceed a 24-GB card; 1024 is the validated default.",
            flush=True,
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    glb_path = output / "pixal3d.glb"
    ply_path = output / "pixal3d_sampled_100k.ply"
    metadata_path = output / "pixal3d_metadata.json"
    if (
        glb_path.is_file()
        and ply_path.is_file()
        and metadata_path.is_file()
        and not args.overwrite
    ):
        print(f"[Skip] completed Pixal3D-MV asset: {output}", flush=True)
        return

    torch.cuda.reset_peak_memory_stats()
    pipeline = init_pipeline(args.model.resolve(), args.dino.resolve(), args.rmbg.resolve())
    rembg = make_rembg(pipeline)
    views = load_views(
        str(args.views_dir.resolve()),
        num_views=args.num_views,
        rembg=rembg,
        image_sizes=(512, 1024),
    )
    check_main_view(views)

    main_rgba, _ = load_rgba(str(args.views_dir / "front.png"), rembg)
    pixal_input = output / "pixal3d_input.png"
    main_rgba.save(pixal_input)
    camera = {
        "camera_angle_x": float(views["camera_angle_x"][0, 0]),
        "distance": float(views["camera_distance"][0, 0]),
        "mesh_scale": float(views["mesh_scale"]),
    }

    started = time.time()
    with torch.no_grad():
        meshes, (_, _, actual_resolution) = pipeline.run_mv(
            views,
            seed=args.seed,
            sparse_structure_sampler_params=SAMPLER_PARAMS["sparse_structure"],
            shape_slat_sampler_params=SAMPLER_PARAMS["shape"],
            tex_slat_sampler_params=SAMPLER_PARAMS["texture"],
            return_latent=True,
            pipeline_type=f"{args.resolution}_cascade",
            max_num_tokens=49_152,
        )
    geometry = export_result(
        pipeline,
        meshes[0],
        int(actual_resolution),
        glb_path,
        ply_path,
        args.point_count,
        args.seed,
        args.decimation_target,
        args.texture_size,
    )
    metadata = {
        "method": "calibrated_pixal3d_multiview",
        "ground_truth_used": False,
        "views_dir": str(args.views_dir.resolve()),
        "transforms": str((args.views_dir / "transforms.json").resolve()),
        "view_order": list(views["view_names"]),
        "num_views": args.num_views,
        "frame_0_is_camera1_gauge": True,
        "model": str(args.model.resolve()),
        "pixal_source": str(PIXAL3D_ROOT),
        "pixal_source_commit": "f7cf38429b0bd264f1995f0f8743a88b1c728b94",
        "dino": str(args.dino.resolve()),
        "rmbg": str(args.rmbg.resolve()),
        "attention_backend": os.environ["ATTN_BACKEND"],
        "low_vram": True,
        "seed": args.seed,
        "requested_resolution": args.resolution,
        "actual_resolution": int(actual_resolution),
        "camera": camera,
        "sampler_params": SAMPLER_PARAMS,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "pixal3d_input": str(pixal_input),
        "glb": str(glb_path),
        "sampled_ply": str(ply_path),
        "elapsed_seconds": time.time() - started,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        **geometry,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] {glb_path}", flush=True)
    print(f"[Done] {ply_path}", flush=True)
    del meshes, pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
