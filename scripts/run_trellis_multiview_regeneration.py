#!/usr/bin/env python3
"""Generate the complete TRELLIS prior from edited multi-view conditions.

The script intentionally has no access to a partial point cloud or ground
truth. It converts fixed edited images into one complete mesh/Gaussian prior
for the subsequent no-GT registration stage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import types

import numpy as np
from PIL import Image
import torch
import trimesh


def _save_preview(frames: list[np.ndarray], output_path: Path, count: int = 8) -> None:
    if not frames:
        return
    ids = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    images = []
    for index in ids:
        array = np.asarray(frames[int(index)])
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(array[..., :3]).convert("RGB"))
    width, height = images[0].size
    board = Image.new("RGB", (width * len(images), height), "white")
    for column, image in enumerate(images):
        board.paste(image, (column * width, 0))
    board.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trellis-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("stochastic", "multidiffusion"), default="stochastic")
    parser.add_argument("--sparse-steps", type=int, default=12)
    parser.add_argument("--sparse-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    parser.add_argument("--sample-points", type=int, default=100_000)
    parser.add_argument("--preview-frames", type=int, default=24)
    args = parser.parse_args()

    if len(args.images) < 2:
        raise ValueError("TRELLIS multi-view regeneration requires at least two images")
    for path in (*args.images, args.model / "pipeline.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.trellis_repo / "trellis" / "pipelines").is_dir():
        raise FileNotFoundError(f"invalid TRELLIS repository: {args.trellis_repo}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.trellis_repo.resolve()))
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "xformers")

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils

    images = [Image.open(path).convert("RGBA") for path in args.images]
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(args.model.resolve()))
    pipeline.cuda()
    # The available LaS-Comp TRELLIS checkout intentionally disabled its GS
    # and RF decoders in ``decode_slat`` for an older completion experiment.
    # Restore the official mesh+Gaussian decode behaviour on this pipeline
    # instance only; never mutate the shared external checkout.
    def decode_mesh_and_gaussian(self, slat, formats):
        decoded = {}
        if "mesh" in formats:
            decoded["mesh"] = self.models["slat_decoder_mesh"](slat)
        if "gaussian" in formats:
            decoded["gaussian"] = self.models["slat_decoder_gs"](slat)
        return decoded

    pipeline.decode_slat = types.MethodType(decode_mesh_and_gaussian, pipeline)
    loaded_seconds = time.time() - started
    generated_started = time.time()
    outputs = pipeline.run_multi_image(
        images,
        seed=int(args.seed),
        formats=["mesh", "gaussian"],
        preprocess_image=True,
        mode=args.mode,
        sparse_structure_sampler_params={
            "steps": int(args.sparse_steps),
            "cfg_strength": float(args.sparse_cfg),
        },
        slat_sampler_params={
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_cfg),
        },
    )
    generation_seconds = time.time() - generated_started
    mesh_result = outputs["mesh"][0]
    gaussian = outputs["gaussian"][0]

    gaussian_path = output_dir / "trellis_gaussian.ply"
    gaussian.save_ply(str(gaussian_path))
    vertices = mesh_result.vertices.detach().float().cpu().numpy()
    faces = mesh_result.faces.detach().long().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    raw_glb_path = output_dir / "trellis_mesh_raw.glb"
    raw_ply_path = output_dir / "trellis_mesh_raw.ply"
    mesh.export(raw_glb_path)
    mesh.export(raw_ply_path)

    np.random.seed(int(args.seed))
    sampled, face_ids = trimesh.sample.sample_surface(mesh, int(args.sample_points))
    sampled_colours = None
    if getattr(mesh_result, "vertex_attrs", None) is not None:
        attributes = mesh_result.vertex_attrs.detach().float().cpu().numpy()
        if attributes.ndim == 2 and attributes.shape[1] >= 3:
            sampled_colours = np.clip(attributes[faces[face_ids]].mean(axis=1)[:, :3], 0.0, 1.0)
    sampled_cloud = trimesh.PointCloud(
        sampled,
        colors=None if sampled_colours is None else (sampled_colours * 255).astype(np.uint8),
    )
    sampled_path = output_dir / "trellis_mesh_sampled_100k.ply"
    sampled_cloud.export(sampled_path)

    preview_path = output_dir / "trellis_gaussian_orbit_preview.png"
    if int(args.preview_frames) > 0:
        frames = render_utils.render_video(
            gaussian,
            resolution=384,
            bg_color=(1, 1, 1),
            num_frames=int(args.preview_frames),
        )["color"]
        _save_preview(frames, preview_path)

    info = {
        "method": "trellis_image_large_tuning_free_multi_image_regeneration",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "partial_used": False,
        "model": str(args.model.resolve()),
        "trellis_repository": str(args.trellis_repo.resolve()),
        "trellis_commit": "2301d7ba00f6f101695022123bc01363603e4828",
        "images": [str(path.resolve()) for path in args.images],
        "seed": int(args.seed),
        "multi_image_mode": args.mode,
        "sparse_structure_sampler": {"steps": args.sparse_steps, "cfg_strength": args.sparse_cfg},
        "structured_latent_sampler": {"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
        "formats": ["mesh", "gaussian"],
        "sample_points": int(args.sample_points),
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "timing_seconds": {"model_load": loaded_seconds, "generation": generation_seconds},
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "outputs": {
            "gaussian": str(gaussian_path.resolve()),
            "raw_glb": str(raw_glb_path.resolve()),
            "raw_mesh": str(raw_ply_path.resolve()),
            "sampled_100k": str(sampled_path.resolve()),
            "preview": str(preview_path.resolve()) if preview_path.is_file() else None,
        },
    }
    (output_dir / "trellis_regeneration_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
