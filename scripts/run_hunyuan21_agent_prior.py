#!/usr/bin/env python3
"""Generate an image-conditioned Hunyuan3D-2.1 agent-prior candidate.

Hunyuan3D-2.1 is deliberately treated as an *image-to-shape* executor.  It
does not expose a mesh/text edit API in the installed public pipeline, so any
agent text action must first create an approved semantic image.  The resulting
native mesh is never assumed to share the Redwood frame: it enters the common
global/PCA proper-Sim(3) registration before it can compete with the anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT.parents[1] / "Hunyuan3D-2.1"
DEFAULT_MODEL = ROOT.parent / "models" / "Hunyuan3D-2.1"


def _load_rgba(image_path: Path, repo_root: Path) -> Image.Image:
    shape_root = repo_root / "hy3dshape"
    if not shape_root.exists():
        raise FileNotFoundError(f"Hunyuan3D-2.1 code not found: {shape_root}")
    if str(shape_root) not in sys.path:
        sys.path.insert(0, str(shape_root))
    from hy3dshape.rembg import BackgroundRemover

    image = Image.open(image_path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    if image.mode == "RGB" or image.getchannel("A").getextrema() == (255, 255):
        image = BackgroundRemover()(image.convert("RGB"))
    return image.convert("RGBA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--point-count", type=int, default=100000)
    parser.add_argument("--caption", default=None,
                        help="Audit-only agent text; the public shape API does not consume it.")
    args = parser.parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    if not (args.model_root / args.subfolder).exists():
        raise FileNotFoundError(args.model_root / args.subfolder)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shape_root = args.repo_root / "hy3dshape"
    if str(shape_root) not in sys.path:
        sys.path.insert(0, str(shape_root))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    image = _load_rgba(args.image, args.repo_root)
    image.save(args.output_dir / "hunyuan_input_rgba.png")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(args.model_root), subfolder=args.subfolder, variant="fp16",
        device="cuda", dtype=torch.float16,
    )
    mesh = pipeline(
        image=image, num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution, num_chunks=args.num_chunks,
        generator=torch.Generator("cuda").manual_seed(args.seed), output_type="trimesh",
    )[0]
    mesh_path = args.output_dir / "hunyuan21_native.glb"
    mesh.export(mesh_path)
    points, _ = trimesh.sample.sample_surface(
        mesh, count=args.point_count, seed=np.random.default_rng(args.seed)
    )
    points_path = args.output_dir / "hunyuan21_native_sampled_100k.ply"
    trimesh.points.PointCloud(points).export(points_path)
    metadata = {
        "backend": "Hunyuan3D-2.1-Shape", "strict_zero_shot": True,
        "input_image": str(args.image.resolve()), "caption": args.caption,
        "caption_consumed_by_shape_model": False,
        "text_action_route": "upstream_semantic_image_only",
        "repo_root": str(args.repo_root.resolve()), "model_root": str(args.model_root.resolve()),
        "subfolder": args.subfolder, "seed": args.seed, "steps": args.steps,
        "octree_resolution": args.octree_resolution, "num_chunks": args.num_chunks,
        "native_to_method": "unknown_before_common_global_or_PCA_proper_Sim3_registration",
        "anisotropic_scale_used": False, "nonrigid_deformation_used": False,
        "outputs": {"mesh": str(mesh_path), "points": str(points_path)},
    }
    (args.output_dir / "hunyuan21_agent_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["outputs"], indent=2))


if __name__ == "__main__":
    main()
