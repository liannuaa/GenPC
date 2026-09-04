#!/usr/bin/env python3
"""Execute a Nano3D mesh-edit action from an externally verified target image.

The controller supplies a registered complete mesh, its deterministic Nano3D
front render, and an agent-produced target render.  Nano3D's FlowEdit and
Voxel/SLAT merge propose a complete mesh; it is still only a candidate and
must pass the shared saved-view proper-Sim(3) gate afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import cv2
from PIL import Image
import torch
import trimesh


# The shared 24 GB environment provides xFormers but not FlashAttention. Keep
# the backend explicit so a worktree invocation reproduces the verified runner
# rather than inheriting an unavailable vendor default from the shell.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
NANO3D_ROOT = PROJECT_ROOT / "models" / "Nano3D"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(NANO3D_ROOT) not in sys.path:
    sys.path.insert(0, str(NANO3D_ROOT))


def white_background(source: Path, destination: Path) -> None:
    image = Image.open(source).convert("RGBA")
    result = Image.new("RGB", image.size, "white")
    result.paste(image, mask=image.getchannel("A"))
    result.save(destination)


def camera_locked_source(
    registered_prior: Path,
    partial: Path,
    camera_path: Path,
    semantic_target: Path,
    source_destination: Path,
    target_destination: Path,
    prior_mask_destination: Path,
    partial_mask_destination: Path,
    *,
    padding: float,
    target_mode: str,
) -> None:
    """Compose a source image in exactly the saved semantic-camera frame.

    Nano3D still encodes the source mesh from its canonical multiview renders.
    This image is only the 2-D edit condition.  Copying the semantic target
    inside the registered-prior silhouette makes the requested change a
    camera-locked silhouette/residual edit, rather than an accidental camera
    conversion from Nano3D's canonical ``front`` image.
    """
    from scripts.run_registration_deformation_fusion_ablation import SavedCameraProjector

    prior = np.asarray(trimesh.load(registered_prior, force="mesh", process=False).vertices)
    if prior.size == 0:
        # Point-cloud PLYs are loaded as PointCloud by trimesh.
        prior_object = trimesh.load(registered_prior, process=False)
        prior = np.asarray(prior_object.vertices)
    partial_object = trimesh.load(partial, process=False)
    partial_points = np.asarray(partial_object.vertices)
    target_image = Image.open(semantic_target).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    target_rgb = np.asarray(target_image)
    projector = SavedCameraProjector.from_partial(
        partial_points, camera_path, padding=padding, image_shape=target_rgb.shape[:2], device="cpu"
    )
    pixel, depth = projector.project(prior)
    height, width = target_rgb.shape[:2]
    x = np.rint(pixel[:, 0]).astype(np.int64)
    y = np.rint(pixel[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & np.isfinite(depth)
    if not np.any(valid):
        raise RuntimeError("registered prior has no points inside the saved-camera crop")
    # The exact depth convention is irrelevant for silhouette support.  Keep
    # the closest sample per pixel to avoid rear-surface holes/overdraw.
    order = np.argsort(depth[valid])
    flat = (y[valid][order] * width + x[valid][order])
    first = np.empty(len(flat), dtype=bool)
    first[0] = True
    first[1:] = flat[1:] != flat[:-1]
    prior_mask = np.zeros((height, width), dtype=np.uint8)
    prior_mask.flat[flat[first]] = 255
    # Samples originate from a surface cloud, so fill only sub-pixel gaps; a
    # large close would invent geometry and defeat the conservative edit goal.
    prior_mask = cv2.dilate(prior_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    prior_mask = cv2.morphologyEx(prior_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    source = np.full_like(target_rgb, 255)
    source[prior_mask > 0] = target_rgb[prior_mask > 0]

    partial_pixel, _ = projector.project(partial_points)
    partial_x = np.rint(partial_pixel[:, 0]).astype(np.int64)
    partial_y = np.rint(partial_pixel[:, 1]).astype(np.int64)
    partial_valid = ((partial_x >= 0) & (partial_x < width) &
                     (partial_y >= 0) & (partial_y < height))
    partial_mask = np.zeros((height, width), dtype=np.uint8)
    partial_mask[partial_y[partial_valid], partial_x[partial_valid]] = 255
    partial_mask = cv2.dilate(partial_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    partial_mask = cv2.morphologyEx(partial_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    if target_mode == "global_semantic":
        edit_target = target_rgb
    elif target_mode == "partial_supported":
        # Only introduce target evidence where the depth scan actually has
        # support.  Everything else is byte-identical to the source condition,
        # so FlowEdit cannot interpret unobserved semantic pixels as a request
        # to rewrite or enlarge hidden geometry.
        edit_target = source.copy()
        edit_target[partial_mask > 0] = target_rgb[partial_mask > 0]
    else:
        raise ValueError(f"Unknown camera-locked target mode: {target_mode}")
    Image.fromarray(source).save(source_destination)
    Image.fromarray(edit_target).save(target_destination)
    Image.fromarray(prior_mask).save(prior_mask_destination)
    Image.fromarray(partial_mask).save(partial_mask_destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-mesh", required=True, type=Path)
    parser.add_argument("--source-render", type=Path,
                        help="Canonical source render (legacy mode only).")
    parser.add_argument("--target-render", required=True, type=Path)
    parser.add_argument("--registered-prior", type=Path,
                        help="Registered prior point cloud for a saved-camera edit condition.")
    parser.add_argument("--partial", type=Path,
                        help="Partial cloud defining the saved-camera crop.")
    parser.add_argument("--camera", type=Path,
                        help="DepthPrompting saved camera for camera-locked editing.")
    parser.add_argument("--camera-padding", type=float, default=0.15)
    parser.add_argument("--camera-locked-target-mode", choices=("global_semantic", "partial_supported"),
                        default="global_semantic",
                        help="Use only partial-supported semantic evidence for a conservative local edit.")
    parser.add_argument("--trellis-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--views", type=int, default=150)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--st-step", type=int, default=12,
                        help="Nano3D edit start step; larger values make a more conservative edit.")
    parser.add_argument("--source-encoding-dir", type=Path,
                        help="Reuse a completed canonical source encoding (150 renders + DINO features).")
    parser.add_argument("--source-voxel-latent", type=Path,
                        help="Reuse the matching source voxel latent. Required with --source-encoding-dir.")
    parser.add_argument("--sample-count", type=int, default=100000)
    parser.add_argument("--fast-geometry-renders", action="store_true",
                        help="Use Nano3D's official one-sample geometry override for source encoding.")
    args = parser.parse_args()

    camera_locked = any(value is not None for value in (args.registered_prior, args.partial, args.camera))
    if camera_locked and not all(value is not None for value in (args.registered_prior, args.partial, args.camera)):
        raise ValueError("--registered-prior, --partial and --camera must be supplied together")
    required = [args.source_mesh, args.target_render]
    required.extend((args.registered_prior, args.partial, args.camera) if camera_locked else (args.source_render,))
    if any(path is None or not path.exists() for path in required):
        raise FileNotFoundError("source mesh, target render and complete edit-condition inputs must exist")
    if not (args.trellis_model / "pipeline.json").exists():
        raise FileNotFoundError("--trellis-model must be a local TRELLIS-image-large checkout")
    if (args.source_encoding_dir is None) != (args.source_voxel_latent is None):
        raise ValueError("--source-encoding-dir and --source-voxel-latent must be supplied together")
    if args.source_encoding_dir is not None:
        required_cache = (args.source_encoding_dir / "voxels.ply", args.source_encoding_dir / "features.npz",
                          args.source_voxel_latent)
        if not all(path.exists() for path in required_cache):
            raise FileNotFoundError("cached source encoding is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Nano3D's vendor code accepts this explicit local source for its two
    # encoders, preventing hidden hub downloads during an agent action.
    os.environ["NANO3D_TRELLIS_IMAGE_PATH"] = str(args.trellis_model.resolve())
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    from inference.model_utils import extract_and_decode_voxel, inject_methods, load_sparse_structure_encoder
    from inference.rendering import render_3d_asset
    from inference.voxelization import extract_features

    source_image = args.output_dir / "source_render_white.png"
    target_image = args.output_dir / "target_render_512.png"
    if camera_locked:
        camera_locked_source(
            args.registered_prior, args.partial, args.camera, args.target_render, source_image, target_image,
            args.output_dir / "source_render_saved_camera_mask.png",
            args.output_dir / "partial_saved_camera_mask.png", padding=args.camera_padding,
            target_mode=args.camera_locked_target_mode,
        )
    else:
        white_background(args.source_render, source_image)
        Image.open(args.target_render).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS).save(target_image)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(args.trellis_model.resolve()))
    pipeline.cuda()
    pipeline = inject_methods(load_sparse_structure_encoder(pipeline))
    if args.source_encoding_dir is None:
        dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True).eval().cuda()
        render_dir = args.output_dir / "source_encoding"
        render_3d_asset(
            model_path=str(args.source_mesh), output_dir=str(render_dir), num_views=args.views,
            resolution=512, engine="CYCLES", geo_mode=bool(args.fast_geometry_renders), save_mesh=True,
        )
        extract_features(str(render_dir), dinov2_model, batch_size=args.feature_batch_size, voxel_size=1 / 64)
        source_latent_dir = args.output_dir / "source_voxel"
        source_latent = extract_and_decode_voxel(pipeline, str(render_dir), str(source_latent_dir))["latent"]
        del dinov2_model
    else:
        render_dir = args.source_encoding_dir
        source_latent = args.source_voxel_latent
    torch.cuda.empty_cache()
    outputs = pipeline.run(
        str(source_image), str(target_image),
        source_ply_path=str(render_dir / "voxels.ply"),
        source_voxel_latent_path=str(source_latent),
        source_slat_path=str(render_dir / "features.npz"),
        editing_mode="replace", seed=args.seed, output_path=str(args.output_dir), st_step=args.st_step,
    )
    mesh_path = args.output_dir / "nano3d_native.glb"
    # Textured export is optional for the geometric registration loop.  The
    # upstream texture baker requires a separate CUDA extension; preserve the
    # edited mesh if that extension is not available on the active GPU stack.
    try:
        with torch.enable_grad():
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0], outputs["mesh"][0], simplify=.95,
                fill_holes=False, texture_size=1024)
        glb.export(mesh_path)
        export_mode = "textured_glb"
    except ModuleNotFoundError as exc:
        if exc.name != "diff_gaussian_rasterization":
            raise
        edited_mesh = outputs["mesh"][0]
        geometry = trimesh.Trimesh(
            vertices=edited_mesh.vertices.detach().cpu().numpy(),
            faces=edited_mesh.faces.detach().cpu().numpy(),
            process=False,
        )
        geometry.export(mesh_path)
        export_mode = "geometry_only_glb"
        print("Texture rasterizer unavailable; exported geometry-only GLB.")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, count=args.sample_count,
                                               seed=np.random.default_rng(args.seed))
    points_path = args.output_dir / "nano3d_native_sampled_100k.ply"
    trimesh.points.PointCloud(points).export(points_path)
    (args.output_dir / "nano3d_agent_metadata.json").write_text(json.dumps({
        "backend": "Nano3D", "strict_zero_shot": True, "source_mesh": str(args.source_mesh.resolve()),
        "source_render": str(args.source_render.resolve()) if args.source_render else None,
        "target_render": str(args.target_render.resolve()),
        "camera_locked_edit_condition": bool(camera_locked),
        "camera_locked_target_mode": args.camera_locked_target_mode if camera_locked else None,
        "registered_prior": str(args.registered_prior.resolve()) if args.registered_prior else None,
        "partial": str(args.partial.resolve()) if args.partial else None,
        "camera": str(args.camera.resolve()) if args.camera else None,
        "trellis_model": str(args.trellis_model.resolve()), "seed": args.seed, "views": args.views,
        "fast_geometry_renders": bool(args.fast_geometry_renders), "editing_mode": "replace",
        "st_step": args.st_step, "source_encoding_cache": str(args.source_encoding_dir.resolve()) if args.source_encoding_dir else None,
        "export_mode": export_mode,
        "native_output_requires_global_proper_sim3": True, "nonrigid_deformation_used": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"mesh": str(mesh_path), "points": str(points_path)}, indent=2))


if __name__ == "__main__":
    main()
