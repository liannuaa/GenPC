#!/usr/bin/env python3
"""Select the TRELLIS canonical view matching a Camera-1 condition image."""

from __future__ import annotations

import argparse
import cv2
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import trimesh


def _normalise(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _look_at(eye: np.ndarray, up: np.ndarray, roll_degrees: float = 0.0) -> np.ndarray:
    backward = _normalise(eye)
    right = _normalise(np.cross(up, backward))
    true_up = _normalise(np.cross(backward, right))
    roll = math.radians(float(roll_degrees))
    rolled_right = math.cos(roll) * right + math.sin(roll) * true_up
    rolled_up = -math.sin(roll) * right + math.cos(roll) * true_up
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([rolled_right, rolled_up, backward], axis=1)
    pose[:3, 3] = eye
    return pose


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = [mesh for mesh in loaded.dump() if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        raise ValueError(f"no mesh in {path}")
    return trimesh.util.concatenate(meshes)


def _render_coloured_points(
    vertices: np.ndarray,
    colours: np.ndarray,
    pose: np.ndarray,
    fov: float,
    resolution: int,
) -> np.ndarray:
    """Render a dense coloured carrier when an EGL context is unavailable."""
    inverse = np.linalg.inv(pose)
    camera = vertices @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -camera[:, 2]
    focal = 0.5 * resolution / math.tan(fov * 0.5)
    valid = depth > 1e-6
    x = np.rint(focal * camera[:, 0] / np.maximum(depth, 1e-8) + 0.5 * resolution).astype(int)
    y = np.rint(-focal * camera[:, 1] / np.maximum(depth, 1e-8) + 0.5 * resolution).astype(int)
    valid &= (x >= 0) & (x < resolution) & (y >= 0) & (y < resolution)
    ids = np.flatnonzero(valid)
    canvas = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    if len(ids) == 0:
        return canvas

    flat = y[ids] * resolution + x[ids]
    order = np.lexsort((depth[ids], flat))
    ordered_flat = flat[order]
    front = np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]
    selected = ids[order[front]]
    object_rgb = np.zeros_like(canvas)
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    object_rgb[y[selected], x[selected]] = colours[selected, :3]
    mask[y[selected], x[selected]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(mask, kernel)
    dilated_rgb = np.stack(
        [cv2.dilate(object_rgb[..., channel], kernel) for channel in range(3)],
        axis=-1,
    )
    canvas[dilated_mask > 0] = dilated_rgb[dilated_mask > 0]
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--colored-points", type=Path,
                        help="Optional TRELLIS coloured carrier in the same canonical frame.")
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yaw-step", type=float, default=10.0)
    parser.add_argument("--pitches", type=float, nargs="+", default=(-30, -15, 0, 15, 30, 45))
    parser.add_argument("--rolls", type=float, nargs="+", default=(0, 90, 180, 270))
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh = _load_mesh(args.mesh)
    centre = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    mesh.vertices -= centre
    radius = max(float(np.linalg.norm(mesh.vertices, axis=1).max()), 1e-6)
    point_vertices = point_colours = None
    if args.colored_points is not None:
        cloud = trimesh.load(str(args.colored_points), process=False)
        point_vertices = np.asarray(cloud.vertices, dtype=np.float32) - centre.astype(np.float32)
        point_colours = np.asarray(cloud.colors, dtype=np.uint8)
        if point_colours.ndim != 2 or point_colours.shape[1] < 3:
            point_colours = np.full((len(point_vertices), 3), 128, dtype=np.uint8)
    fov = math.radians(40.0)
    distance = radius / math.sin(fov * 0.5) * 1.08
    renderer = scene = camera = light = None
    render_backend = "pyrender_egl"
    try:
        scene = pyrender.Scene(
            bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
            ambient_light=np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32),
        )
        if point_vertices is None:
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        else:
            scene.add(pyrender.Mesh.from_points(point_vertices, colors=point_colours))
        camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.2)
        renderer = pyrender.OffscreenRenderer(args.resolution, args.resolution)
    except (RuntimeError, ValueError):
        if point_vertices is None:
            raise
        render_backend = "coloured_point_zbuffer"
    frames: list[Image.Image] = []
    records: list[dict] = []
    try:
        for pitch_deg in args.pitches:
            pitch = math.radians(float(pitch_deg))
            for yaw_deg in np.arange(0.0, 360.0, float(args.yaw_step)):
                yaw = math.radians(float(yaw_deg))
                eye = distance * np.array([
                    math.sin(yaw) * math.cos(pitch),
                    math.cos(yaw) * math.cos(pitch),
                    math.sin(pitch),
                ])
                for roll_deg in args.rolls:
                    pose = _look_at(eye, np.array([0.0, 0.0, 1.0]), roll_deg)
                    if renderer is None:
                        colour = _render_coloured_points(
                            point_vertices, point_colours, pose, fov, args.resolution,
                        )
                    else:
                        camera_node = scene.add(camera, pose=pose)
                        light_node = scene.add(light, pose=pose)
                        colour, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                        scene.remove_node(camera_node)
                        scene.remove_node(light_node)
                    frames.append(Image.fromarray(colour[..., :3], mode="RGB"))
                    records.append({
                        "yaw_degrees": float(yaw_deg),
                        "pitch_degrees": float(pitch_deg),
                        "roll_degrees": float(roll_deg),
                        "camera_pose": pose.tolist(),
                    })
    finally:
        if renderer is not None:
            renderer.delete()

    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True,
    ).eval().cuda()
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    target = transform(Image.open(args.condition).convert("RGB"))[None].cuda()
    target_rgb = np.asarray(
        Image.open(args.condition).convert("RGB").resize(
            (args.resolution, args.resolution), Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )
    target_mask = np.any(target_rgb < 245, axis=2)
    target_blur = cv2.GaussianBlur(target_rgb.astype(np.float32) / 255.0, (15, 15), 0)
    target_y, target_x = np.nonzero(target_mask)
    target_aspect = (target_x.max() - target_x.min() + 1) / max(
        target_y.max() - target_y.min() + 1, 1,
    )
    with torch.no_grad():
        target_features = model.forward_features(target)
        target_feature = F.normalize(target_features["x_norm_clstoken"], dim=-1)
        target_patches = F.normalize(target_features["x_norm_patchtokens"], dim=-1)
        patch_side = int(round(math.sqrt(target_patches.shape[1])))
        target_patch_mask = F.interpolate(
            torch.as_tensor(target_mask, dtype=torch.float32, device="cuda")[None, None],
            size=(patch_side, patch_side), mode="area",
        ).reshape(1, -1)
        target_patch_mask = (0.20 + 0.80 * target_patch_mask) / (
            0.20 + 0.80 * target_patch_mask
        ).sum(dim=1, keepdim=True)
        scores = []
        spatial_scores = []
        for start in range(0, len(frames), args.batch_size):
            batch = torch.stack([transform(image) for image in frames[start:start + args.batch_size]]).cuda()
            features = model.forward_features(batch)
            global_features = F.normalize(features["x_norm_clstoken"], dim=-1)
            patch_features = F.normalize(features["x_norm_patchtokens"], dim=-1)
            scores.append((global_features @ target_feature.T).squeeze(1).cpu())
            spatial_scores.append(
                ((patch_features * target_patches).sum(dim=-1) * target_patch_mask).sum(dim=1).cpu()
            )
    scores_np = torch.cat(scores).numpy()
    spatial_np = torch.cat(spatial_scores).numpy()
    joint_scores = []
    for index, (score, spatial_score, frame) in enumerate(zip(scores_np, spatial_np, frames)):
        frame_mask = np.any(np.asarray(frame, dtype=np.uint8) < 245, axis=2)
        frame_blur = cv2.GaussianBlur(np.asarray(frame, dtype=np.float32) / 255.0, (15, 15), 0)
        union_mask = np.logical_or(frame_mask, target_mask)
        rgb_layout_error = float(
            np.mean(np.abs(frame_blur - target_blur)[union_mask])
        )
        intersection = np.logical_and(frame_mask, target_mask).sum()
        union = np.logical_or(frame_mask, target_mask).sum()
        silhouette_iou = float(intersection / max(union, 1))
        frame_y, frame_x = np.nonzero(frame_mask)
        frame_aspect = (frame_x.max() - frame_x.min() + 1) / max(
            frame_y.max() - frame_y.min() + 1, 1,
        )
        aspect_error = float(abs(math.log(max(frame_aspect, 1e-6) / max(target_aspect, 1e-6))))
        joint = float(
            0.25 * score + 0.75 * spatial_score
            + 0.85 * silhouette_iou - 0.30 * min(aspect_error, 2.0)
            - 1.8 * rgb_layout_error
        )
        records[index].update({
            "dino_cosine": float(score),
            "dino_spatial_cosine": float(spatial_score),
            "silhouette_iou": silhouette_iou,
            "aspect_error": aspect_error,
            "rgb_layout_error": rgb_layout_error,
            "joint_score": joint,
        })
        joint_scores.append(joint)
    order = np.argsort(np.asarray(joint_scores))[::-1]
    selected = int(order[0])
    frames[selected].save(output / "selected_trellis_camera1_view.png")
    top_count = min(12, len(order))
    board = Image.new("RGB", (args.resolution * top_count, args.resolution), "white")
    for column, index in enumerate(order[:top_count]):
        board.paste(frames[int(index)], (column * args.resolution, 0))
    board.save(output / "top_trellis_camera1_views.png")
    result = {
        "method": "dino_discrete_camera1_canonical_view_selection",
        "ground_truth_used": False,
        "render_backend": render_backend,
        "mesh": str(args.mesh.resolve()),
        "colored_points": None if args.colored_points is None else str(args.colored_points.resolve()),
        "condition": str(args.condition.resolve()),
        "mesh_centre": centre.tolist(),
        "selected_index": selected,
        "selected": records[selected],
        "top": [{"index": int(index), **records[int(index)]} for index in order[:top_count]],
        "candidates": records,
    }
    (output / "trellis_camera1_view.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
