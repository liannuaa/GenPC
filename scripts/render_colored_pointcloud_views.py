#!/usr/bin/env python3
"""Render a coloured point cloud from three PCA-aligned orthographic views."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


def _normalise(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _camera_pose(centre: np.ndarray, backward: np.ndarray, up: np.ndarray, distance: float) -> np.ndarray:
    backward = _normalise(backward)
    right = _normalise(np.cross(up, backward))
    true_up = _normalise(np.cross(backward, right))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([right, true_up, backward], axis=1)
    pose[:3, 3] = centre + distance * backward
    return pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=180_000)
    parser.add_argument(
        "--caption", default="gray: partial    red: registered complete prior",
    )
    args = parser.parse_args()

    # Prefer the normal GLX/pyglet backend when a display is available (for
    # example under xvfb).  Headless workers without DISPLAY still use EGL.
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    cloud = trimesh.load(str(args.input), process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    colours = np.asarray(cloud.colors, dtype=np.uint8)
    if colours.ndim != 2 or colours.shape[1] < 3:
        colours = np.full((len(points), 4), 220, dtype=np.uint8)
    elif colours.shape[1] == 3:
        colours = np.concatenate([colours, np.full((len(colours), 1), 255, dtype=np.uint8)], axis=1)
    if len(points) > args.max_points:
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(points), size=args.max_points, replace=False)
        points, colours = points[chosen], colours[chosen]

    centre = np.median(points, axis=0)
    centred = points - centre
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    eigenvalues, basis = np.linalg.eigh(covariance)
    basis = basis[:, np.argsort(eigenvalues)[::-1]]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1

    views = [
        ("PCA 0-1", basis[:, 2], basis[:, 1], (0, 1)),
        ("PCA 0-2", basis[:, 1], basis[:, 2], (0, 2)),
        ("PCA 1-2", basis[:, 0], basis[:, 2], (1, 2)),
    ]
    scene = pyrender.Scene(
        bg_color=np.array([28, 28, 28, 255], dtype=np.uint8),
        ambient_light=np.ones(4, dtype=np.float32),
    )
    scene.add(pyrender.Mesh.from_points(points, colors=colours))
    renderer = pyrender.OffscreenRenderer(args.resolution, args.resolution, point_size=args.point_size)
    frames: list[Image.Image] = []
    try:
        for label, backward, up, plane in views:
            projected = centred @ basis[:, list(plane)]
            half_extent = max(float(np.ptp(projected[:, 0])), float(np.ptp(projected[:, 1]))) * 0.56
            camera = pyrender.OrthographicCamera(xmag=half_extent, ymag=half_extent)
            distance = max(float(np.linalg.norm(centred, axis=1).max()) * 2.0, 1.0)
            node = scene.add(camera, pose=_camera_pose(centre, backward, up, distance))
            colour, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            scene.remove_node(node)
            frame = Image.fromarray(colour[..., :3], mode="RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 122, 30), fill=(28, 28, 28))
            draw.text((10, 8), label, fill=(240, 240, 240))
            frames.append(frame)
    finally:
        renderer.delete()

    board = Image.new("RGB", (args.resolution * len(frames), args.resolution + 42), (28, 28, 28))
    for index, frame in enumerate(frames):
        board.paste(frame, (index * args.resolution, 0))
    draw = ImageDraw.Draw(board)
    draw.text((12, args.resolution + 14), args.caption, fill=(240, 240, 240))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
