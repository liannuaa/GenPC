#!/usr/bin/env python3
"""Render one textured mesh with the exact camera poses of a saved manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trellis_multiview_probe import _load_world_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registered-carrier", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=768)
    args = parser.parse_args()

    import pyrender

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mesh = _load_world_mesh(args.mesh)
    radius = max(float(np.linalg.norm(mesh.vertices - mesh.bounding_box.centroid, axis=1).max()), 1e-4)
    scene = pyrender.Scene(
        bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.array([.55, .55, .55, 1.], dtype=np.float32),
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    fov = math.radians(float(manifest["field_of_view_degrees"]))
    camera = pyrender.PerspectiveCamera(
        yfov=fov, aspectRatio=1.0, znear=max(radius * .01, 1e-4),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(args.resolution, args.resolution)
    outputs = []
    try:
        for view in manifest["views"]:
            pose = np.asarray(view["camera_pose"], dtype=np.float64)
            camera_node = scene.add(camera, pose=pose)
            light_node = scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=2.6), pose=pose,
            )
            colour, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            path = args.output_dir / f"deformed_{view['name']}.png"
            Image.fromarray(colour, mode="RGBA").convert("RGB").save(path)
            outputs.append({**view, "image": str(path.resolve())})
            scene.remove_node(camera_node)
            scene.remove_node(light_node)
    finally:
        renderer.delete()
    gutter = max(8, args.resolution // 32)
    board = Image.new(
        "RGB", (args.resolution * len(outputs) + gutter * (len(outputs) - 1), args.resolution), "white",
    )
    for index, view in enumerate(outputs):
        board.paste(Image.open(view["image"]).convert("RGB"), (index * (args.resolution + gutter), 0))
    board_path = args.output_dir / "deformed_front_side_back_right_board.png"
    board.save(board_path)
    output_manifest = {
        **manifest,
        "method": "exact_saved_camera_render_of_shared_3d_deformation",
        "glb": str(args.mesh.resolve()),
        "registered_carrier": (
            str(args.registered_carrier.resolve()) if args.registered_carrier else manifest.get("registered_carrier")
        ),
        "resolution": int(args.resolution),
        "views": outputs,
        "board": str(board_path.resolve()),
    }
    manifest_path = args.output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
