#!/usr/bin/env python3
"""Deform a registered textured prior using all cameras in a render manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multiview_proxy_deformation import OrbitProjector, deform_shared_multiview
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.trellis_multiview_probe import _load_world_mesh, apply_transform, estimate_ordered_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-size", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--anchor-count", type=int, default=6000)
    parser.add_argument("--maximum-local-log-scale", type=float, default=.35)
    parser.add_argument("--edited-view-dir", type=Path)
    parser.add_argument("--foreground-threshold", type=int, default=245)
    parser.add_argument("--silhouette-weight", type=float, default=.45)
    parser.add_argument("--max-silhouette-pixel-distance", type=float, default=36.0)
    parser.add_argument(
        "--coarse-pixel-distance-ratio", type=float, default=0.0,
        help="Initial long-range correspondence radius as a fraction of image size.",
    )
    parser.add_argument("--coarse-iterations", type=int, default=0)
    parser.add_argument("--coarse-anchor-ratio", type=float, default=.25)
    parser.add_argument("--coarse-smoothness-multiplier", type=float, default=4.0)
    parser.add_argument("--sample-count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=6830)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = load_points(Path(manifest["source_carrier"]))
    registered = load_points(Path(manifest["registered_carrier"]))
    partial = load_points(args.partial)
    transform, fit_rmse = estimate_ordered_similarity(source, registered)
    mesh = _load_world_mesh(Path(manifest["glb"]))
    mesh.vertices = apply_transform(mesh.vertices, transform)
    projectors = [
        OrbitProjector(
            np.asarray(view["camera_pose"], dtype=np.float64),
            float(manifest["field_of_view_degrees"]),
            (int(args.projection_size), int(args.projection_size)),
        )
        for view in manifest["views"]
    ]
    edited_silhouettes = None
    if args.edited_view_dir is not None:
        edited_silhouettes = []
        for view in manifest["views"]:
            path = args.edited_view_dir / f"{view['name']}.png"
            image = Image.open(path).convert("RGB").resize(
                (int(args.projection_size), int(args.projection_size)),
                Image.Resampling.LANCZOS,
            )
            edited_silhouettes.append(
                np.any(np.asarray(image, dtype=np.uint8) < int(args.foreground_threshold), axis=2)
            )
    deformed_carrier, deformed_vertices, status, diagnostics = deform_shared_multiview(
        registered, mesh.vertices, mesh.faces, partial, projectors,
        iterations=args.iterations, anchor_count=args.anchor_count,
        maximum_local_log_scale=args.maximum_local_log_scale,
        edited_silhouettes=edited_silhouettes,
        silhouette_weight=args.silhouette_weight,
        max_silhouette_pixel_distance=args.max_silhouette_pixel_distance,
        coarse_pixel_distance_ratio=args.coarse_pixel_distance_ratio,
        coarse_iterations=args.coarse_iterations,
        coarse_anchor_ratio=args.coarse_anchor_ratio,
        coarse_smoothness_multiplier=args.coarse_smoothness_multiplier,
    )
    mesh.vertices = deformed_vertices
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / "geometry_consistent_deformed_pixal.glb"
    mesh.export(mesh_path)
    carrier_path = args.output_dir / "geometry_consistent_deformed_carrier_100k.ply"
    write_points(carrier_path, deformed_carrier)
    write_compare(args.output_dir / "partial_gray_deformed_pixal_red.ply", partial, deformed_carrier)
    colours = np.zeros((len(deformed_carrier), 3), dtype=np.uint8)
    colours[:] = (70, 120, 255)       # unsupported / smoothly propagated
    colours[status == 1] = (255, 150, 30)  # supported residual
    colours[status == 2] = (40, 210, 80)   # supported stable
    status_cloud = trimesh.points.PointCloud(deformed_carrier, colors=colours)
    status_cloud.export(args.output_dir / "support_states_blue_orange_green.ply")
    np.random.seed(args.seed)
    sampled, _ = trimesh.sample.sample_surface(mesh, int(args.sample_count))
    write_points(args.output_dir / "geometry_consistent_deformed_mesh_100k.ply", sampled)
    record = {
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "partial": str(args.partial.resolve()),
            "glb": manifest["glb"],
            "source_carrier": manifest["source_carrier"],
            "registered_carrier": manifest["registered_carrier"],
            "edited_view_dir": (
                None if args.edited_view_dir is None else str(args.edited_view_dir.resolve())
            ),
        },
        "ordered_similarity_fit_rmse": fit_rmse,
        "outputs": {
            "mesh": str(mesh_path.resolve()),
            "carrier": str(carrier_path.resolve()),
        },
        "diagnostics": diagnostics,
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(jsonable(record), indent=2), encoding="utf-8",
    )
    print(json.dumps(jsonable(record), indent=2))


if __name__ == "__main__":
    main()
