#!/usr/bin/env python3
"""Run an isolated topology-preserving mesh-attached Gaussian deformation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mesh_attached_gaussian_deformation import (
    deform_mesh_from_camera_depth,
    deform_mesh_from_camera_carrier_controls,
    deform_mesh_from_camera_projection,
    deform_registered_mesh,
    fit_carrier_affine,
    fit_carrier_sim3,
)
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, world_to_camera_axes


def _as_mesh(asset) -> trimesh.Trimesh:
    if isinstance(asset, trimesh.Scene):
        meshes = [geometry for geometry in asset.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if len(meshes) != 1:
            raise ValueError("the isolated prototype currently requires one textured mesh geometry")
        return meshes[0].copy()
    if isinstance(asset, trimesh.Trimesh):
        return asset.copy()
    raise ValueError("unable to load a triangular mesh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--source-carrier", type=Path, required=True)
    parser.add_argument("--registered-carrier", type=Path, required=True)
    parser.add_argument("--carrier-adaptation", type=Path,
                        help="optional slot-preserving affine-adapted carrier in the registered frame")
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-support", type=Path)
    parser.add_argument("--auxiliary-support-transform", type=Path,
                        help="optional 4x4 transform that maps auxiliary support into the partial frame")
    parser.add_argument("--auxiliary-support-max-distance-ratio", type=float, default=.04)
    parser.add_argument("--influence-geodesic-radius-ratio", type=float, default=.10)
    parser.add_argument("--maximum-influence-fraction", type=float, default=.30)
    parser.add_argument("--maximum-log-scale", type=float, default=.60)
    parser.add_argument("--local-mode", choices=("arap", "camera-depth", "camera-projection", "camera-carrier"), default="arap")
    parser.add_argument("--camera-depth-residual-ratio", type=float, default=.035)
    parser.add_argument("--camera-depth-stable-ratio", type=float, default=.0175)
    parser.add_argument("--camera-depth-minimum-component-pixels", type=int, default=32)
    parser.add_argument("--camera-depth-data-weight", type=float, default=.08)
    parser.add_argument("--camera-projection-max-pixel-distance", type=float, default=4.)
    parser.add_argument("--camera-projection-residual-ratio", type=float, default=.035)
    parser.add_argument("--camera-projection-stable-ratio", type=float, default=.0175)
    parser.add_argument("--camera-projection-data-weight", type=float, default=.02)
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-local-adaptation", action="store_true",
                        help="export only the global partial-supported mesh adaptation")
    args = parser.parse_args()
    source = load_points(args.source_carrier)
    registered = load_points(args.registered_carrier)
    partial = load_points(args.partial)
    support = load_points(args.auxiliary_support) if args.auxiliary_support else None
    support_transform = None
    if args.auxiliary_support_transform is not None:
        if support is None:
            raise ValueError("--auxiliary-support-transform requires --auxiliary-support")
        support_transform = np.asarray(np.load(args.auxiliary_support_transform), dtype=np.float64)
        if support_transform.shape != (4, 4):
            raise ValueError("auxiliary support transform must be a 4x4 matrix")
        support = support @ support_transform[:3, :3].T + support_transform[:3, 3]
    transform, transform_info = fit_carrier_sim3(source, registered)
    if transform_info["carrier_sim3_max_residual"] > 1e-6:
        raise ValueError("registered carrier no longer has slot-preserving global Sim(3) correspondence")
    mesh = _as_mesh(trimesh.load(args.mesh, force="scene", process=False))
    mesh.apply_transform(transform)
    affine_transform, affine_info = None, None
    if args.carrier_adaptation is not None:
        adapted = load_points(args.carrier_adaptation)
        affine_transform, affine_info = fit_carrier_affine(registered, adapted)
        if affine_info["carrier_affine_max_residual"] > 1e-6:
            raise ValueError("carrier adaptation is not slot-preserving affine correspondence")
        mesh.apply_transform(affine_transform)
        registered = adapted
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    if args.skip_local_adaptation:
        displacement = np.zeros_like(mesh.vertices, dtype=np.float64)
        estimate = {"active": False, "reason": "local_adaptation_explicitly_skipped"}
    elif args.local_mode == "arap":
        displacement, estimate = deform_registered_mesh(
            mesh.vertices, mesh.faces, registered, partial, projector,
            camera_axes=world_to_camera_axes(projector.camera), auxiliary_support=support,
            auxiliary_support_max_distance=(float(args.auxiliary_support_max_distance_ratio) * diagonal if support is not None else None),
            influence_geodesic_radius_ratio=float(args.influence_geodesic_radius_ratio),
            maximum_influence_fraction=float(args.maximum_influence_fraction),
            maximum_log_scale=float(args.maximum_log_scale),
        )
    elif args.local_mode == "camera-depth":
        displacement, estimate = deform_mesh_from_camera_depth(
            mesh.vertices, mesh.faces, partial, projector,
            camera_axes=world_to_camera_axes(projector.camera),
            depth_residual_ratio=float(args.camera_depth_residual_ratio),
            stable_depth_ratio=float(args.camera_depth_stable_ratio),
            minimum_component_pixels=int(args.camera_depth_minimum_component_pixels),
            depth_data_weight=float(args.camera_depth_data_weight),
            influence_geodesic_radius_ratio=float(args.influence_geodesic_radius_ratio),
            maximum_influence_fraction=float(args.maximum_influence_fraction),
        )
    elif args.local_mode == "camera-projection":
        displacement, estimate = deform_mesh_from_camera_projection(
            mesh.vertices, mesh.faces, partial, projector,
            camera_axes=world_to_camera_axes(projector.camera),
            max_pixel_distance=float(args.camera_projection_max_pixel_distance),
            projection_residual_ratio=float(args.camera_projection_residual_ratio),
            stable_residual_ratio=float(args.camera_projection_stable_ratio),
            minimum_component_pixels=int(args.camera_depth_minimum_component_pixels),
            projection_data_weight=float(args.camera_projection_data_weight),
            influence_geodesic_radius_ratio=float(args.influence_geodesic_radius_ratio),
            maximum_influence_fraction=float(args.maximum_influence_fraction),
        )
    else:
        displacement, estimate = deform_mesh_from_camera_carrier_controls(
            mesh.vertices, mesh.faces, registered, partial, projector,
            camera_axes=world_to_camera_axes(projector.camera),
            core_geodesic_radius_ratio=.035,
            influence_geodesic_radius_ratio=float(args.influence_geodesic_radius_ratio),
            maximum_influence_fraction=float(args.maximum_influence_fraction),
            projection_data_weight=float(args.camera_projection_data_weight),
        )
    if estimate.get("active", False):
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + displacement
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / "mesh_attached_deformed_prior.glb"
    mesh.export(mesh_path)
    np.random.seed(int(args.seed))
    sampled, _ = trimesh.sample.sample_surface(mesh, int(args.sample_count))
    points_path = args.output_dir / "mesh_attached_deformed_prior_100k.ply"
    write_points(points_path, sampled)
    write_compare(args.output_dir / "partial_gray_mesh_attached_prior_red.ply", partial, sampled)
    record = {
        "inputs": {"mesh": str(args.mesh.resolve()), "source_carrier": str(args.source_carrier.resolve()),
                   "registered_carrier": str(args.registered_carrier.resolve()), "partial": str(args.partial.resolve()),
                   "camera": str(args.camera.resolve()),
                   "carrier_adaptation": str(args.carrier_adaptation.resolve()) if args.carrier_adaptation else None,
                   "auxiliary_support": str(args.auxiliary_support.resolve()) if args.auxiliary_support else None,
                   "auxiliary_support_transform": (str(args.auxiliary_support_transform.resolve())
                                                   if args.auxiliary_support_transform else None)},
        "auxiliary_support_transform": support_transform.tolist() if support_transform is not None else None,
        "global_carrier_sim3": transform.tolist(), "global_carrier_sim3_info": transform_info,
        "carrier_adaptation_affine": affine_transform.tolist() if affine_transform is not None else None,
        "carrier_adaptation_affine_info": affine_info,
        "local_mode": str(args.local_mode),
        "camera_depth_parameters": {
            "residual_ratio": float(args.camera_depth_residual_ratio),
            "stable_ratio": float(args.camera_depth_stable_ratio),
            "minimum_component_pixels": int(args.camera_depth_minimum_component_pixels),
            "data_weight": float(args.camera_depth_data_weight),
        },
        "camera_projection_parameters": {
            "max_pixel_distance": float(args.camera_projection_max_pixel_distance),
            "residual_ratio": float(args.camera_projection_residual_ratio),
            "stable_ratio": float(args.camera_projection_stable_ratio),
            "data_weight": float(args.camera_projection_data_weight),
        },
        "estimate": estimate, "outputs": {"mesh": str(mesh_path.resolve()), "points": str(points_path.resolve())},
    }
    (args.output_dir / "mesh_attached_deformation_info.json").write_text(
        json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(jsonable(record), indent=2))


if __name__ == "__main__":
    main()
