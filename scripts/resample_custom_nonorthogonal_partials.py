#!/usr/bin/env python3
"""Regenerate Redwood-style custom partial scans from fixed oblique GLB views.

The complete point clouds are left untouched.  Each partial is sampled from a
single visible mesh surface with a deterministic, non-axis-aligned pinhole
camera, so its depth image and camera metadata remain consistent with the
overwritten partial PLY.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import kaolin as kal
import numpy as np
import open3d as o3d
import torch
import trimesh


POINT_COUNT = 16_384
FOV_DEGREES = 50.0
FRAME_FILL = 0.82


@dataclass(frozen=True)
class SampleSpec:
    directory: str
    label: str
    azimuth_degrees: float
    elevation_degrees: float
    seed: int


# Deliberately oblique: neither azimuth nor elevation is an orthogonal view.
SPECS = (
    SampleSpec("handheld_power_drill", "handheld_power_drill", 68.0, -24.0, 614501),
    SampleSpec("mirrorless_camera", "mirrorless_camera", 31.0, 28.0, 614502),
    SampleSpec("single_engine_airplane", "single_engine_airplane", 239.0, 23.0, 614503),
    SampleSpec("dragon_character", "dragon_character", 129.0, -19.0, 614504),
    # Separates all three heads and necks while retaining a complete oblique body view.
    SampleSpec("hydra_creature", "hydra_creature", 289.0, 18.0, 614505),
    SampleSpec("domestic_pig", "domestic_pig", 197.0, -22.0, 614506),
    SampleSpec("light_helicopter", "light_helicopter", 311.0, 17.0, 614507),
    SampleSpec("cartoon_fox", "cartoon_fox", 73.0, -26.0, 614508),
    SampleSpec("tyrannosaurus_rex_skeleton", "tyrannosaurus_rex_skeleton", 163.0, 25.0, 614509),
    SampleSpec("wolf", "wolf", 277.0, -18.0, 614510),
)


def _normalised_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene")
    mesh = scene.to_geometry()
    extent = np.ptp(mesh.bounds, axis=0).max()
    if extent <= 0:
        raise ValueError(f"Degenerate mesh: {path}")
    mesh.vertices = (mesh.vertices - mesh.bounds.mean(axis=0)) / extent
    return mesh


def _camera_rotation(azimuth_degrees: float, elevation_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    eye_direction = np.array((
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.sin(azimuth),
    ), dtype=np.float64)
    forward = -eye_direction
    world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack((right, -up, forward), axis=0)
    return rotation, eye_direction


def _fit_camera_distance(vertices: np.ndarray, rotation: np.ndarray, focal: float, resolution: int) -> float:
    transformed = vertices @ rotation.T
    near = float(-transformed[:, 2].min() + 0.05)
    low = max(near, 0.1)
    high = max(low * 2.0, 1.0)
    target_span = FRAME_FILL * (resolution - 1)

    def span(distance: float) -> float:
        depth = transformed[:, 2] + distance
        pixels = focal * transformed[:, :2] / depth[:, None]
        return float(np.ptp(pixels, axis=0).max())

    while span(high) > target_span:
        high *= 2.0
    for _ in range(36):
        middle = (low + high) * 0.5
        if span(middle) > target_span:
            low = middle
        else:
            high = middle
    return high


def _raycast_partial(mesh: trimesh.Trimesh, spec: SampleSpec) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rotation, eye_direction = _camera_rotation(spec.azimuth_degrees, spec.elevation_degrees)
    for resolution in (1024, 1536, 2048):
        focal = 0.5 * (resolution - 1) / math.tan(math.radians(FOV_DEGREES) * 0.5)
        distance = _fit_camera_distance(mesh.vertices, rotation, focal, resolution)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = (rotation @ (-eye_direction * distance))
        intrinsic = np.array(((focal, 0.0, (resolution - 1) * 0.5),
                              (0.0, focal, (resolution - 1) * 0.5),
                              (0.0, 0.0, 1.0)), dtype=np.float64)
        o3d_mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(mesh.vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(mesh.faces, dtype=o3d.core.Dtype.Int32),
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d_mesh)
        rays = scene.create_rays_pinhole(intrinsic, extrinsic, resolution, resolution)
        hit_distance = scene.cast_rays(rays)["t_hit"].numpy()
        valid = np.isfinite(hit_distance)
        if int(valid.sum()) >= POINT_COUNT:
            ray_values = rays.numpy()
            hits = ray_values[..., :3][valid] + ray_values[..., 3:][valid] * hit_distance[valid, None]
            depth = np.zeros((resolution, resolution), dtype=np.float32)
            depth[valid] = (hits @ rotation[2] + extrinsic[2, 3]).astype(np.float32)
            camera = {
                "azimuth_degrees": spec.azimuth_degrees,
                "elevation_degrees": spec.elevation_degrees,
                "eye": (eye_direction * distance).tolist(),
                "target": [0.0, 0.0, 0.0],
                "fov_degrees": FOV_DEGREES,
                "frame_fill": FRAME_FILL,
                "camera_distance": distance,
                "image_size": resolution,
                "intrinsic": intrinsic.tolist(),
                "extrinsic_world_to_camera": extrinsic.tolist(),
                "visible_surface_points": int(valid.sum()),
                "sampler": "Open3D RaycastingScene pinhole",
                "projection_model": "pinhole",
                "point_uv_origin": "top-left",
            }
            return hits.astype(np.float64), depth, camera
    raise RuntimeError(f"{spec.label}: fewer than {POINT_COUNT} ray hits at 2048 px")


def _write_depth(path: Path, depth: np.ndarray) -> None:
    valid = depth > 0
    visual = np.zeros_like(depth, dtype=np.uint8)
    values = depth[valid]
    if len(values):
        normalized = (values - values.min()) / max(float(np.ptp(values)), 1e-8)
        visual[valid] = np.rint(25 + (1.0 - normalized) * 204).astype(np.uint8)
    if not cv2.imwrite(str(path), visual):
        raise IOError(f"Could not write {path}")


def _target_paths(root: Path, spec: SampleSpec) -> tuple[Path, Path, Path]:
    scan = root / spec.directory / "partial_data" / "single_scan"
    return scan / f"{spec.label}_partial.ply", scan / "depth.npy", scan / "depth.png"


def _update_sample_record(root: Path, glb_root: Path, spec: SampleSpec, camera: dict[str, object]) -> None:
    record_path = root / spec.directory / "sample.json"
    if not record_path.is_file():
        return
    record = json.loads(record_path.read_text())
    partial, depth_npy, depth_png = _target_paths(root, spec)
    gt = next((root / spec.directory / "gt_data").glob("*_gt.ply"))
    record["category"] = spec.label.replace("_", " ")
    record["source_glb"] = str((glb_root / f"{spec.label}.glb").resolve())
    record["camera"] = camera
    record["partial_points"] = POINT_COUNT
    record["resampled_partial"] = {
        "view": "fixed non-orthogonal pinhole",
        "seed": spec.seed,
        "partial": str(partial.resolve()),
        "depth_npy": str(depth_npy.resolve()),
        "depth_png": str(depth_png.resolve()),
    }
    outputs = record.get("outputs")
    if isinstance(outputs, dict):
        if "gt" in outputs:
            outputs["gt"] = str(gt.resolve())
            outputs["partial"] = str(partial.resolve())
            outputs["depth_npy"] = str(depth_npy.resolve())
            outputs["depth_png"] = str(depth_png.resolve())
            outputs["camera"] = str((partial.parent / "camera.json").resolve())
        if "gt_complete" in outputs:
            outputs["gt_complete"] = str(gt.resolve())
            outputs["source_partial"] = str(partial.resolve())
    record_path.write_text(json.dumps(record, indent=2) + "\n")


def _update_collection_manifests(root: Path, glb_root: Path,
                                 records: list[tuple[SampleSpec, dict[str, object]]]) -> None:
    by_directory = {spec.directory: (spec, camera) for spec, camera in records}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest.get("samples", []):
            match = by_directory.get(item.get("sample_id"))
            if match is None:
                continue
            spec, camera = match
            partial, depth_npy, depth_png = _target_paths(root, spec)
            gt = next((root / spec.directory / "gt_data").glob("*_gt.ply"))
            item["source_glb"] = str((glb_root / f"{spec.label}.glb").resolve())
            item["camera"] = camera
            item["partial_points"] = POINT_COUNT
            outputs = item.setdefault("outputs", {})
            outputs.update({"gt": str(gt.resolve()), "partial": str(partial.resolve()),
                            "depth_npy": str(depth_npy.resolve()), "depth_png": str(depth_png.resolve()),
                            "camera": str((partial.parent / "camera.json").resolve())})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    new_glb_manifest_path = root / "new_glb_redwood_20260903_manifest.json"
    if new_glb_manifest_path.is_file():
        manifest = json.loads(new_glb_manifest_path.read_text())
        for item in manifest.get("cases", []):
            match = by_directory.get(item.get("case_key"))
            if match is None:
                continue
            spec, camera = match
            partial, depth_npy, depth_png = _target_paths(root, spec)
            gt = next((root / spec.directory / "gt_data").glob("*_gt.ply"))
            label = spec.label.replace("_", " ")
            item["category"] = label
            item["category_source"] = f"renamed physical type: {spec.label}.glb"
            item["source_glb"] = str((glb_root / f"{spec.label}.glb").resolve())
            construction = item.setdefault("dataset_construction", {})
            construction.update({"partial": "single visible-surface scan via fixed non-orthogonal Open3D pinhole",
                                 "partial_points": POINT_COUNT, "sensor_camera": camera["eye"],
                                 "sensor_fov_deg": FOV_DEGREES, "sensor_resolution": camera["image_size"],
                                 "visible_surface_points": camera["visible_surface_points"]})
            outputs = item.setdefault("outputs", {})
            outputs["gt_complete"] = str(gt.resolve())
            outputs["source_partial"] = str(partial.resolve())
            outputs["depth"] = str(depth_png.resolve())
            outputs["camera"] = str((partial.parent / "camera.json").resolve())
        new_glb_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _write_outputs(root: Path, glb_root: Path, spec: SampleSpec, points: np.ndarray,
                   depth: np.ndarray, camera: dict[str, object]) -> None:
    partial, depth_npy, depth_png = _target_paths(root, spec)
    partial.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if not o3d.io.write_point_cloud(str(partial), cloud, write_ascii=False, compressed=False):
        raise IOError(f"Could not write {partial}")
    np.save(depth_npy, depth)
    _write_depth(depth_png, depth)
    (partial.parent / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")
    shutil.copy2(depth_png, partial.parent / "raw_depth.png")
    valid_mask = np.where(depth > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(partial.parent / "mask.png"), valid_mask):
        raise IOError(f"Could not write {partial.parent / 'mask.png'}")
    rotation = np.asarray(camera["extrinsic_world_to_camera"], dtype=np.float64)[:3, :3]
    translation = np.asarray(camera["extrinsic_world_to_camera"], dtype=np.float64)[:3, 3]
    intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
    camera_points = points @ rotation.T + translation
    uv = np.stack((
        (intrinsic[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsic[0, 2]) /
        (int(camera["image_size"]) - 1),
        (intrinsic[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsic[1, 2]) /
        (int(camera["image_size"]) - 1),
    ), axis=1).astype(np.float32)
    np.save(partial.parent / "point_uv.npy", uv)
    eye = np.asarray(camera["eye"], dtype=np.float32)
    np.save(partial.parent / "viewpoint.npy", eye)
    up = -rotation[1]
    saved_camera = kal.render.camera.Camera.from_args(
        eye=torch.as_tensor(eye),
        at=torch.zeros(3, dtype=torch.float32),
        up=torch.as_tensor(up, dtype=torch.float32),
        fov=math.radians(FOV_DEGREES),
        width=int(camera["image_size"]),
        height=int(camera["image_size"]),
        device="cpu",
    )
    torch.save(saved_camera, partial.parent / "camera.pth")
    _update_sample_record(root, glb_root, spec, camera)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/custom"))
    parser.add_argument("--glb-root", type=Path, default=Path("data/custom/glb"))
    parser.add_argument("--mirror-root", type=Path,
                        help="Optional second custom root that receives identical resampled partial assets.")
    parser.add_argument("--mirror-glb-root", type=Path,
                        help="GLB directory corresponding to --mirror-root (defaults to --mirror-root).")
    parser.add_argument("--samples", nargs="*", choices=[spec.label for spec in SPECS])
    args = parser.parse_args()
    selected = [spec for spec in SPECS if not args.samples or spec.label in args.samples]
    mirror_glb_root = args.mirror_glb_root or args.mirror_root
    generated: list[tuple[SampleSpec, dict[str, object]]] = []
    for spec in selected:
        mesh = _normalised_mesh(args.glb_root / f"{spec.label}.glb")
        hits, depth, camera = _raycast_partial(mesh, spec)
        indices = np.random.default_rng(spec.seed).choice(len(hits), POINT_COUNT, replace=False)
        points = hits[indices]
        _write_outputs(args.root, args.glb_root, spec, points, depth, camera)
        if args.mirror_root is not None:
            if mirror_glb_root is None:
                raise ValueError("--mirror-glb-root is required when --mirror-root is set")
            _write_outputs(args.mirror_root, mirror_glb_root, spec, points, depth, camera)
        generated.append((spec, camera))
        print(json.dumps({"sample": spec.label, "view": [spec.azimuth_degrees, spec.elevation_degrees],
                          "visible_points": len(hits), "partial_points": len(points)}, sort_keys=True))
    _update_collection_manifests(args.root, args.glb_root, generated)
    if args.mirror_root is not None:
        _update_collection_manifests(args.mirror_root, mirror_glb_root, generated)


if __name__ == "__main__":
    main()
