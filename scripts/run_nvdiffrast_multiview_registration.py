#!/usr/bin/env python3
"""Refine a regenerated mesh with official nvdiffrast and PyTorch3D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nvdiffrast_multiview_registration import (
    RasterRefinementConfig,
    capture_shared_multiview_translation,
    initialise_from_condition_camera,
    initialise_from_multiview_condition_cameras,
    refine_joint_object_and_cameras,
    render_multiview_masks,
)
from src.bidirectional_cycle_registration import partial_to_prior_inverse_step
from src.offline_metrics import evaluate_cd_emd
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.ray_consistent_registration import apply_transform
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from src.trellis_condition_registration import (
    draw_condition_projection_board,
    load_condition_contract,
)


def _load_mesh(path: Path, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, process=False, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(faces) <= int(max_faces):
        return vertices, faces
    # Use Open3D's mature QEM implementation; face subsampling would create
    # holes and corrupt silhouette supervision.
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    simplified = o3d_mesh.simplify_quadric_decimation(int(max_faces))
    return (
        np.asarray(simplified.vertices, dtype=np.float64),
        np.asarray(simplified.triangles, dtype=np.int32),
    )


def _mesh_vertex_colours(prior_path: Path, mesh_vertices: np.ndarray) -> np.ndarray | None:
    """Transfer the coloured carrier to mesh vertices in the shared frame."""
    loaded = trimesh.load(prior_path, process=False)
    if not isinstance(loaded, trimesh.points.PointCloud):
        return None
    colours = np.asarray(loaded.colors)
    points = np.asarray(loaded.vertices, dtype=np.float64)
    if colours.ndim != 2 or colours.shape[1] < 3 or len(colours) != len(points):
        return None
    rgb = colours[:, :3].astype(np.float32) / 255.0
    if float(np.std(rgb)) < 1e-3:
        return None
    _, indices = cKDTree(points).query(np.asarray(mesh_vertices), k=1, workers=-1)
    return rgb[indices]


def _load_initial(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if "object_transform" in record:
        transform = record["object_transform"]
    elif "transform" in record:
        transform = record["transform"]
    else:
        transform = record["final"]["transform"]
    auxiliary = {
        name: np.asarray(vector, dtype=np.float64)
        for name, vector in record.get("auxiliary_view_rotation_vectors", {}).items()
    }
    return np.asarray(transform, dtype=np.float64), auxiliary


def _draw_mask_board(
    path: Path,
    cameras,
    rendered: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    panels, statistics = [], {}
    for camera in cameras:
        target = np.asarray(camera.mask, dtype=bool)
        predicted = np.asarray(rendered[camera.name], dtype=bool)
        intersection = target & predicted
        union = target | predicted
        panel = np.zeros((*target.shape, 3), dtype=np.uint8)
        panel[target] = np.array([0, 220, 255], dtype=np.uint8)
        panel[predicted] = np.array([255, 60, 60], dtype=np.uint8)
        panel[intersection] = 255
        iou = float(intersection.sum() / max(union.sum(), 1))
        coverage = float(intersection.sum() / max(target.sum(), 1))
        leakage = float((predicted & ~target).sum() / max(predicted.sum(), 1))
        cv2.putText(
            panel, f"{camera.name}: IoU {iou:.3f}", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 255, 170), 1, cv2.LINE_AA,
        )
        panels.append(panel)
        statistics[camera.name] = {
            "iou": iou, "coverage": coverage, "leakage": leakage,
        }
    board = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(board).save(path)
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--initial-info", type=Path)
    initial.add_argument(
        "--selection-json", type=Path, nargs="+",
        help="One front selection, or front/side/back selections in condition order.",
    )
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--back", type=Path, required=True)
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--semantic", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.018)
    parser.add_argument("--max-faces", type=int, default=100_000)
    parser.add_argument("--max-object-rotation", type=float, default=3.0)
    parser.add_argument("--max-scale-ratio", type=float, default=1.025)
    parser.add_argument("--max-translation-ratio", type=float, default=0.020)
    parser.add_argument("--max-auxiliary-rotation", type=float, default=3.0)
    parser.add_argument(
        "--reset-auxiliary-rotations", action="store_true",
        help=(
            "Start auxiliary condition cameras from the manifest instead of "
            "inheriting residuals from an earlier, possibly inconsistent basin."
        ),
    )
    parser.add_argument("--partial-weight", type=float, default=0.35)
    parser.add_argument(
        "--camera1-rgb-weight", type=float, default=0.0,
        help="Optional robust low-frequency Camera-1 texture/semantic alignment weight.",
    )
    parser.add_argument(
        "--partial-inverse-rounds", type=int, default=0,
        help=(
            "Optional Camera-1 partial-to-prior inverse Sim(3) rounds before "
            "raster refinement. This uses only the observed partial and its "
            "saved camera, and is disabled by default."
        ),
    )
    parser.add_argument("--partial-inverse-max-rotation", type=float, default=6.0)
    parser.add_argument("--partial-inverse-max-scale-ratio", type=float, default=1.08)
    parser.add_argument("--partial-inverse-max-translation-ratio", type=float, default=0.06)
    parser.add_argument("--centroid-translation-rounds", type=int, default=0)
    parser.add_argument("--centroid-translation-max-ratio", type=float, default=0.10)
    parser.add_argument("--view-consistency-scale-degrees", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    mesh_vertices, mesh_faces = _load_mesh(args.mesh, args.max_faces)
    source, partial = load_points(args.prior), load_points(args.partial)
    mesh_colours = _mesh_vertex_colours(args.prior, mesh_vertices)
    conditions = {"front": args.front, "side": args.side, "back": args.back}
    cameras, manifest = load_condition_contract(
        args.render_manifest, conditions, resolution=args.resolution,
    )
    if args.initial_info is not None:
        initial_transform, initial_auxiliary = _load_initial(args.initial_info)
        initial_record = {"method": "loaded_transform", "path": str(args.initial_info.resolve())}
    else:
        if len(args.selection_json) == 1:
            initial_transform, initial_record = initialise_from_condition_camera(
                source, args.selection_json[0], manifest,
            )
            initial_auxiliary = {}
        elif len(args.selection_json) == len(cameras):
            initial_transform, initial_record = initialise_from_multiview_condition_cameras(
                source,
                {camera.name: path for camera, path in zip(cameras, args.selection_json)},
                manifest,
                consistency_scale_degrees=args.view_consistency_scale_degrees,
            )
            initial_auxiliary = {
                name: np.asarray(vector, dtype=np.float64)
                for name, vector in initial_record[
                    "initial_auxiliary_rotation_vectors"
                ].items()
            }
        else:
            raise ValueError("--selection-json expects one path or one path per condition view")

    inverse_records = []
    if int(args.partial_inverse_rounds) > 0:
        if args.camera is None:
            raise ValueError("--camera is required with --partial-inverse-rounds")
        projector = SavedCameraProjector.from_partial(
            partial, args.camera, padding=0.15, image_shape=(512, 512), device="cpu",
        )
        diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
        current = apply_transform(source, initial_transform)
        for round_index in range(int(args.partial_inverse_rounds)):
            current, step, record = partial_to_prior_inverse_step(
                current,
                partial,
                projector,
                diagonal=diagonal,
                pixel_radius=8.0 if round_index == 0 else 5.0,
                max_rotation_deg=float(args.partial_inverse_max_rotation),
                scale_bounds=(
                    1.0 / float(args.partial_inverse_max_scale_ratio),
                    float(args.partial_inverse_max_scale_ratio),
                ),
                max_translation_ratio=float(args.partial_inverse_max_translation_ratio),
                fractions=(0.25, 0.5, 0.75, 1.0),
                return_best_candidate=True,
            )
            initial_transform = step @ initial_transform
            inverse_records.append({"round": round_index, **record})
            if np.allclose(step, np.eye(4), atol=1e-9):
                break
    if args.reset_auxiliary_rotations:
        initial_auxiliary = {camera.name: np.zeros(3) for camera in cameras}
    centroid_translation = None
    if int(args.centroid_translation_rounds) > 0:
        initial_transform, centroid_translation = capture_shared_multiview_translation(
            mesh_vertices,
            mesh_faces,
            initial_transform,
            cameras,
            float(manifest["field_of_view_degrees"]),
            initial_auxiliary,
            rounds=int(args.centroid_translation_rounds),
            max_translation_ratio=float(args.centroid_translation_max_ratio),
            device=args.device,
        )
    config = RasterRefinementConfig(
        resolution=args.resolution,
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_object_rotation_degrees=args.max_object_rotation,
        max_log_scale=math.log(args.max_scale_ratio),
        max_translation_ratio=args.max_translation_ratio,
        max_auxiliary_rotation_degrees=args.max_auxiliary_rotation,
        partial_weight=args.partial_weight,
        camera1_rgb_weight=args.camera1_rgb_weight,
    )
    result = refine_joint_object_and_cameras(
        mesh_vertices, mesh_faces, source, partial, initial_transform,
        cameras, float(manifest["field_of_view_degrees"]),
        config=config,
        initial_auxiliary_rotation_vectors=initial_auxiliary,
        mesh_vertex_colors=mesh_colours,
        device=args.device,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registered = apply_transform(source, result.transform)
    prediction = output / "trellis_registered_100k.ply"
    write_points(prediction, registered)
    write_compare(output / "partial_gray_trellis_red.ply", partial, registered)

    final_cameras, final_manifest = load_condition_contract(
        args.render_manifest, conditions, resolution=512,
    )
    rendered = render_multiview_masks(
        mesh_vertices, mesh_faces, result.transform, final_cameras,
        float(final_manifest["field_of_view_degrees"]),
        result.auxiliary_rotation_vectors, device=args.device,
    )
    hard_mask_statistics = _draw_mask_board(
        output / "condition_multiview_mask_alignment.png", final_cameras, rendered,
    )

    # The RGB board remains useful for semantic-part inspection; rotate the
    # points by each nuisance orbit only in this diagnostic view.
    pivot = np.median(registered, axis=0)
    from scipy.spatial.transform import Rotation
    calibrated_points = []
    for camera in final_cameras:
        orbit = Rotation.from_rotvec(
            result.auxiliary_rotation_vectors[camera.name]
        ).as_matrix()
        calibrated_points.append((registered - pivot) @ orbit.T + pivot)
    # draw_condition_projection_board expects one point set for every panel;
    # construct the panels independently and concatenate them.
    panel_paths = []
    for camera, points in zip(final_cameras, calibrated_points):
        panel_path = output / f".{camera.name}_rgb_overlay.png"
        draw_condition_projection_board(
            panel_path, points, {camera.name: conditions[camera.name]}, [camera],
            float(final_manifest["field_of_view_degrees"]),
        )
        panel_paths.append(panel_path)
    rgb_board = np.concatenate([
        np.asarray(Image.open(path).convert("RGB")) for path in panel_paths
    ], axis=1)
    Image.fromarray(rgb_board).save(output / "condition_multiview_projection.png")
    for path in panel_paths:
        path.unlink()

    if args.camera is not None and args.semantic is not None:
        projector = SavedCameraProjector.from_partial(
            partial, args.camera, padding=0.15, image_shape=(512, 512), device="cpu",
        )
        draw_projection_overlay(
            output / "partial_camera1_projection.png",
            args.semantic, partial, registered, projector,
        )

    info = {
        "method": "joint_shared_sim3_auxiliary_camera_orbit",
        "ground_truth_used_for_registration": False,
        "libraries": {
            "rasterizer": "nvdiffrast",
            "rotation_maps": "pytorch3d",
            "mesh_decimation": "open3d QEM",
        },
        "object_transform": result.transform.tolist(),
        "initialization": initial_record,
        "auxiliary_view_rotation_vectors": {
            name: vector.tolist()
            for name, vector in result.auxiliary_rotation_vectors.items()
        },
        "hard_mask_statistics_512": hard_mask_statistics,
        "optimization": result.diagnostics,
        "partial_inverse_initialization": inverse_records,
        "centroid_translation_initialization": centroid_translation,
        "camera1_rgb_alignment": {
            "requested_weight": float(args.camera1_rgb_weight),
            "coloured_mesh_available": mesh_colours is not None,
        },
        "inputs": {key: str(value.resolve()) for key, value in {
            "mesh": args.mesh, "prior": args.prior, "partial": args.partial,
            "render_manifest": args.render_manifest,
        }.items()},
    }
    if args.initial_info is not None:
        info["inputs"]["initial_info"] = str(args.initial_info.resolve())
    else:
        info["inputs"]["selection_json"] = [str(path.resolve()) for path in args.selection_json]
    if args.ground_truth is not None:
        cd, emd = evaluate_cd_emd(prediction, args.ground_truth, count=16_384, seed=6145)
        info["offline_metrics_only"] = {
            "ground_truth": str(args.ground_truth.resolve()),
            "cd_l1_x100": float(cd * 100.0),
            "emd_x100": float(emd * 100.0),
        }
    (output / "registration_info.json").write_text(
        json.dumps(jsonable(info), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(jsonable({
        "output": str(prediction),
        "hard_mask_statistics_512": hard_mask_statistics,
        "offline_metrics_only": info.get("offline_metrics_only"),
    }), indent=2))


if __name__ == "__main__":
    main()
