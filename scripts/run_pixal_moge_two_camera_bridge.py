#!/usr/bin/env python3
"""Compose camera-anchored Pixal--MoGe with a Camera-1 partial bridge.

The script is an isolated registration diagnostic.  It produces no fusion and
does not inspect GT/CD/EMD.  Its only global motion comes from indexed
camera-1/camera-2 pixel correspondences, followed by composition with the
already validated analytic Pixal-to-native-MoGe transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexed_pixel_sim3 import robust_indexed_sim3, unique_pixel_matches
from src.moge_camera import MoGeProjector
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.bidirectional_cycle_registration import visible_score
from src.pixal_moge_analytic_registration import local_pixal_partial_refine, pixal_moge_render_score
from src.ray_consistent_registration import apply_transform
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from src.two_camera_moge_bridge import (
    affine_mask_iou,
    apply_image_affine,
    bbox_affine,
    conjugate_native_residual_to_partial,
    transferred_partial_to_moge_matches,
)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def _load_points(path: Path) -> np.ndarray:
    points = load_points(path)
    if len(points) < 96:
        raise ValueError(f"Expected at least 96 points in {path}, got {len(points)}")
    return points


def _subsample_matches(matches: np.ndarray, maximum: int) -> np.ndarray:
    if len(matches) <= int(maximum):
        return matches
    # Deterministic uniform spatial/image coverage without sample-specific
    # routing; RANSAC itself retains its fixed project seed.
    return matches[np.linspace(0, len(matches) - 1, int(maximum), dtype=np.int64)]


def _subsample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= int(maximum):
        return points
    return points[np.linspace(0, len(points) - 1, int(maximum), dtype=np.int64)]


def _compact_visible_score(score: dict) -> dict:
    """Keep scalar evidence only; ray correspondence arrays stay on disk-free TTO."""
    geometric, projection = score["geometric"], score["projection"]
    return {
        "objective": float(score["objective"]),
        "geometric_objective": float(geometric["objective"]),
        "pair_count": int(len(geometric["partial_ids"])),
        "projection": {key: float(value) for key, value in projection.items()},
    }


def _write_mask_bridge_debug(
    path: Path,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    affine: np.ndarray,
    source_pixels: np.ndarray,
) -> None:
    height, width = target_mask.shape
    source = cv2.warpAffine(
        (source_mask > 0).astype(np.uint8) * 255, affine[:2].astype(np.float32),
        (width, height), flags=cv2.INTER_NEAREST,
    ) > 0
    target = target_mask > 0
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[target] = (170, 170, 170)
    canvas[source] = (40, 40, 230)
    canvas[target & source] = (40, 200, 40)
    mapped = apply_image_affine(source_pixels, affine)
    # A sparse blue lattice makes origin flips/crop mistakes immediately visible.
    for point in mapped[::max(len(mapped) // 3000, 1)]:
        x, y = np.rint(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            canvas[y, x] = (230, 160, 40)
    cv2.imwrite(str(path), canvas)


def _write_two_cloud_compare(path: Path, partial: np.ndarray, moved: np.ndarray, color: tuple[int, int, int]) -> None:
    points = np.concatenate((partial, moved), axis=0)
    colors = np.concatenate((
        np.tile(np.array((145, 145, 145), dtype=np.uint8), (len(partial), 1)),
        np.tile(np.asarray(color, dtype=np.uint8), (len(moved), 1)),
    ))
    trimesh.points.PointCloud(points, colors=colors).export(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--point-uv", type=Path, required=True)
    parser.add_argument("--source-mask", type=Path, required=True,
                        help="Camera-1 semantic foreground mask, aligned with point_uv.")
    parser.add_argument("--target-mask", type=Path, required=True,
                        help="Camera-2 Pixal-input foreground mask.")
    parser.add_argument("--native-moge", type=Path, required=True)
    parser.add_argument("--native-moge-info", type=Path, required=True)
    parser.add_argument("--pixal-prior", type=Path, required=True)
    parser.add_argument("--partial-camera", type=Path, required=True)
    parser.add_argument("--saved-view-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixel-distance", type=float, default=3.0)
    parser.add_argument("--fit-max-matches", type=int, default=30_000)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True,
                        help="Run only a tightly bounded partial residual after the two-camera bridge.")
    parser.add_argument("--refine-points", type=int, default=32_000,
                        help="Deterministic Camera-1 partial subset for the local residual score.")
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    partial = _load_points(args.partial)
    prior = _load_points(args.pixal_prior)
    moge = _load_points(args.native_moge)
    point_uv = np.load(args.point_uv)
    if len(point_uv) != len(partial):
        raise ValueError("point_uv and partial must have identical point counts")
    source_mask, target_mask = _read_mask(args.source_mask), _read_mask(args.target_mask)
    camera1_to_camera2 = bbox_affine(source_mask, target_mask)
    native_info = json.loads(args.native_moge_info.read_text(encoding="utf-8"))
    moge_info = native_info["moge"]
    projector = MoGeProjector(
        np.asarray(moge_info["output_keys"]["intrinsics"], dtype=np.float64),
        tuple(moge_info["image_hw"]), device=args.device,
    )
    moge_pixels, _ = projector.project(moge)
    valid_moge = np.isfinite(moge_pixels).all(axis=1)
    if not valid_moge.all():
        # Output PLYs generated by the native stage contain valid camera rays;
        # fail loudly if this invariant changes rather than silently shifting IDs.
        raise ValueError("native MoGe PLY contains non-projectable points")
    matches, match_info = transferred_partial_to_moge_matches(
        point_uv, source_mask.shape, camera1_to_camera2, moge_pixels,
        max_pixel_distance=args.max_pixel_distance, flip_y=True,
    )
    # The dense mapped pixel lattice is used only for a PNG diagnostic below;
    # never serialize 96k coordinates into the compact experiment record.
    match_info.pop("mapped_partial_pixels", None)
    if len(matches) < 96:
        raise RuntimeError("two-camera bridge produced too few pixel correspondences")
    fit_matches = _subsample_matches(matches, args.fit_max_matches)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    moge_to_partial, fit = robust_indexed_sim3(
        moge, partial, fit_matches, diagonal=diagonal,
    )
    aligned_moge = apply_transform(moge, moge_to_partial)
    pixal_to_moge = np.asarray(native_info["prior_to_native_moge"], dtype=np.float64)
    pixal_to_partial = moge_to_partial @ pixal_to_moge
    registered_bridge = apply_transform(prior, pixal_to_partial)
    pairs = unique_pixel_matches(matches)
    residual = np.linalg.norm(
        aligned_moge[pairs[:, 1].astype(np.int64)] - partial[pairs[:, 0].astype(np.int64)], axis=1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "two_camera_pixal_moge"
    np.save(Path(f"{stem}_camera1_to_camera2.npy"), camera1_to_camera2)
    np.save(Path(f"{stem}_partial_to_native_moge_matches.npy"), matches)
    np.save(Path(f"{stem}_native_moge_to_partial.npy"), moge_to_partial)
    _write_two_cloud_compare(Path(f"{stem}_partial_gray_moge_blue.ply"), partial, aligned_moge, (45, 125, 230))
    source_pixels = point_uv.copy()
    source_pixels[:, 1] = 1. - source_pixels[:, 1]
    source_pixels[:, 0] *= max(source_mask.shape[1] - 1, 1)
    source_pixels[:, 1] *= max(source_mask.shape[0] - 1, 1)
    _write_mask_bridge_debug(Path(f"{stem}_camera1_to_camera2.png"), source_mask, target_mask,
                             camera1_to_camera2, source_pixels)
    saved_projector = SavedCameraProjector.from_partial(
        partial, args.partial_camera, padding=args.padding, image_shape=source_mask.shape, device="cpu"
    )
    prior_native_moge = apply_transform(prior, pixal_to_moge)
    native_before = pixal_moge_render_score(moge, prior_native_moge, projector)
    partial_before = visible_score(partial, registered_bridge, saved_projector, diagonal, pixel_radius=5.)
    registered, residual_partial = registered_bridge, np.eye(4, dtype=np.float64)
    refinement = {
        "enabled": bool(args.refine), "applied": False, "reason": "disabled",
        "partial_before": _compact_visible_score(partial_before),
        "native_before": native_before,
    }
    if args.refine:
        # This is a micro-correction only: use the Pixal--MoGe local
        # camera-coordinate descent, now with Camera-1 partial evidence added
        # as the primary term. There is no PCA/global partial registration.
        _, residual_native, local_search = local_pixal_partial_refine(
            prior_native_moge, moge, projector,
            _subsample_points(partial, args.refine_points), saved_projector,
            moge_to_partial, partial_diagonal=diagonal,
        )
        candidate_native = apply_transform(prior_native_moge, residual_native)
        candidate = apply_transform(candidate_native, moge_to_partial)
        partial_after = visible_score(partial, candidate, saved_projector, diagonal, pixel_radius=5.)
        native_after = pixal_moge_render_score(moge, candidate_native, projector)
        residual_partial = conjugate_native_residual_to_partial(residual_native, moge_to_partial)
        refinement = {
            "enabled": True, "applied": True,
            "reason": "fixed_two_camera_joint_objective",
            "parameters": {
                "style": "same local camera-frame coordinate descent as Pixal--MoGe",
                "score": "Camera-1 partial 2D+3D primary, Camera-2 Pixal--MoGe weighted auxiliary",
                "search_points": int(args.refine_points),
                "proposal_gate": False,
            },
            "partial_before": _compact_visible_score(partial_before),
            "partial_after": _compact_visible_score(partial_after),
            "native_before": native_before, "native_after": native_after,
            "local_search": local_search,
            "partial_residual": residual_partial,
            "native_residual": residual_native,
        }
        registered = candidate
        pixal_to_partial = moge_to_partial @ residual_native @ pixal_to_moge
    np.save(Path(f"{stem}_pixal_to_partial_bridge.npy"), moge_to_partial @ pixal_to_moge)
    np.save(Path(f"{stem}_pixal_to_partial.npy"), pixal_to_partial)
    write_points(Path(f"{stem}_bridge_registered_100k.ply"), registered_bridge)
    write_points(Path(f"{stem}_registered_100k.ply"), registered)
    write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, registered)
    draw_projection_overlay(
        Path(f"{stem}_saved_view_projection.png"), args.saved_view_image,
        partial, registered, saved_projector,
    )
    singular = np.linalg.svd(pixal_to_partial[:3, :3], compute_uv=False)
    record = {
        "method": "two_camera_pixel_bridge_plus_analytic_pixal_native_moge_sim3",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "fusion_run": False,
        "partial": str(args.partial.resolve()), "point_uv": str(args.point_uv.resolve()),
        "camera1": {"semantic_mask": str(args.source_mask.resolve()), "saved_camera": str(args.partial_camera.resolve())},
        "camera2": {"pixal_mask": str(args.target_mask.resolve()), "native_moge": str(args.native_moge.resolve()),
                    "pixal_native_info": str(args.native_moge_info.resolve()),
                    "moge_contract": "Pixal3D preprocessed pixal3d_input.png + MoGeModel.infer, matching Pixal camera estimation"},
        "camera1_to_camera2": {
            "transform": camera1_to_camera2, "foreground_iou": affine_mask_iou(source_mask, target_mask, camera1_to_camera2),
            "contract": "foreground bbox crop/resize only; no 3-D pose search",
        },
        "pixel_matches": {**match_info, "fit_matches": int(len(fit_matches))},
        "native_moge_to_partial": fit,
        "all_match_residual": {
            "mean": float(residual.mean()), "median": float(np.median(residual)),
            "p90": float(np.quantile(residual, .90)), "count": int(len(residual)),
        },
        "pixal_to_native_moge": pixal_to_moge,
        "pixal_to_partial": pixal_to_partial,
        "pixal_partial_micro_refinement": refinement,
        "transform_contract": {
            "proper_sim3": bool(np.linalg.det(pixal_to_partial[:3, :3]) > 0.),
            "isotropic_singular_values": singular,
            "all_pixal_points_preserved": bool(len(registered) == len(prior)),
            "nonrigid_deformation_used": False,
        },
        "next_step": "This output is registration-only; preserve the completed Pixal body for any later fusion/edit stage.",
    }
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({
        "output_dir": args.output_dir, "mask_iou": record["camera1_to_camera2"]["foreground_iou"],
        "matches": record["pixel_matches"]["matched_partial_pixels"], "fit": fit,
        "all_match_residual": record["all_match_residual"],
    }), indent=2))


if __name__ == "__main__":
    main()
