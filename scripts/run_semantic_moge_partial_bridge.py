#!/usr/bin/env python3
"""Lift a Camera-1 semantic condition with MoGe and bridge it to a partial.

This is an isolated TRELLIS registration observation.  It reuses the saved
partial pixel indices and estimates one proper Sim(3) between the semantic
MoGe camera and the partial frame.  It never reads a complete target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pixal_native_moge_registration import native_pixal_moge_observation
from src.indexed_pixel_sim3 import robust_indexed_sim3
from src.moge_camera import MoGeProjector
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.ray_consistent_registration import apply_transform
from src.two_camera_moge_bridge import bbox_affine, transferred_partial_to_moge_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--source-mask", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--point-uv", type=Path, required=True)
    parser.add_argument("--moge-model", type=Path, required=True)
    parser.add_argument("--rmbg-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixel-distance", type=float, default=4.0)
    parser.add_argument(
        "--identity-affine", action="store_true",
        help="Use when semantic and partial depth already share the saved 512px frame.",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    partial = load_points(args.partial)
    point_uv = np.load(args.point_uv)
    if len(partial) != len(point_uv):
        raise ValueError("partial and point_uv must have the same point count")
    moge, colours, moge_info = native_pixal_moge_observation(
        args.semantic, args.moge_model, args.rmbg_model, output,
        device=args.device, fp16=True,
    )
    source_mask = cv2.imread(str(args.source_mask), cv2.IMREAD_GRAYSCALE)
    target_mask = cv2.imread(str(output / "pixal_input_object_mask.png"), cv2.IMREAD_GRAYSCALE)
    if source_mask is None or target_mask is None:
        raise FileNotFoundError("source or inferred semantic mask is missing")
    affine = (
        np.eye(3, dtype=np.float64)
        if args.identity_affine else bbox_affine(source_mask, target_mask)
    )
    projector = MoGeProjector(
        np.asarray(moge_info["output_keys"]["intrinsics"], dtype=np.float64),
        tuple(moge_info["image_hw"]), device=args.device,
    )
    pixels, _ = projector.project(moge)
    matches, match_info = transferred_partial_to_moge_matches(
        point_uv, source_mask.shape, affine, pixels,
        max_pixel_distance=float(args.max_pixel_distance), flip_y=True,
    )
    match_info.pop("mapped_partial_pixels", None)
    if len(matches) < 96:
        raise RuntimeError(f"Only {len(matches)} semantic-MoGe/partial matches")
    if len(matches) > 30_000:
        matches = matches[np.linspace(0, len(matches) - 1, 30_000, dtype=np.int64)]
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    transform, fit = robust_indexed_sim3(moge, partial, matches, diagonal=diagonal)
    aligned = apply_transform(moge, transform)
    write_points(output / "semantic_moge_camera_points.ply", moge)
    write_points(output / "semantic_moge_aligned_to_partial.ply", aligned)
    write_compare(output / "partial_gray_semantic_moge_blue.ply", partial, aligned)
    record = {
        "method": "camera1_pixel_indexed_semantic_moge_to_partial_sim3",
        "strict_zero_shot": True,
        "ground_truth_used": False,
        "semantic": str(args.semantic.resolve()),
        "partial": str(args.partial.resolve()),
        "moge": moge_info,
        "camera1_to_semantic_affine": affine,
        "matches": {**match_info, "fit_count": int(len(matches))},
        "fit": fit,
        "semantic_moge_to_partial": transform,
    }
    (output / "semantic_moge_bridge_info.json").write_text(
        json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(jsonable({
        "output": str(output), "moge_points": len(moge),
        "matches": len(matches), "fit": fit,
    }), indent=2))


if __name__ == "__main__":
    main()
