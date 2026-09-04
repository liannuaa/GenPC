#!/usr/bin/env python3
"""Export a uniform 100k point cloud from a partial-anchored Gaussian edit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from scripts.run_registration_deformation_fusion_ablation import SavedCameraProjector
from src.partial_anchored_gaussian_decode import decode_partial_anchored_gaussians
from src.multiview_partial_correspondence import (
    concatenate_positive_pairs,
    save_virtual_overlay_board,
    virtual_view_positive_pairs,
)
from src.visible_pixel_sim3_refinement import visible_pixel_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edited-prior", type=Path, required=True,
                        help="100k editable Pixal Gaussian means after 3DGS optimization")
    parser.add_argument("--view-reference", type=Path,
                        help="Optional frozen registered prior that fixes the virtual PCA camera frame.")
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixel-distance", type=float, default=2.0)
    parser.add_argument("--virtual-positive-views", type=int, default=6)
    parser.add_argument("--virtual-render-size", type=int, default=384)
    parser.add_argument("--virtual-max-pixel-distance", type=float, default=2.)
    parser.add_argument("--max-anchor-residual-ratio", type=float, default=.06)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_pixel_distance < 0. or args.max_anchor_residual_ratio <= 0.:
        raise ValueError("pixel distance must be non-negative and residual ratio positive")

    prior, partial = base.load_points(args.edited_prior), base.load_points(args.partial)
    view_reference = prior if args.view_reference is None else base.load_points(args.view_reference)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    saved_pairs, saved_pairing = visible_pixel_pairs(
        partial, prior, projector, max_pixel_distance=float(args.max_pixel_distance),
        max_pairs=min(len(partial), len(prior)),
    )
    virtual_pairs, virtual_pairing = virtual_view_positive_pairs(
        partial, prior, frame_points=view_reference, views=int(args.virtual_positive_views),
        resolution=int(args.virtual_render_size),
        max_pixel_distance=float(args.virtual_max_pixel_distance),
    )
    pairs = concatenate_positive_pairs(saved_pairs, virtual_pairs)
    pairing = {
        "saved_camera": saved_pairing, "virtual_positive_overlap": virtual_pairing,
        "combined_pairs": int(len(pairs)),
    }
    decoded, selected, decode = decode_partial_anchored_gaussians(
        prior, partial, pairs, max_residual=float(args.max_anchor_residual_ratio) * diagonal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "partial_anchored_gaussian"
    base.write_points(Path(f"{stem}_decoded_100k.ply"), decoded)
    base.write_points(Path(f"{stem}_edited_prior_100k.ply"), prior)
    base.write_compare(Path(f"{stem}_partial_gray_decoded_red.ply"), partial, decoded)
    base.draw_projection_overlay(
        Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, decoded, projector,
    )
    virtual_board = Path(f"{stem}_virtual_view_board.png")
    save_virtual_overlay_board(
        virtual_board, partial, decoded, frame_points=view_reference,
        views=int(args.virtual_positive_views),
    )
    np.save(Path(f"{stem}_anchor_pairs.npy"), selected)
    np.save(Path(f"{stem}_positive_pixel_pairs.npy"), pairs)
    record = {
        "method": "partial_anchored_dual_3dgs_collision_free_decode",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "fusion": "all editable Pixal Gaussian slots retained; one-to-one hard partial-anchor replacement",
        "inputs": {"edited_prior": str(args.edited_prior.resolve()), "partial": str(args.partial.resolve()),
                   "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve())},
        "pairing": pairing, "decoder": decode,
        "parameters": {"max_pixel_distance": float(args.max_pixel_distance),
                       "max_anchor_residual_ratio": float(args.max_anchor_residual_ratio),
                       "virtual_positive_views": int(args.virtual_positive_views),
                       "virtual_render_size": int(args.virtual_render_size),
                       "virtual_max_pixel_distance": float(args.virtual_max_pixel_distance),
                       "view_reference": None if args.view_reference is None else str(args.view_reference.resolve())},
        "outputs": {"prediction": str(Path(f"{stem}_decoded_100k.ply").resolve()),
                    "comparison": str(Path(f"{stem}_partial_gray_decoded_red.ply").resolve()),
                    "saved_view": str(Path(f"{stem}_saved_view_projection.png").resolve()),
                    "virtual_view_board": str(virtual_board.resolve()),
                    "anchor_pairs": str(Path(f"{stem}_anchor_pairs.npy").resolve()),
                    "positive_pairs": str(Path(f"{stem}_positive_pixel_pairs.npy").resolve())},
    }
    Path(f"{stem}_info.json").write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"pairing": pairing, "decoder": decode, "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
