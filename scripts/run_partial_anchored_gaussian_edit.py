#!/usr/bin/env python3
"""Apply a compact partial-conditioned Gaussian edit without GT selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dual_gaussian_fields import make_dual_gaussian_field
from src.multiview_partial_correspondence import (
    concatenate_positive_pairs,
    save_virtual_overlay_board,
    virtual_view_positive_pairs,
)
from src.partial_anchored_gaussian_edit import (
    boundary_conditioned_graph_displacement,
    compact_anchor_displacement,
)
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from src.visible_pixel_sim3_refinement import visible_pixel_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dual-field", type=Path,
                        help="Optional existing dual-Gaussian NPZ. If omitted, initialize the field from prior + partial.")
    parser.add_argument("--max-pixel-distance", type=float, default=1.0)
    parser.add_argument("--virtual-positive-views", type=int, default=6)
    parser.add_argument("--virtual-render-size", type=int, default=384)
    parser.add_argument("--virtual-max-pixel-distance", type=float, default=1.0)
    parser.add_argument("--max-anchor-residual-ratio", type=float, default=.06)
    parser.add_argument("--edit-method", choices=("graph", "compact"), default="graph")
    parser.add_argument("--support-radius-ratio", type=float, default=.08)
    parser.add_argument("--max-displacement-ratio", type=float, default=.06)
    parser.add_argument("--neighbours", type=int, default=8)
    parser.add_argument("--graph-edge-ratio", type=float, default=1.8)
    parser.add_argument("--graph-screening", type=float, default=.003)
    parser.add_argument("--prior-protection-views", type=int, default=6)
    parser.add_argument("--prior-protection-weight", type=float, default=.02)
    parser.add_argument("--protection-exclusion-ratio", type=float, default=.08)
    parser.add_argument("--remote-gain", type=float, default=1.)
    parser.add_argument("--remote-gain-radius-ratio", type=float, default=.08)
    parser.add_argument("--remote-displacement-cap-multiplier", type=float, default=1.)
    parser.add_argument("--graph-cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--graph-cg-max-iterations", type=int, default=240)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.max_anchor_residual_ratio, args.support_radius_ratio, args.max_displacement_ratio) <= 0.:
        raise ValueError("all shared metric ratios must be positive")
    prior, partial = load_points(args.prior), load_points(args.partial)
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
        partial, prior, views=int(args.virtual_positive_views),
        resolution=int(args.virtual_render_size),
        max_pixel_distance=float(args.virtual_max_pixel_distance),
    )
    pairs = concatenate_positive_pairs(saved_pairs, virtual_pairs)
    pairing = {
        "saved_camera": saved_pairing, "virtual_positive_overlap": virtual_pairing,
        "combined_pairs": int(len(pairs)),
    }
    method_args = {
        "max_anchor_residual": float(args.max_anchor_residual_ratio) * diagonal,
        "max_displacement": float(args.max_displacement_ratio) * diagonal,
        "neighbours": int(args.neighbours),
    }
    if args.edit_method == "graph":
        edited, anchors, edit = boundary_conditioned_graph_displacement(
            prior, partial, pairs,
            edge_ratio=float(args.graph_edge_ratio), screening=float(args.graph_screening),
            prior_protection_views=int(args.prior_protection_views),
            prior_protection_weight=float(args.prior_protection_weight),
            protection_exclusion_ratio=float(args.protection_exclusion_ratio),
            remote_gain=float(args.remote_gain),
            remote_gain_radius_ratio=float(args.remote_gain_radius_ratio),
            remote_displacement_cap_multiplier=float(args.remote_displacement_cap_multiplier),
            cg_tolerance=float(args.graph_cg_tolerance),
            cg_max_iterations=int(args.graph_cg_max_iterations), **method_args,
        )
    else:
        edited, anchors, edit = compact_anchor_displacement(
            prior, partial, pairs,
            support_radius=float(args.support_radius_ratio) * diagonal, **method_args,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "partial_anchored_gaussian_edit"
    write_points(Path(f"{stem}_editable_prior_100k.ply"), edited)
    write_compare(Path(f"{stem}_partial_gray_prior_red.ply"), partial, edited)
    draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, edited, projector)
    virtual_board = Path(f"{stem}_virtual_view_board.png")
    save_virtual_overlay_board(virtual_board, partial, edited, views=int(args.virtual_positive_views))
    np.save(Path(f"{stem}_anchor_pairs.npy"), anchors)
    np.save(Path(f"{stem}_positive_pixel_pairs.npy"), pairs)
    field_output = Path(f"{stem}_field.npz")
    if args.dual_field is not None:
        state = np.load(args.dual_field)
        payload = {name: state[name] for name in state.files}
        anchors_mask = payload.get("observed_anchor")
        if anchors_mask is None or int((~anchors_mask.astype(bool)).sum()) != len(edited):
            raise ValueError("--dual-field editable population does not match --prior")
        payload["means"] = payload["means"].copy()
        payload["means"][~anchors_mask.astype(bool)] = edited.astype(np.float32)
    else:
        field = make_dual_gaussian_field(edited, partial)
        payload = {
            "means": field.means, "log_scales": field.log_scales,
            "colors": field.colors, "opacity_logits": field.opacity_logits,
            "observed_anchor": field.observed_anchor, "confidence": field.confidence,
        }
    np.savez_compressed(field_output, **payload)
    record = {
        "method": f"partial_anchored_{args.edit_method}_gaussian_mean_edit",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "fusion": "dual Gaussian populations; partial anchors immutable and prior means compactly editable",
        "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                   "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve())},
        "pairing": pairing,
        "edit": edit,
        "parameters": {"max_pixel_distance": float(args.max_pixel_distance),
                       "virtual_positive_views": int(args.virtual_positive_views),
                       "virtual_render_size": int(args.virtual_render_size),
                       "virtual_max_pixel_distance": float(args.virtual_max_pixel_distance),
                       "max_anchor_residual_ratio": float(args.max_anchor_residual_ratio),
                       "support_radius_ratio": float(args.support_radius_ratio),
                       "max_displacement_ratio": float(args.max_displacement_ratio),
                       "neighbours": int(args.neighbours), "edit_method": args.edit_method,
                       "graph_edge_ratio": float(args.graph_edge_ratio),
                       "graph_screening": float(args.graph_screening),
                       "prior_protection_views": int(args.prior_protection_views),
                       "prior_protection_weight": float(args.prior_protection_weight),
                       "protection_exclusion_ratio": float(args.protection_exclusion_ratio),
                       "remote_gain": float(args.remote_gain),
                       "remote_gain_radius_ratio": float(args.remote_gain_radius_ratio),
                       "remote_displacement_cap_multiplier": float(args.remote_displacement_cap_multiplier),
                       "graph_cg_tolerance": float(args.graph_cg_tolerance),
                       "graph_cg_max_iterations": int(args.graph_cg_max_iterations)},
        "outputs": {"editable_prior": str(Path(f"{stem}_editable_prior_100k.ply").resolve()),
                    "comparison": str(Path(f"{stem}_partial_gray_prior_red.ply").resolve()),
                    "saved_view": str(Path(f"{stem}_saved_view_projection.png").resolve()),
                    "virtual_view_board": str(virtual_board.resolve()),
                    "positive_pairs": str(Path(f"{stem}_positive_pixel_pairs.npy").resolve()),
                    "field": str(field_output.resolve())},
    }
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"pairing": pairing, "edit": edit, "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
