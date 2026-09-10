#!/usr/bin/env python3
"""Run the isolated structure-aware partial ARAP Gaussian experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_camera_conditioned_gaussian_adaptation import save_camera_status_overlay
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.ray_consistent_registration import apply_transform
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from src.structure_aware_arap_gaussian import (
    moge_unsupported_partial_mask,
    structure_aware_arap_gaussian_adaptation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--locked-field", type=Path, default=None,
        help="Optional previous *_field.npz; its anchors and stable slots are frozen for residual continuation.",
    )
    parser.add_argument(
        "--minimum-component-points", type=int, default=64,
        help="Minimum support of one coherent residual component (default: 64).",
    )
    parser.add_argument("--bridge-moge", type=Path, default=None,
                        help="Optional native MoGe cloud used only to identify unsupported partial regions.")
    parser.add_argument("--bridge-moge-to-partial", type=Path, default=None,
                        help="Proper Sim(3) mapping --bridge-moge into the partial frame.")
    parser.add_argument("--bridge-unsupported-distance-ratio", type=float, default=.045)
    parser.add_argument("--bridge-unsupported-normal-agreement", type=float, default=.35)
    parser.add_argument("--bridge-unknown-normal-agreement", type=float, default=.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()
    prior, partial = load_points(args.prior), load_points(args.partial)
    locked_prior_mask = None
    if args.locked_field is not None:
        field = np.load(args.locked_field)
        required = {"anchors", "stable"}
        missing = required.difference(field.files)
        if missing:
            raise ValueError(f"locked field is missing masks: {sorted(missing)}")
        locked_prior_mask = np.asarray(field["anchors"], dtype=bool) | np.asarray(field["stable"], dtype=bool)
        if locked_prior_mask.shape != (len(prior),):
            raise ValueError("locked field and --prior have different carrier cardinalities")
    bridge_unknown_partial_mask, bridge_support = None, None
    if (args.bridge_moge is None) != (args.bridge_moge_to_partial is None):
        raise ValueError("--bridge-moge and --bridge-moge-to-partial must be supplied together")
    if args.bridge_moge is not None:
        transform = np.asarray(np.load(args.bridge_moge_to_partial), dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("--bridge-moge-to-partial must contain one 4x4 transform")
        bridge_moge = apply_transform(load_points(args.bridge_moge), transform)
        bridge_unknown_partial_mask, bridge_support = moge_unsupported_partial_mask(
            partial, bridge_moge,
            maximum_distance_ratio=float(args.bridge_unsupported_distance_ratio),
            minimum_normal_agreement=float(args.bridge_unsupported_normal_agreement),
        )
    projector = SavedCameraProjector.from_partial(partial, args.camera, padding=float(args.padding),
                                                  image_shape=(int(args.render_size), int(args.render_size)),
                                                  device=args.device)
    edited, estimate, masks = structure_aware_arap_gaussian_adaptation(
        prior, partial,
        locked_prior_mask=locked_prior_mask,
        bridge_unknown_partial_mask=bridge_unknown_partial_mask,
        bridge_unknown_normal_agreement=float(args.bridge_unknown_normal_agreement),
        minimum_component_points=int(args.minimum_component_points),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "structure_aware_arap_gaussian"
    write_points(Path(f"{stem}_prior_100k.ply"), edited)
    write_compare(Path(f"{stem}_partial_gray_prior_red.ply"), partial, edited)
    draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, edited, projector)
    status_masks = {"protected": ~masks["influence"], "moved": masks["moved"],
                    "locked": masks["stable"] | masks["boundary"], "editable": masks["anchors"]}
    status_path = Path(f"{stem}_status_projection.png")
    save_camera_status_overlay(status_path, args.semantic, prior, status_masks, projector)
    np.savez_compressed(Path(f"{stem}_field.npz"), means=edited.astype(np.float32), **masks)
    record = {"method": "structure_aware_partial_arap_gaussian_adaptation", "strict_zero_shot": True,
              "ground_truth_cd_emd_used": False,
              "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                         "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve())},
              "continuation": {
                  "locked_field": None if args.locked_field is None else str(args.locked_field.resolve()),
                  "minimum_component_points": int(args.minimum_component_points),
                  "inherited_locked_gaussians": 0 if locked_prior_mask is None else int(locked_prior_mask.sum()),
              },
              "bridge_unknown": bridge_support,
              "estimate": estimate,
              "outputs": {"prior": str(Path(f"{stem}_prior_100k.ply").resolve()),
                          "comparison": str(Path(f"{stem}_partial_gray_prior_red.ply").resolve()),
                          "saved_view": str(Path(f"{stem}_saved_view_projection.png").resolve()),
                          "status_view": str(status_path.resolve()),
                          "field": str(Path(f"{stem}_field.npz").resolve())}}
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"estimate": estimate, "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
