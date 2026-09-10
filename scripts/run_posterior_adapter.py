#!/usr/bin/env python3
"""Run the generic complete-prior posterior adapter on one registered case."""

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
from src.multiview_partial_correspondence import save_virtual_overlay_board
from src.posterior_adapter import PosteriorAdapter, PosteriorAdapterConfig
from src.saved_camera import SavedCameraProjector, draw_projection_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=.15)
    args = parser.parse_args()

    prior, partial = load_points(args.prior), load_points(args.partial)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    adapter = PosteriorAdapter(PosteriorAdapterConfig(), device=args.device)
    posterior, info, masks = adapter.run(prior, partial, projector)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_points(output / "posterior_prior_100k.ply", posterior)
    write_points(output / "coarse_prior_100k.ply", prior + masks["coarse_displacement"])
    write_compare(output / "partial_gray_posterior_red.ply", partial, posterior)
    draw_projection_overlay(
        output / "posterior_saved_view_projection.png", args.semantic,
        partial, posterior, projector,
    )
    save_virtual_overlay_board(
        output / "posterior_virtual_views.png", partial, posterior,
        frame_points=prior,
    )
    status_masks = {
        "locked": masks["supported_stable"],
        "editable": masks["supported_residual"],
        "protected": masks["unsupported"],
        "moved": masks["moved"],
    }
    save_camera_status_overlay(
        output / "posterior_support_status.png", args.semantic, prior, status_masks, projector,
    )
    pair_table = np.c_[
        masks["transport_partial_ids"], masks["transport_prior_ids"],
        masks["transport_weights"], masks["transport_costs"],
    ]
    np.save(output / "partial_ot_pairs.npy", pair_table)
    np.savez_compressed(
        output / "posterior_field.npz", means=posterior.astype(np.float32),
        **{key: value for key, value in masks.items() if not key.startswith("transport_")},
    )
    record = {
        "method": "structure_aware_partial_ot_complete_gaussian_posterior",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "category_or_part_rules_used": False,
        "inputs": {
            "prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
            "camera": str(args.camera.resolve()), "semantic": str(args.semantic.resolve()),
        },
        "estimate": info,
        "outputs": {
            "posterior": str((output / "posterior_prior_100k.ply").resolve()),
            "coarse": str((output / "coarse_prior_100k.ply").resolve()),
            "comparison": str((output / "partial_gray_posterior_red.ply").resolve()),
            "saved_view": str((output / "posterior_saved_view_projection.png").resolve()),
            "virtual_views": str((output / "posterior_virtual_views.png").resolve()),
            "support_status": str((output / "posterior_support_status.png").resolve()),
            "transport_pairs": str((output / "partial_ot_pairs.npy").resolve()),
            "field": str((output / "posterior_field.npz").resolve()),
        },
    }
    (output / "posterior_info.json").write_text(
        json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(jsonable({"output_dir": output.resolve(), "estimate": info}), indent=2))


if __name__ == "__main__":
    main()
