#!/usr/bin/env python3
"""Materialize one support-anchored axial Gaussian-deformation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attachment_aware_gaussian import support_anchored_axial_gaussian_deformation
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, world_to_camera_axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-support", type=Path)
    parser.add_argument("--auxiliary-support-max-distance-ratio", type=float, default=.04)
    parser.add_argument("--core-geodesic-radius-ratio", type=float, default=.04)
    parser.add_argument("--influence-geodesic-radius-ratio", type=float, default=.10)
    parser.add_argument("--maximum-influence-fraction", type=float, default=.30)
    parser.add_argument("--maximum-log-scale", type=float, default=.4054651081081644)
    parser.add_argument("--minimum-component-improvement", type=float, default=.25)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prior, partial = load_points(args.prior), load_points(args.partial)
    support = load_points(args.auxiliary_support) if args.auxiliary_support else None
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    displacement, estimate = support_anchored_axial_gaussian_deformation(
        prior, partial, projector, camera_axes=world_to_camera_axes(projector.camera),
        auxiliary_support=support,
        auxiliary_support_max_distance=(float(args.auxiliary_support_max_distance_ratio) * diagonal if support is not None else None),
        core_geodesic_radius_ratio=float(args.core_geodesic_radius_ratio),
        influence_geodesic_radius_ratio=float(args.influence_geodesic_radius_ratio),
        maximum_influence_fraction=float(args.maximum_influence_fraction),
        maximum_log_scale=float(args.maximum_log_scale),
        minimum_component_improvement=float(args.minimum_component_improvement),
    )
    candidate = prior + displacement if estimate.get("active", False) else prior.copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "attachment_aware_gaussian_means_100k.ply"
    write_points(output, candidate)
    write_compare(args.output_dir / "partial_gray_attachment_aware_gaussian_red.ply", partial, candidate)
    record = {"inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                         "camera": str(args.camera.resolve())},
              "estimate": estimate, "output": str(output.resolve())}
    (args.output_dir / "attachment_aware_gaussian_info.json").write_text(
        json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(jsonable({"output": str(output.resolve()), "estimate": estimate}), indent=2))


if __name__ == "__main__":
    main()
