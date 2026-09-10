#!/usr/bin/env python3
"""Materialize one partial-supported coherent residual-registration candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.residual_component_registration import coherent_residual_component_translation
from src.saved_camera import SavedCameraProjector, world_to_camera_axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-support", type=Path,
                        help="Optional already-bridged soft observation in the partial coordinate frame.")
    parser.add_argument("--auxiliary-support-max-distance-ratio", type=float, default=.04)
    parser.add_argument("--max-pixel-distance", type=float, default=1.)
    parser.add_argument("--minimum-residual-ratio", type=float, default=.15)
    parser.add_argument("--screen-component-radius", type=float, default=3.)
    parser.add_argument("--minimum-component-pairs", type=int, default=128)
    parser.add_argument("--minimum-directional-coherence", type=float, default=.85)
    parser.add_argument("--minimum-camera-axis-dominance", type=float, default=.75)
    parser.add_argument("--maximum-translation-ratio", type=float, default=.30)
    parser.add_argument("--component-direction-cosine", type=float, default=.95)
    parser.add_argument("--graph-neighbours", type=int, default=8)
    parser.add_argument("--graph-edge-ratio", type=float, default=1.8)
    parser.add_argument("--stable-residual-ratio", type=float, default=.05,
                        help="Visible partial/prior matches below this diagonal-relative residual remain fixed.")
    parser.add_argument("--minimum-stable-pairs", type=int, default=512)
    parser.add_argument("--graph-screening", type=float, default=.0015)
    parser.add_argument("--component-geodesic-radius-ratio", type=float, default=.040,
                        help="Surface radius of the rigid residual component before supported continuation.")
    parser.add_argument("--maximum-component-fraction", type=float, default=.20)
    parser.add_argument("--influence-geodesic-radius-ratio", type=float, default=.10,
                        help="Maximum local surface radius influenced by a residual component.")
    parser.add_argument("--maximum-influence-fraction", type=float, default=.22,
                        help="Reject candidates whose continuation covers too much of the complete carrier.")
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prior, partial = load_points(args.prior), load_points(args.partial)
    auxiliary_support = load_points(args.auxiliary_support) if args.auxiliary_support else None
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=float(args.padding),
        image_shape=(int(args.render_size), int(args.render_size)), device=args.device,
    )
    displacement, info = coherent_residual_component_translation(
        prior, partial, projector, camera_axes=world_to_camera_axes(projector.camera),
        auxiliary_support=auxiliary_support,
        auxiliary_support_max_distance=(
            float(args.auxiliary_support_max_distance_ratio) * diagonal
            if auxiliary_support is not None else None
        ),
        max_pixel_distance=float(args.max_pixel_distance),
        minimum_residual_ratio=float(args.minimum_residual_ratio),
        screen_component_radius=float(args.screen_component_radius),
        minimum_component_pairs=int(args.minimum_component_pairs),
        minimum_directional_coherence=float(args.minimum_directional_coherence),
        minimum_camera_axis_dominance=float(args.minimum_camera_axis_dominance),
        maximum_translation_ratio=float(args.maximum_translation_ratio),
        component_direction_cosine=float(args.component_direction_cosine),
        graph_neighbours=int(args.graph_neighbours), graph_edge_ratio=float(args.graph_edge_ratio),
        stable_residual_ratio=float(args.stable_residual_ratio),
        minimum_stable_pairs=int(args.minimum_stable_pairs),
        graph_screening=float(args.graph_screening),
        component_geodesic_radius_ratio=float(args.component_geodesic_radius_ratio),
        maximum_component_fraction=float(args.maximum_component_fraction),
        influence_geodesic_radius_ratio=float(args.influence_geodesic_radius_ratio),
        maximum_influence_fraction=float(args.maximum_influence_fraction),
    )
    candidate = prior + displacement if bool(info.get("active", False)) else prior.copy()
    if candidate.shape != prior.shape or not np.isfinite(candidate).all():
        raise ValueError("residual-component registration produced an invalid carrier")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "component_registered_prior_100k.ply"
    write_points(output, candidate)
    write_compare(args.output_dir / "partial_gray_component_registration_red.ply", partial, candidate)
    record = {
        "method": "camera1_coherent_residual_component_registration",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "inputs": {"prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
                   "camera": str(args.camera.resolve())},
        "estimate": info, "output": str(output.resolve()),
        "carrier_slots_preserved": bool(len(candidate) == len(prior)),
    }
    (args.output_dir / "component_registration_info.json").write_text(
        json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve()), "estimate": jsonable(info)}, indent=2))


if __name__ == "__main__":
    main()
