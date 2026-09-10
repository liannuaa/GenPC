#!/usr/bin/env python3
"""Capture a regenerated complete prior directly from Camera-1 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera1_rendered_sim3 import camera1_rendered_sim3_capture
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument(
        "--camera1-condition", type=Path,
        help="Complete white-background image rendered from the Camera-1 direction.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--rotation-candidates", type=int, default=4096)
    parser.add_argument("--source-samples", type=int, default=8192)
    parser.add_argument("--render-resolution", type=int, default=128)
    parser.add_argument("--shortlist", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prior, partial = load_points(args.prior), load_points(args.partial)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=0.15, image_shape=(512, 512), device="cpu",
    )
    result, info = camera1_rendered_sim3_capture(
        prior, partial, projector,
        camera1_condition=args.camera1_condition,
        seed=args.seed,
        rotation_candidates=args.rotation_candidates,
        source_samples=args.source_samples,
        render_resolution=args.render_resolution,
        shortlist=args.shortlist,
        device=args.device,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "camera1_rendered_sim3_registered_100k.ply"
    write_points(prediction, result)
    write_compare(output / "partial_gray_trellis_red.ply", partial, result)
    draw_projection_overlay(
        output / "camera1_rendered_sim3_saved_view.png",
        args.semantic, partial, result, projector,
    )
    info["inputs"] = {
        "prior": str(args.prior.resolve()),
        "partial": str(args.partial.resolve()),
        "camera": str(args.camera.resolve()),
        "semantic": str(args.semantic.resolve()),
        "camera1_condition": (
            None if args.camera1_condition is None else str(args.camera1_condition.resolve())
        ),
    }
    info["output"] = str(prediction)
    (output / "camera1_rendered_sim3_info.json").write_text(
        json.dumps(jsonable(info), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(prediction), "selected": info["selected"]}, indent=2))


if __name__ == "__main__":
    main()
