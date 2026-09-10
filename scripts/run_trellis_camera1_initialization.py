#!/usr/bin/env python3
"""Place a TRELLIS point carrier from a selected canonical Camera-1 view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from src.trellis_camera1_initialization import initialise_from_selected_camera1_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prior, partial = load_points(args.prior), load_points(args.partial)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=0.15, image_shape=(512, 512), device="cpu",
    )
    result, info = initialise_from_selected_camera1_view(
        prior, partial, projector, args.selection_json,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "trellis_camera1_initial_registered_100k.ply"
    write_points(prediction, result)
    write_compare(output / "partial_gray_trellis_red.ply", partial, result)
    draw_projection_overlay(
        output / "trellis_camera1_initial_saved_view.png",
        args.semantic, partial, result, projector,
    )
    info["output"] = str(prediction)
    (output / "trellis_camera1_initial_info.json").write_text(
        json.dumps(jsonable(info), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(prediction), "scale": info["scale"],
                      "cost": info["least_squares_cost"]}, indent=2))


if __name__ == "__main__":
    main()
