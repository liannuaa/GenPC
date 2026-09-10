#!/usr/bin/env python3
"""Repair a zero-overlap Camera-1 proposal using rendered 2-D evidence only."""

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
from src.projection_sim3_rescue import center_scale_projection_rescue
from src.saved_camera import SavedCameraProjector, draw_projection_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--registered-prior", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    partial, prior = load_points(args.partial), load_points(args.registered_prior)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=.15, image_shape=(512, 512), device=args.device,
    )
    rescued, transform, record = center_scale_projection_rescue(partial, prior, projector)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "camera1_projection_rescue"
    output = Path(f"{stem}_registered_100k.ply")
    write_points(output, rescued)
    write_compare(Path(f"{stem}_partial_gray_pixal_red.ply"), partial, rescued)
    draw_projection_overlay(Path(f"{stem}_saved_view_projection.png"), args.semantic, partial, rescued, projector)
    np.save(Path(f"{stem}_residual.npy"), transform)
    Path(f"{stem}_info.json").write_text(json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **record["selected"]}, indent=2))


if __name__ == "__main__":
    main()
