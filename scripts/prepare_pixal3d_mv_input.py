#!/usr/bin/env python3
"""Convert accepted FRONT/SIDE/BACK/RIGHT conditions to Pixal3D-MV input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pixal_multiview import (
    VIEWS,
    native_camera_distance_from_render_manifest,
    prepare_pixal_mv_views,
    read_render_fov,
    read_render_yaws,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-dir", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--framing-margin", type=float, default=1.1)
    parser.add_argument(
        "--camera-template-dir",
        type=Path,
        help="Directory containing pixal_registered_<view>.png camera templates.",
    )
    args = parser.parse_args()
    templates = None
    if args.camera_template_dir is not None:
        templates = {
            name: args.camera_template_dir / f"pixal_registered_{name}.png"
            for name in VIEWS
        }
    manifest = prepare_pixal_mv_views(
        {name: args.conditions_dir / f"{name}.png" for name in VIEWS},
        args.output_dir,
        field_of_view_degrees=read_render_fov(args.render_manifest),
        camera_distance=native_camera_distance_from_render_manifest(args.render_manifest),
        framing_margin=args.framing_margin,
        source_manifest=args.render_manifest,
        camera_templates=templates,
        orbit_yaws_degrees=read_render_yaws(args.render_manifest),
    )
    print(manifest.resolve())


if __name__ == "__main__":
    main()
