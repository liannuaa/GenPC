#!/usr/bin/env python3
"""Bridge a regenerated complete prior to a registered reference and Camera-1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.complete_prior_bridge import bridge_regenerated_complete_prior
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-points", type=int, default=12_000)
    parser.add_argument("--visible-shortlist", type=int, default=8)
    parser.add_argument("--full-score-slack", type=float, default=1.15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source = load_points(args.source)
    reference = load_points(args.reference)
    partial = load_points(args.partial)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=0.15, image_shape=(512, 512), device=args.device,
    )
    result, info = bridge_regenerated_complete_prior(
        source, reference, partial, projector,
        seed=args.seed,
        sample_points=args.sample_points,
        visible_shortlist=args.visible_shortlist,
        full_score_slack=args.full_score_slack,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "trellis_complete_bridge_registered_100k.ply"
    write_points(result_path, result)
    write_compare(output_dir / "partial_gray_trellis_red.ply", partial, result)
    draw_projection_overlay(
        output_dir / "trellis_complete_bridge_saved_view.png",
        args.semantic, partial, result, projector,
    )
    info["inputs"] = {
        "source": str(args.source.resolve()),
        "reference": str(args.reference.resolve()),
        "partial": str(args.partial.resolve()),
        "camera": str(args.camera.resolve()),
        "semantic": str(args.semantic.resolve()),
    }
    info["output"] = str(result_path)
    (output_dir / "trellis_complete_bridge_info.json").write_text(
        json.dumps(jsonable(info), indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(result_path), "selected": info["selected_candidate"]}, indent=2))


if __name__ == "__main__":
    main()
