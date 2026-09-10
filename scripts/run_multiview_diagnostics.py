#!/usr/bin/env python3
"""Render four informative prior views and partial-supported residual evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multiview_diagnostics import (
    render_registered_multiview,
    render_registered_pointcloud_multiview,
)
from src.multiview_partial_evidence import build_partial_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path)
    parser.add_argument("--source-carrier", type=Path)
    parser.add_argument("--registered-prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-resolution", type=int, default=768)
    parser.add_argument("--evidence-resolution", type=int, default=512)
    parser.add_argument("--fov", type=float, default=38.0)
    parser.add_argument(
        "--render-only", action="store_true",
        help="Save selected views, masks, depths, and cameras without residual evidence.",
    )
    args = parser.parse_args()

    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    output = args.output_dir.resolve()
    if (args.glb is None) != (args.source_carrier is None):
        parser.error("--glb and --source-carrier must be provided together")
    if args.glb is not None:
        render = render_registered_multiview(
            glb_path=args.glb.resolve(),
            source_carrier_path=args.source_carrier.resolve(),
            registered_carrier_path=args.registered_prior.resolve(),
            camera_path=args.camera.resolve(),
            output_dir=output / "render",
            resolution=int(args.render_resolution),
            field_of_view_degrees=float(args.fov),
            informative_partial_path=args.partial.resolve(),
            num_views=4,
        )
    else:
        render = render_registered_pointcloud_multiview(
            registered_prior_path=args.registered_prior.resolve(),
            partial_path=args.partial.resolve(),
            camera_path=args.camera.resolve(),
            output_dir=output / "render",
            resolution=int(args.render_resolution),
            field_of_view_degrees=float(args.fov),
            num_views=4,
        )
    if args.render_only:
        record = {
            "method": "camera_consistent_four_view_observation",
            "ground_truth_used": False,
            "render_manifest": str((output / "render" / "render_manifest.json").resolve()),
            "covered_partial_points": render.get("view_selection", {}).get("covered_partial_points"),
        }
        (output / "diagnostic_manifest.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, indent=2))
        return
    evidence = build_partial_evidence(
        manifest_path=output / "render" / "render_manifest.json",
        registered_prior_path=args.registered_prior.resolve(),
        partial_path=args.partial.resolve(),
        output_dir=output / "evidence",
        resolution=int(args.evidence_resolution),
    )
    record = {
        "method": "camera_consistent_four_view_posterior_diagnostic",
        "ground_truth_used": False,
        "render_manifest": str((output / "render" / "render_manifest.json").resolve()),
        "evidence_manifest": str((output / "evidence" / "partial_evidence_manifest.json").resolve()),
        "covered_partial_points": render.get("view_selection", {}).get("covered_partial_points"),
        "paper_residual_board": evidence["paper_residual_board"],
    }
    (output / "diagnostic_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
