#!/usr/bin/env python3
"""Register one regenerated TRELLIS prior from its calibrated view contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline_metrics import evaluate_cd_emd
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from src.trellis_condition_registration import (
    draw_condition_projection_board,
    load_condition_contract,
    register_from_multiview_conditions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument(
        "--selection-json", type=Path,
        help="Optional RGB-selected TRELLIS Camera-1 basin for differentiable refinement.",
    )
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--back", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path,
                        help="Optional offline-only metric target; never enters registration.")
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--rotation-candidates", type=int, default=4096)
    parser.add_argument("--source-samples", type=int, default=8192)
    parser.add_argument("--render-resolution", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source, partial = load_points(args.prior), load_points(args.partial)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=0.15, image_shape=(512, 512), device="cpu",
    )
    conditions = {"front": args.front, "side": args.side, "back": args.back}
    result, info = register_from_multiview_conditions(
        source, partial, projector,
        manifest_path=args.render_manifest,
        condition_paths=conditions,
        selection_json=args.selection_json,
        semantic_path=args.semantic,
        seed=args.seed,
        rotation_candidates=args.rotation_candidates,
        source_samples=args.source_samples,
        render_resolution=args.render_resolution,
        device=args.device,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "trellis_condition_registered_100k.ply"
    write_points(prediction, result)
    write_compare(output / "partial_gray_trellis_red.ply", partial, result)
    draw_projection_overlay(
        output / "partial_camera1_projection.png", args.semantic, partial, result, projector,
    )
    cameras, manifest = load_condition_contract(
        args.render_manifest, conditions, resolution=args.render_resolution,
    )
    draw_condition_projection_board(
        output / "condition_multiview_projection.png", result, conditions, cameras,
        float(manifest["field_of_view_degrees"]),
    )
    info["inputs"] = {
        "prior": str(args.prior.resolve()),
        "partial": str(args.partial.resolve()),
        "camera": str(args.camera.resolve()),
        "semantic": str(args.semantic.resolve()),
        "render_manifest": str(args.render_manifest.resolve()),
        "conditions": {name: str(path.resolve()) for name, path in conditions.items()},
    }
    info["output"] = str(prediction)
    if args.ground_truth is not None:
        cd, emd = evaluate_cd_emd(prediction, args.ground_truth, count=16_384, seed=6145)
        info["offline_metrics_only"] = {
            "ground_truth": str(args.ground_truth.resolve()),
            "cd_l1_x100": float(cd * 100.0),
            "emd_x100": float(emd * 100.0),
        }
    (output / "trellis_condition_registration_info.json").write_text(
        json.dumps(jsonable(info), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(prediction),
        "final": info["final"],
        "offline_metrics_only": info.get("offline_metrics_only"),
    }, indent=2))


if __name__ == "__main__":
    main()
