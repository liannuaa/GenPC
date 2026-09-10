#!/usr/bin/env python3
"""Refine a registered complete prior with all partial support and Camera-1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline_metrics import evaluate_cd_emd
from src.partial_supported_sim3 import AllSupportSim3Config, capture_all_support_sim3
from src.pointcloud_io import jsonable, load_points, write_compare, write_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--camera1-condition", type=Path, required=True)
    parser.add_argument("--initial-transform", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rotation-degrees", type=float, default=22.0)
    parser.add_argument("--scale-ratio", type=float, default=1.14)
    parser.add_argument("--translation-ratio", type=float, default=0.18)
    parser.add_argument("--iterations", type=int, default=22)
    parser.add_argument("--population", type=int, default=7)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--silhouette-weight", type=float, default=0.0)
    parser.add_argument("--visible-weight", type=float, default=0.55)
    args = parser.parse_args()

    prior, partial = load_points(args.initial_prior), load_points(args.partial)
    projector = SavedCameraProjector.from_partial(
        partial, args.camera, padding=0.15, image_shape=(512, 512), device="cpu",
    )
    config = AllSupportSim3Config(
        rotation_degrees=args.rotation_degrees,
        log_scale=float(np.log(args.scale_ratio)),
        translation_ratio=args.translation_ratio,
        iterations=args.iterations,
        population=args.population,
        seed=args.seed,
        silhouette_weight=args.silhouette_weight,
        visible_weight=args.visible_weight,
    )
    registered, residual, info = capture_all_support_sim3(
        prior, partial, projector, args.camera1_condition, config=config,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "trellis_registered_100k.ply"
    write_points(prediction, registered)
    write_compare(output / "partial_gray_trellis_red.ply", partial, registered)
    draw_projection_overlay(
        output / "partial_camera1_projection.png",
        args.semantic, partial, registered, projector,
    )
    if args.initial_transform is not None:
        record = json.loads(args.initial_transform.read_text(encoding="utf-8"))
        initial = record.get("object_transform", record.get("transform"))
        if initial is None:
            raise ValueError("initial transform record has no object_transform or transform")
        info["composed_source_to_partial"] = (
            residual @ np.asarray(initial, dtype=np.float64)
        ).tolist()
    info["inputs"] = {
        "initial_prior": str(args.initial_prior.resolve()),
        "partial": str(args.partial.resolve()),
        "camera": str(args.camera.resolve()),
        "semantic": str(args.semantic.resolve()),
        "camera1_condition": str(args.camera1_condition.resolve()),
    }
    info["output"] = str(prediction)
    # Metrics are computed only after the no-GT transform is frozen.
    if args.ground_truth is not None:
        cd, emd = evaluate_cd_emd(prediction, args.ground_truth)
        info["offline_metrics_only"] = {
            "ground_truth": str(args.ground_truth.resolve()),
            "cd_l1_x100": float(cd * 100.0),
            "emd_x100": float(emd * 100.0),
        }
    (output / "registration_info.json").write_text(
        json.dumps(jsonable(info), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(jsonable({
        "output": str(prediction),
        "before": info["before"],
        "after": info["after"],
        "parameters": info["parameters"],
        "offline_metrics_only": info.get("offline_metrics_only"),
    }), indent=2))


if __name__ == "__main__":
    main()
