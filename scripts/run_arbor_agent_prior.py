#!/usr/bin/env python3
"""Run Arbor's text + explicit partial-hull agent action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"
ARBOUR_ROOT = MODEL_ROOT / "Arbor"
if str(ARBOUR_ROOT) not in sys.path:
    sys.path.insert(0, str(ARBOUR_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--hull", required=True, type=Path)
    parser.add_argument("--constraint-contract", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=MODEL_ROOT / "Arbor-weights")
    parser.add_argument("--slat-pipeline", type=Path, default=MODEL_ROOT / "TRELLIS-text-xlarge")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=100000)
    args = parser.parse_args()
    if not args.hull.exists() or not args.constraint_contract.exists():
        raise FileNotFoundError("Arbor requires a hull mesh and its coordinate contract")
    if not (args.weights / "denoiser_ema0.9999_step0053000.pt").exists():
        raise FileNotFoundError("Arbor control weights are incomplete")
    if not (args.slat_pipeline / "pipeline.json").exists():
        raise FileNotFoundError("Arbor requires a local TRELLIS text pipeline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from arbor import ArborPipeline, ConstraintSet

    pipeline = ArborPipeline(
        model_dir=args.weights, slat_pipeline=args.slat_pipeline, device="cuda", download=False)
    result = pipeline.run(
        prompt=args.prompt,
        constraints=ConstraintSet(hull=(args.hull.resolve(),)),
        output_dir=args.output_dir / "native",
        seed=args.seed,
        steps=args.steps,
    )
    if result.empty or result.mesh_path is None:
        raise RuntimeError(f"Arbor returned no mesh: {result.warnings}")
    mesh = trimesh.load(result.mesh_path, process=False)
    mesh.export(args.output_dir / "arbor_native.glb")
    points, _ = trimesh.sample.sample_surface(mesh, count=args.sample_count, seed=np.random.default_rng(args.seed))
    trimesh.points.PointCloud(points).export(args.output_dir / "arbor_native_sampled_100k.ply")
    contract = json.loads(args.constraint_contract.read_text())
    (args.output_dir / "arbor_agent_metadata.json").write_text(json.dumps({
        "backend": "Arbor", "strict_zero_shot": True,
        "prompt": args.prompt, "seed": args.seed, "steps": args.steps,
        "hull": str(args.hull.resolve()), "weights": str(args.weights.resolve()),
        "slat_pipeline": str(args.slat_pipeline.resolve()),
        "constraint_coordinate_contract": contract,
        "native_output_requires_global_proper_sim3": True,
        "nonrigid_deformation_used": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"mesh": str(args.output_dir / "arbor_native.glb"),
                      "points": str(args.output_dir / "arbor_native_sampled_100k.ply"),
                      "runtime_s": result.runtime_s}, indent=2))


if __name__ == "__main__":
    main()
