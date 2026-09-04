#!/usr/bin/env python3
"""Prepare a neutral partial-cloud condition and coordinate audit for SPAR3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spar3d_agent_adapter import prepare_spar3d_condition


def load_points(path: Path) -> np.ndarray:
    import trimesh
    value = trimesh.load(path, process=False)
    if isinstance(value, trimesh.Scene):
        geometries = [geometry for geometry in value.geometry.values() if hasattr(geometry, "vertices")]
        return np.concatenate([np.asarray(geometry.vertices) for geometry in geometries], axis=0)
    return np.asarray(value.vertices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition, record = prepare_spar3d_condition(
        load_points(args.partial), max_points=args.max_points, seed=args.seed)
    np.save(args.output_dir / "partial_condition_xyzrgb.npy", condition)
    (args.output_dir / "coordinate_contract.json").write_text(
        json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"points": int(len(condition)), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
