#!/usr/bin/env python3
"""Prepare a generic watertight partial-scan hull for Arbor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arbor_agent_constraints import partial_to_arbor_hull


def load_points(path: Path) -> np.ndarray:
    value = trimesh.load(path, process=False)
    if isinstance(value, trimesh.Scene):
        return np.concatenate([np.asarray(mesh.vertices) for mesh in value.geometry.values() if hasattr(mesh, "vertices")])
    return np.asarray(value.vertices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, contract = partial_to_arbor_hull(
        load_points(args.partial), max_points=args.max_points, seed=args.seed)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(args.output_dir / "partial_hull_constraint.ply")
    (args.output_dir / "coordinate_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"vertices": len(vertices), "faces": len(faces),
                      "radius": contract.support_radius}, indent=2))


if __name__ == "__main__":
    main()
