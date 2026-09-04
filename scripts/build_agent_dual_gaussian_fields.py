#!/usr/bin/env python3
"""Initialize a prior/editable + partial/locked dual Gaussian state."""

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

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.dual_gaussian_fields import make_dual_gaussian_field


def coloured_mesh_samples(path: Path, count: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    scene = trimesh.load(path, force="scene", process=False)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("registered prior mesh must contain faces")
    state = np.random.get_state(); np.random.seed(seed)
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.float64)
        colors = colors[np.asarray(mesh.faces)[face_ids]].mean(axis=1)[:, :3].astype(np.uint8)
    except Exception:
        colors = None
    return points, colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-mesh", type=Path)
    parser.add_argument("--prior-points", type=int, default=100000)
    parser.add_argument("--partial-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=6145)
    args = parser.parse_args()
    if args.prior_mesh:
        prior, colors = coloured_mesh_samples(args.prior_mesh, args.prior_points, args.seed)
        source = "registered_glb_surface"
    else:
        prior = base.subset(base.load_points(args.prior), args.prior_points)
        colors = None
        source = "registered_ply"
    partial = base.subset(base.load_points(args.partial), args.partial_points)
    field = make_dual_gaussian_field(prior, partial, prior_colors=colors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    field.to_npz(args.output_dir / "dual_gaussian_fields.npz")
    # Standard coloured PLY allows a Splatfacto-like backend to initialize from
    # exactly the same union; the NPZ retains the crucial lock/confidence tags.
    trimesh.points.PointCloud(field.means, colors=field.colors).export(args.output_dir / "dual_gaussian_init.ply")
    (args.output_dir / "dual_gaussian_contract.json").write_text(json.dumps({
        "strict_zero_shot": True,
        "coordinate_frame": "registered complete prior in partial method frame",
        "prior": str(args.prior.resolve()), "partial": str(args.partial.resolve()),
        "prior_mesh": str(args.prior_mesh.resolve()) if args.prior_mesh else None,
        "prior_source": source,
        "prior_gaussians": int(np.sum(field.prior_mask)),
        "partial_anchor_gaussians": int(np.sum(field.observed_anchor)),
        "partial_anchor_policy": "positions and local scales are immutable under edit; only prior Gaussians are transformed or edited",
        "registration_policy": "proper Sim(3) applies only to prior Gaussians, partial anchors stay in the observed frame",
        "decoding_policy": "sample accepted dual field then reuse common observation-conditioned surface decoding",
        "ground_truth_cd_emd_used": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "prior": int(np.sum(field.prior_mask)),
                      "partial_anchor": int(np.sum(field.observed_anchor)), "textured": colors is not None}, indent=2))


if __name__ == "__main__":
    main()
