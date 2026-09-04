#!/usr/bin/env python3
"""Propose one TRELLIS mesh+text prior variant from registered scan evidence.

This is deliberately only a proposal action.  It has no access to ground
truth, metrics, labels, replay assets, or the final decoder.  A downstream
agent loop must re-register and gate this output before it can replace its
input prior.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
# ``ROOT`` can itself be an isolated git worktree.  Model weights remain
# shared at the canonical GenPC repository root, not under ``.codex-worktrees``.
PROJECT_ROOT = ROOT.parents[1]
for path in (ROOT, PROJECT_ROOT / "third_party" / "LaS-Comp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.text_prior_feedback import build_text_edit_feedback


def _mesh_from_scene(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    meshes = [geometry for geometry in scene.geometry.values()
              if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)]
    if not meshes:
        raise ValueError(f"{path}: no triangular geometry")
    return trimesh.util.concatenate(meshes)


def _load_points(path: Path) -> np.ndarray:
    # The experiment PLYs are point-only binary files.  Open3D is the reader
    # already used by the registration pipeline and does not reinterpret them
    # as degenerate triangle meshes.
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"{path}: empty point cloud")
    return points


def _load_pipeline(model_path: Path):
    """Load only models used by ``run_variant`` to fit a 24 GB GPU."""
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    from trellis.pipelines import TrellisTextTo3DPipeline

    pipeline = TrellisTextTo3DPipeline.from_pretrained(str(model_path))
    # ``run_variant`` directly uses text conditioning, the SLAT flow, and the
    # mesh decoder.  It never samples sparse structure or decodes Gaussian/RF
    # outputs, so keep those CPU-side modules out of the GPU memory budget.
    for name in ("sparse_structure_decoder", "sparse_structure_flow_model",
                 "slat_decoder_gs", "slat_decoder_rf"):
        pipeline.models.pop(name, None)
    pipeline.cuda()
    return pipeline


def _trellis_mesh_to_trimesh(result, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    mesh = result["mesh"][0]
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    return trimesh.Trimesh(vertices=vertices * scale + center, faces=faces, process=False)


def _sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def process(args: argparse.Namespace) -> dict:
    mesh = _mesh_from_scene(args.mesh)
    prior = _load_points(args.prior)
    partial = _load_points(args.partial)
    feedback = build_text_edit_feedback(
        partial, prior, minimum_q90_ratio=args.minimum_q90_ratio,
        extent_change_ratio=args.extent_change_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "text_edit_prompt.txt").write_text(feedback.prompt + "\n", encoding="utf-8")
    record = {
        "method": "trellis_mesh_text_variant_proposal",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "input": {"registered_mesh": str(args.mesh), "registered_prior": str(args.prior),
                  "partial": str(args.partial)},
        "feedback": feedback.to_dict(),
        "shared_parameters": {"seed": args.seed, "slat_steps": args.slat_steps,
                              "slat_cfg_strength": args.slat_cfg_strength,
                              "minimum_q90_ratio": args.minimum_q90_ratio,
                              "extent_change_ratio": args.extent_change_ratio},
    }
    if args.dry_run or not feedback.eligible:
        record["proposed"] = False
        record["reason"] = "dry_run" if args.dry_run else "residual_below_shared_trigger"
        (args.output_dir / "text_variant_info.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    bounds = mesh.bounds
    center = .5 * (bounds[0] + bounds[1])
    scale = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"{args.mesh}: degenerate bounds")
    pipeline = _load_pipeline(args.model_path)
    try:
        import open3d as o3d

        base = o3d.geometry.TriangleMesh()
        base.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
        base.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
        result = pipeline.run_variant(
            base, feedback.prompt, seed=int(args.seed), formats=["mesh"],
            slat_sampler_params={"steps": int(args.slat_steps),
                                 "cfg_strength": float(args.slat_cfg_strength)},
        )
        proposal = _trellis_mesh_to_trimesh(result, center, scale)
    finally:
        del pipeline
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    mesh_path = args.output_dir / "text_variant.glb"
    point_path = args.output_dir / "text_variant_100k.ply"
    proposal.export(mesh_path)
    trimesh.PointCloud(_sample_mesh(proposal, args.sample_points, args.seed)).export(point_path)
    record.update({"proposed": True, "reason": "eligible_text_variant",
                   "normalization": {"registered_mesh_center": center.tolist(), "scale": scale},
                   "outputs": {"mesh": str(mesh_path), "points": str(point_path)}})
    (args.output_dir / "text_variant_info.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path,
                        default=PROJECT_ROOT / "models" / "TRELLIS-text-xlarge")
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument("--slat-cfg-strength", type=float, default=7.5)
    parser.add_argument("--sample-points", type=int, default=100000)
    parser.add_argument("--minimum-q90-ratio", type=float, default=.018)
    parser.add_argument("--extent-change-ratio", type=float, default=.035)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = process(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
