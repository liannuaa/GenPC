#!/usr/bin/env python3
"""Run SPAR3D as a point-aware image-to-3D agent action.

The semantic image is an agent-editable appearance/action channel; the raw
partial scan is a native SPAR3D geometry condition.  Its output is intentionally
left in its native frame and must subsequently pass the common global Sim(3)
registration and saved-view gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
SPAR3D_ROOT = PROJECT_ROOT / "models" / "SPAR3D"
if str(SPAR3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SPAR3D_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spar3d_agent_adapter import prepare_spar3d_condition


def load_points(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = [np.asarray(item.vertices) for item in loaded.geometry.values() if hasattr(item, "vertices")]
        return np.concatenate(parts, axis=0)
    return np.asarray(loaded.vertices)


def semantic_to_rgba(path: Path, threshold: int = 248) -> Image.Image:
    """Create an explicit alpha mask without re-cropping the saved camera view."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    alpha = (np.min(rgb, axis=-1) < threshold).astype(np.uint8) * 255
    if not np.any(alpha):
        raise ValueError(f"No foreground recovered from white-background semantic image: {path}")
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--partial", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-condition-points", type=int, default=2048)
    parser.add_argument("--alpha-threshold", type=int, default=248)
    parser.add_argument("--sample-count", type=int, default=100000)
    args = parser.parse_args()
    if not (args.weights / "config.yaml").exists() or not (args.weights / "model.safetensors").exists():
        raise FileNotFoundError("SPAR3D weights must contain config.yaml and model.safetensors")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Low-VRAM mode keeps the 24 GB GPU headroom needed by the surrounding
    # registration pipeline. It is an executor setting, not a sample rule.
    os.environ["SPAR3D_LOW_VRAM"] = "1"
    from spar3d.system import SPAR3D

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rgba = semantic_to_rgba(args.semantic, args.alpha_threshold)
    rgba.save(args.output_dir / "semantic_rgba.png")
    condition, contract = prepare_spar3d_condition(
        load_points(args.partial), max_points=args.max_condition_points, seed=args.seed)
    np.save(args.output_dir / "partial_condition_xyzrgb.npy", condition)

    model = SPAR3D.from_pretrained(
        str(args.weights.resolve()), config_name="config.yaml", weight_name="model.safetensors", low_vram_mode=True)
    model.to("cuda").eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mesh, _ = model.run_image(
            rgba, bake_resolution=512, pointcloud=condition, remesh="none", return_points=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("SPAR3D returned an empty mesh")
    mesh.export(args.output_dir / "spar3d_native.glb")
    points, _ = trimesh.sample.sample_surface(mesh, count=args.sample_count, seed=np.random.default_rng(args.seed))
    trimesh.points.PointCloud(points).export(args.output_dir / "spar3d_native_sampled_100k.ply")
    (args.output_dir / "spar3d_agent_metadata.json").write_text(json.dumps({
        "backend": "SPAR3D", "strict_zero_shot": True,
        "semantic": str(args.semantic.resolve()), "partial": str(args.partial.resolve()),
        "weights": str(args.weights.resolve()), "seed": args.seed,
        "alpha_threshold": args.alpha_threshold,
        "max_condition_points": args.max_condition_points,
        "coordinate_contract": contract.to_dict(),
        "native_output_requires_global_proper_sim3": True,
        "nonrigid_deformation_used": False,
        "texture_baker_available": bool(getattr(model, "baker", None) is not None),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"mesh": str(args.output_dir / "spar3d_native.glb"),
                      "points": str(args.output_dir / "spar3d_native_sampled_100k.ply")}, indent=2))


if __name__ == "__main__":
    main()
