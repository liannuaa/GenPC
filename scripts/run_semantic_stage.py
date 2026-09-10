#!/usr/bin/env python3
"""Stage 1: saved-camera depth, with an optional local Qwen baseline."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from munch import Munch

ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.depth_prompting import DepthPrompting
from src.mainline_data import load_partial
from src.mainline_paths import REDWOOD10_SAMPLE_IDS, redwood_partial_root, sample_file
from src.moge_pixel_bridge import run_rmbg_mask, save_mask_png


def _load_config(path: Path, *, output_root: Path, partial_root: Path, models_root: Path) -> Munch:
    cfg = Munch.fromDict(yaml.safe_load(path.read_text(encoding="utf-8")))
    cfg.paths.output_dir = str(output_root.resolve())
    cfg.paths.data_dir = str(partial_root.resolve())
    cfg.paths.models_dir = str(models_root.resolve())
    return cfg


def _complete(cfg: Munch, sample: str, *, depth_only: bool) -> bool:
    names = ("depth.png", "raw_depth.png", "camera.pth", "point_uv.npy", "viewpoint.npy")
    if not depth_only:
        names += ("img.png", f"{sample}_moge_to_raw_partial_object_mask.png")
    return all(sample_file(cfg, sample, name).is_file() for name in names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "mainline_redwood.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, default=redwood_partial_root(ROOT))
    parser.add_argument("--models-root", type=Path, default=SHARED_ROOT / "models")
    parser.add_argument(
        "--viewpoint-root",
        type=Path,
        default=None,
        help=(
            "Optional directory with <sample>/viewpoint.npy from the camera that "
            "generated each partial scan. When supplied, semantic depth is rasterised "
            "from that partial-only camera rather than reselecting a view."
        ),
    )
    parser.add_argument("--samples", nargs="+", default=list(REDWOOD10_SAMPLE_IDS),
                        help="Sample identifiers; add prompt_overrides in the config for new objects.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--depth-only", action="store_true",
        help="Save the selected camera and depth observation without loading Qwen.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config, output_root=args.output_root, partial_root=args.partial_root, models_root=args.models_root)
    manifest_path = args.output_root / "semantic_stage_manifest.json"
    manifest = {
        "method": (
            "saved_view_depth_observation"
            if args.depth_only else "saved_view_depth_then_qwen_semantic_completion"
        ),
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "samples": {},
        "config": {
            "depth_projection": cfg.depth_projection,
            "camera_resolution": int(cfg.cam_res),
            "qwen_steps": None if args.depth_only else int(cfg.qwen_edit_steps),
            "qwen_size": None if args.depth_only else int(cfg.qwen_edit_generate_res),
        },
    }
    editor = None
    try:
        for sample in args.samples:
            partial = args.partial_root / f"{sample}.ply"
            if not partial.is_file():
                raise FileNotFoundError(partial)
            if args.resume and _complete(cfg, sample, depth_only=bool(args.depth_only)):
                manifest["samples"][sample] = {
                    "state": "reused",
                    "depth": str(sample_file(cfg, sample, "depth.png")),
                    "semantic": None if args.depth_only else str(sample_file(cfg, sample, "img.png")),
                }
                continue
            if editor is None:
                editor = DepthPrompting(cfg)
            points, colors = load_partial(partial)
            xyz = torch.from_numpy(points).to(cfg.device)
            rgb = torch.from_numpy(colors).to(cfg.device)
            viewpoint = None
            viewpoint_source = "saved_view_selection"
            if args.viewpoint_root is not None:
                viewpoint_path = args.viewpoint_root / sample / "viewpoint.npy"
                if not viewpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Missing partial-camera viewpoint for {sample}: {viewpoint_path}"
                    )
                viewpoint = np.load(viewpoint_path).astype(np.float32)
                viewpoint_source = str(viewpoint_path.resolve())
            if args.depth_only:
                editor.save_depth_observation(
                    xyz=xyz, flag=sample, rgb=rgb, viewpoint_override=viewpoint,
                )
                manifest["samples"][sample] = {
                    "state": "depth_complete",
                    "depth": str(sample_file(cfg, sample, "depth.png")),
                    "viewpoint_source": viewpoint_source,
                }
            else:
                editor.getImage(
                    xyz=xyz,
                    flag=sample,
                    rgb=rgb,
                    depth_gen=True,
                    img_gen=True,
                    viewpoint_override=viewpoint,
                )
                semantic = sample_file(cfg, sample, "img.png")
                alpha = run_rmbg_mask(
                    semantic,
                    sample_file(cfg, sample, "img_rmbg.png"),
                    args.models_root / cfg.models.rmbg_model_path,
                )
                save_mask_png(sample_file(cfg, sample, f"{sample}_moge_to_raw_partial_object_mask.png"), alpha)
                manifest["samples"][sample] = {
                    "state": "complete",
                    "semantic": str(semantic),
                    "viewpoint_source": viewpoint_source,
                }
            del xyz, rgb, points, colors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        if editor is not None:
            editor.close()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
