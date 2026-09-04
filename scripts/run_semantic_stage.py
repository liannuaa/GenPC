#!/usr/bin/env python3
"""Stage 1 of the mainline: partial scan -> depth -> Qwen semantic image."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
import yaml
from munch import Munch

ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DepthPrompting import DepthPrompting
from src.mainline_data import load_partial
from src.mainline_paths import redwood_partial_root, sample_file
from src.moge_pixel_bridge import run_rmbg_mask, save_mask_png


SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830", "07136", "07306", "09639")


def _load_config(path: Path, *, output_root: Path, partial_root: Path, models_root: Path) -> Munch:
    cfg = Munch.fromDict(yaml.safe_load(path.read_text(encoding="utf-8")))
    cfg.paths.output_dir = str(output_root.resolve())
    cfg.paths.data_dir = str(partial_root.resolve())
    cfg.paths.models_dir = str(models_root.resolve())
    return cfg


def _complete(cfg: Munch, sample: str) -> bool:
    return all(path.is_file() for path in (
        sample_file(cfg, sample, "depth.png"),
        sample_file(cfg, sample, "img.png"),
        sample_file(cfg, sample, f"{sample}_moge_to_raw_partial_object_mask.png"),
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "mainline_redwood.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, default=redwood_partial_root(ROOT))
    parser.add_argument("--models-root", type=Path, default=SHARED_ROOT / "models")
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=list(SAMPLES))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = _load_config(args.config, output_root=args.output_root, partial_root=args.partial_root, models_root=args.models_root)
    manifest_path = args.output_root / "semantic_stage_manifest.json"
    manifest = {
        "method": "saved_view_depth_then_qwen_semantic_completion",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "samples": {},
        "config": {
            "depth_projection": cfg.depth_projection,
            "camera_resolution": int(cfg.cam_res),
            "qwen_steps": int(cfg.qwen_edit_steps),
            "qwen_size": int(cfg.qwen_edit_generate_res),
        },
    }
    editor = None
    try:
        for sample in args.samples:
            partial = args.partial_root / f"{sample}.ply"
            if not partial.is_file():
                raise FileNotFoundError(partial)
            if args.resume and _complete(cfg, sample):
                manifest["samples"][sample] = {"state": "reused", "semantic": str(sample_file(cfg, sample, "img.png"))}
                continue
            if editor is None:
                editor = DepthPrompting(cfg)
            points, colors = load_partial(partial)
            xyz = torch.from_numpy(points).to(cfg.device)
            rgb = torch.from_numpy(colors).to(cfg.device)
            editor.getImage(xyz=xyz, flag=sample, rgb=rgb, depth_gen=True, img_gen=True)
            semantic = sample_file(cfg, sample, "img.png")
            alpha = run_rmbg_mask(
                semantic,
                sample_file(cfg, sample, "img_rmbg.png"),
                args.models_root / cfg.models.rmbg_model_path,
            )
            save_mask_png(sample_file(cfg, sample, f"{sample}_moge_to_raw_partial_object_mask.png"), alpha)
            manifest["samples"][sample] = {"state": "complete", "semantic": str(sample_file(cfg, sample, "img.png"))}
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
