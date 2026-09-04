#!/usr/bin/env python3
"""Fresh partial-to-depth-to-Qwen semantic generation for Agent GenPC+.

This stage has no dependency on any prior semantic image, Pixal mesh, registered
point cloud, or frozen completion.  It writes a standalone run root so later
agent actions can regenerate images and 3-D priors without contaminating the
frozen-anchor audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml
from munch import Munch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main as run_main
from utils.runtime import normalize_runtime_config


SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830",
           "07136", "07306", "09639")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run root: {run_root}")
    cfg = Munch.fromDict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    cfg.paths = getattr(cfg, "paths", Munch())
    cfg.paths.output_dir = str(run_root)
    cfg.paths.models_dir = str(args.models_dir.resolve())
    cfg.sample_ids = list(map(str, args.samples))
    cfg.run_stage1 = True
    cfg.run_stage2 = False
    cfg.run_metric = False
    cfg.pipeline_mode = "per_sample"
    cfg.skip_existing = False
    cfg.outputs.save_intermediates = True
    cfg.outputs.keep_profile = "debug"
    normalize_runtime_config(cfg)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "scratch_manifest.json").write_text(json.dumps({
        "method": "agent_genpc_plus_scratch_stage1",
        "strict_zero_shot": True,
        "ground_truth_used_before_metrics": False,
        "semantic_or_3d_replay_used": False,
        "samples": cfg.sample_ids,
        "models_dir": cfg.models_dir,
        "config": {"qwen_steps": int(cfg.qwen_edit_steps),
                   "qwen_size": int(cfg.qwen_edit_generate_res),
                   "camera_resolution": int(cfg.cam_res)},
    }, indent=2), encoding="utf-8")
    run_main(cfg)


if __name__ == "__main__":
    main()
