#!/usr/bin/env python3
"""Write the current residual-derived image-edit instruction for one agent round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.saved_view_text_feedback import build_saved_view_text_feedback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--prompt-style", choices=("conservative", "explicit_local"),
                        default="conservative")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    partial, prior = base.load_points(args.partial), base.load_points(args.prior)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=args.padding, image_shape=(512, 512), device="cpu")
    feedback = build_saved_view_text_feedback(
        partial, prior, projector, diagonal=diagonal, prompt_style=args.prompt_style)
    (args.output_dir / "agent_instruction.txt").write_text(feedback.prompt + "\n", encoding="utf-8")
    (args.output_dir / "agent_instruction.json").write_text(
        json.dumps(feedback.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(feedback.to_dict(), indent=2))


if __name__ == "__main__":
    main()
