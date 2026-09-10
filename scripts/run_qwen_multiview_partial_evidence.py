#!/usr/bin/env python3
"""Jointly edit four fixed-camera views using positive partial evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multiview_partial_evidence import (
    build_partial_evidence,
    build_prompt,
    split_grid,
)
from tools.qwen_image_edit import QwenImageEdit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registered-prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--object-type", required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=6830)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    evidence = build_partial_evidence(
        manifest_path=args.manifest.resolve(),
        registered_prior_path=args.registered_prior.resolve(),
        partial_path=args.partial.resolve(),
        output_dir=output / "evidence",
        resolution=args.resolution,
    )
    prompt = build_prompt(args.object_type)
    prompt_path = output / "qwen_four_view_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    editor = QwenImageEdit(
        device=args.device,
        transformer_path=args.transformer.resolve(),
        pipeline_path=args.pipeline.resolve(),
        step=args.steps,
        generation_size=2 * args.resolution,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt="text, labels, watermark, diagnostic colors, inconsistent views",
        cpu_offload=True,
    )
    try:
        edited = editor.generate_with_prompt(
            [Path(evidence["prior_board"]), Path(evidence["correspondence_overlay_board"])],
            prompt,
            size=2 * args.resolution,
            seed=args.seed,
        )
    finally:
        editor.close()
    board = output / "edited_front_side_back_right.png"
    edited.save(board)
    views = split_grid(board, output / "conditions")
    record = {
        "method": "qwen_joint_four_view_positive_partial_evidence",
        "ground_truth_used": False,
        "evidence": evidence,
        "prompt": str(prompt_path),
        "board": str(board),
        "conditions": {name: str(path) for name, path in views.items()},
        "generation": {
            "model": str(args.pipeline.resolve()),
            "transformer": str(args.transformer.resolve()),
            "steps": args.steps,
            "true_cfg_scale": args.true_cfg_scale,
            "seed": args.seed,
        },
    }
    manifest = output / "edit_manifest.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
