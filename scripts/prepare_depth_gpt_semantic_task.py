#!/usr/bin/env python3
"""Write the reproducible direct-GPT task for a saved Camera-1 depth image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.depth_semantic_prompt import build_depth_semantic_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--object", required=True, dest="object_description")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.depth.is_file():
        raise FileNotFoundError(args.depth)
    prompt = build_depth_semantic_prompt(
        args.object_description,
        structural_constraints=tuple(args.constraint),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt + "\n", encoding="utf-8")
    task_path = args.output.with_suffix(".json")
    task_path.write_text(json.dumps({
        "method": "depth_mask_conditioned_direct_gpt_semantic_completion",
        "ground_truth_used": False,
        "depth": str(args.depth.resolve()),
        "object_description": args.object_description,
        "structural_constraints": args.constraint,
        "prompt": prompt,
        "output_contract": {
            "semantic_image": "img.png",
            "camera_and_mask_authority": "input depth raster",
            "background": "pure white",
        },
    }, indent=2) + "\n", encoding="utf-8")
    print(task_path)


if __name__ == "__main__":
    main()
