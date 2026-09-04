#!/usr/bin/env python3
"""Turn one bounded agent text action into a camera-locked edit target.

The caller supplies a saved-camera prior render and an instruction derived from
no-GT residual evidence. The output is only a 2-D target for a downstream 3-D
editor; it cannot replace a complete prior unless the common proper-Sim(3) and
multi-view gates later accept the resulting 3-D proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qwen_image_edit import QwenImageEdit
from src.agent_image_action_gate import (
    accept_camera_locked_image_action,
    measure_camera_locked_image_action,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path,
                        help="Immutable saved-camera render used to gate a pre-warped control edit.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if not args.source_image.exists() or not args.instruction_file.exists():
        raise FileNotFoundError("source image and instruction file must exist")
    if args.reference_image is not None and not args.reference_image.exists():
        raise FileNotFoundError(args.reference_image)
    instruction = args.instruction_file.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError("instruction file is empty")
    transformer = args.models_dir / "nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors"
    pipeline = args.models_dir / "Qwen-Image-Edit-2511"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    editor = QwenImageEdit(
        device="cuda", transformer_path=transformer, pipeline_path=pipeline,
        step=args.steps, generation_size=args.size, true_cfg_scale=args.true_cfg_scale,
        negative_prompt=" ", cpu_offload=True,
    )
    try:
        result = editor.generate_with_prompt(
            Image.open(args.source_image).convert("RGB"), instruction,
            size=args.size, seed=args.seed,
        )
        output = args.output_dir / "qwen_agent_target.png"
        result.save(output)
    finally:
        editor.close()
    evidence = measure_camera_locked_image_action(
        Image.open(args.source_image).convert("RGB"), Image.open(output).convert("RGB"))
    reference_evidence = None
    if args.reference_image is not None:
        reference_evidence = measure_camera_locked_image_action(
            Image.open(args.reference_image).convert("RGB"), Image.open(output).convert("RGB"))
    accepted = accept_camera_locked_image_action(evidence) and (
        reference_evidence is None or accept_camera_locked_image_action(reference_evidence))
    record = {
        "method": "agent_text_to_camera_locked_qwen_target",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "source_image": str(args.source_image.resolve()),
        "instruction": instruction,
        "transformer": str(transformer.resolve()),
        "pipeline": str(pipeline.resolve()),
        "seed": args.seed, "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale, "negative_prompt": " ",
        "output": str(output.resolve()),
        "saved_camera_image_evidence": evidence.to_dict(),
        "immutable_reference_image": None if args.reference_image is None else str(args.reference_image.resolve()),
        "immutable_reference_evidence": None if reference_evidence is None else reference_evidence.to_dict(),
        "accepted_for_3d_edit": accepted,
        "downstream_contract": "Nano3D target only; common proper-Sim(3) and multi-view gates required",
    }
    (args.output_dir / "qwen_agent_action.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"target": str(output), "seed": args.seed,
                      "accepted_for_3d_edit": accepted}, indent=2))


if __name__ == "__main__":
    main()
