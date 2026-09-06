#!/usr/bin/env python3
"""Execute already-approved agentic Qwen actions with one shared model load.

The planner decision remains per sample and state-hash-bound.  This runner only
shares the deterministic Qwen executor across independent samples, avoiding a
model reload for every semantic-completion tool invocation.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_probe_state import COMPLETE_SEMANTIC, AgentDecision, ProbeState


def _load_runner():
    spec = importlib.util.spec_from_file_location("agentic_probe_runner", ROOT / "scripts" / "run_agentic_01184_probe.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the probe runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--decision-name", default="02_complete_qwen.json")
    args = parser.parse_args()

    runner = _load_runner()
    roots = [(sample, (args.collection_root / sample).resolve()) for sample in args.samples]
    for sample, root in roots:
        runner.SAMPLE = sample
        state = ProbeState.from_file(root / runner.STATE_FILENAME)
        decision = AgentDecision.from_file(root / "decisions" / args.decision_name)
        decision.validate(state)
        if decision.action != COMPLETE_SEMANTIC:
            raise ValueError(f"{sample}: batch semantic runner only accepts {COMPLETE_SEMANTIC}")

    first_sample, first_root = roots[0]
    runner.SAMPLE = first_sample
    first_cfg = runner._load_cfg(runner._paths(first_root)["camera"].parent)
    editor = runner.QwenImageEdit(
        device="cuda",
        transformer_path=str(runner.model_path(
            first_cfg, "qwen_edit_transformer_path",
            "nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors",
        )),
        pipeline_path=str(runner.model_path(first_cfg, "qwen_edit_pipeline_path", "Qwen-Image-Edit-2511")),
        step=int(first_cfg.qwen_edit_steps),
        generation_size=int(first_cfg.qwen_edit_generate_res),
        true_cfg_scale=float(first_cfg.qwen_edit_true_cfg_scale),
        negative_prompt=str(first_cfg.qwen_edit_negative_prompt),
        cpu_offload=True,
    )
    try:
        for sample, root in roots:
            runner.SAMPLE = sample
            state = ProbeState.from_file(root / runner.STATE_FILENAME)
            decision = AgentDecision.from_file(root / "decisions" / args.decision_name)
            decision.validate(state)
            previous_hash = state.state_hash()
            outputs = runner._complete_semantic(root, decision.arguments.get("strategy", ""), shared_editor=editor)
            state.history.append({
                "step": len(state.history) + 1,
                "previous_state_sha256": previous_hash,
                "decision": decision.as_dict(),
                "tool_outputs": outputs,
                "ground_truth_cd_emd_used": False,
            })
            state.phase = "prior_pending"
            state.budget_remaining -= 1
            state.artifacts.update(outputs)
            runner._write_diagnostics(root, state)
            state.write(root / runner.STATE_FILENAME)
            runner._write_decision_packet(root, state)
            print(f"{sample}: semantic completion complete", flush=True)
    finally:
        editor.close()


if __name__ == "__main__":
    main()
