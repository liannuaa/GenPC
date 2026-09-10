#!/usr/bin/env python3
"""Execute state-approved Pixal prior actions through one shared GPU run.

Inputs and generated assets are hard-linked between per-sample probe folders
and a shared execution cache.  This preserves each sample's isolated state
trace while avoiding redundant model initialization or disk duplication.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_probe_state import GENERATE_PRIOR, AgentDecision, ProbeState


ASSET_NAMES = (
    "pixal3d.glb",
    "pixal3d_input.png",
    "pixal3d_metadata.json",
    "pixal3d_sampled_100k.ply",
    "pixal_moge_fp16_observation.npz",
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("agentic_probe_runner", ROOT / "scripts" / "run_agentic_01184_probe.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the probe runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.stat().st_size != target.stat().st_size:
            raise FileExistsError(f"conflicting target asset: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--decision-name", default="03_generate_prior.json")
    parser.add_argument("--cache-key", default="initial",
                        help="Execution-cache namespace; use a new key after an upstream replan.")
    args = parser.parse_args()

    collection = args.collection_root.resolve()
    runner = _load_runner()
    roots = [(sample, (collection / sample).resolve()) for sample in args.samples]
    for sample, root in roots:
        runner.SAMPLE = sample
        state = ProbeState.from_file(root / runner.STATE_FILENAME)
        decision = AgentDecision.from_file(root / "decisions" / args.decision_name)
        decision.validate(state)
        if decision.action != GENERATE_PRIOR:
            raise ValueError(f"{sample}: batch Pixal runner only accepts {GENERATE_PRIOR}")

    cache_key = str(args.cache_key).strip()
    if not cache_key or any(part in {"", ".", ".."} for part in Path(cache_key).parts):
        raise ValueError("--cache-key must be a non-empty relative namespace")
    cache_root = collection / "_shared_pixal_execution" / cache_key
    cache_input = cache_root / "input"
    cache_output = cache_root / "output"
    for sample, root in roots:
        runner.SAMPLE = sample
        source = runner._paths(root)["pixal_semantic"]
        if not source.is_file():
            raise FileNotFoundError(source)
        _link(source, cache_input / sample / "gpt_image.png")

    command = [
        sys.executable, "scripts/run_pixal3d_gpt_batch.py",
        "--input-root", str(cache_input), "--output-root", str(cache_output),
        "--input-name", "gpt_image.png", "--ids", *args.samples,
        "--seed", "42", "--resolution", "1024", "--point-count", "100000",
    ]
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / "pixal_batch.log").open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"shared Pixal batch failed; see {cache_root / 'pixal_batch.log'}")

    for sample, root in roots:
        runner.SAMPLE = sample
        state = ProbeState.from_file(root / runner.STATE_FILENAME)
        decision = AgentDecision.from_file(root / "decisions" / args.decision_name)
        decision.validate(state)
        cache_sample = cache_output / sample
        paths = runner._paths(root)
        for name in ASSET_NAMES:
            source = cache_sample / name
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            _link(source, paths["pixal"] / name)
        previous_hash = state.state_hash()
        outputs = {
            "prior": str(paths["prior"].resolve()),
            "metadata": str(paths["pixal_metadata"].resolve()),
            "pixal_input": str(paths["pixal_input"].resolve()),
            "shared_execution_cache": str(cache_sample.resolve()),
        }
        state.history.append({
            "step": len(state.history) + 1,
            "previous_state_sha256": previous_hash,
            "decision": decision.as_dict(),
            "tool_outputs": outputs,
            "ground_truth_cd_emd_used": False,
        })
        state.phase = "global_alignment_pending"
        state.budget_remaining -= 1
        state.artifacts.update(outputs)
        runner._write_diagnostics(root, state)
        state.write(root / runner.STATE_FILENAME)
        runner._write_decision_packet(root, state)
        print(f"{sample}: prior generation complete", flush=True)


if __name__ == "__main__":
    main()
