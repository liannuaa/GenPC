#!/usr/bin/env python3
"""Create a separately auditable Redwood run for the GenPC+ agent baseline.

This first run materializes the frozen, GT-free Gaussian-surfel solution as
the controller's trusted action.  The companion policy is deliberately able to
admit a regenerated semantic/Pixal proposal later, but no proposal is silently
substituted for this baseline.  Thus the first agent artifact establishes a
strict no-regression floor before generative actions are enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT.parents[1]
FROZEN_ROOT = (
    ASSET_ROOT / "workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823"
)
FROZEN_FINAL = FROZEN_ROOT / "_semantic_view_gaussian_surfel_v22_20260828"
SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830",
           "07136", "07306", "09639")


def copy_required(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path, destination / name)


def stage_anchor(run_root: Path, samples: tuple[str, ...]) -> dict:
    """Stage immutable agent inputs and the trusted decoded completion action."""
    actions = []
    asset_names = ("depth.png", "qwen_img.png", "gpt_image.png", "prompt.txt",
                   "pixal3d.glb", "pixal3d_sampled_100k.ply")
    for sample in samples:
        copy_required(FROZEN_ROOT / sample, run_root / "inputs" / sample, asset_names)
        copy_required(FROZEN_FINAL / sample, run_root / "final" / sample,
                      (f"{sample}_bidirectional_cycle_registered_uniform.ply",))
        source = run_root / "final" / sample / f"{sample}_bidirectional_cycle_registered_uniform.ply"
        target = run_root / "final" / sample / f"{sample}_agent_completion.ply"
        source.rename(target)
        decision = {
            "sample_id": sample,
            "action": "preserve_prior",
            "reason": "trusted_anchor_before_regeneration_actions",
            "strict_zero_shot": True,
            "ground_truth_used_for_action_selection": False,
            "semantic_regeneration_allowed": True,
            "pixal_regeneration_allowed": True,
            "selected_completion": str(target),
            "source_completion": str(FROZEN_FINAL / sample /
                                     f"{sample}_bidirectional_cycle_registered_uniform.ply"),
        }
        action_dir = run_root / "actions" / sample
        action_dir.mkdir(parents=True, exist_ok=True)
        (action_dir / "decision.json").write_text(json.dumps(decision, indent=2),
                                                    encoding="utf-8")
        actions.append(decision)
    return {"samples": actions}


def evaluate(run_root: Path, samples: tuple[str, ...], python: Path) -> None:
    command = [
        str(python), str(ROOT / "scripts/evaluate_bidirectional_cycle_redwood.py"),
        "--candidate-root", str(run_root / "final"),
        "--candidate-name-template", "{sample}_agent_completion.ply",
        "--v15-root", str(ASSET_ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822"),
        "--output-root", str(run_root / "metrics"),
        "--config", str(ROOT / "configs/config.yaml"),
        "--metric-seed", "6145", "--samples", *samples,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path,
                        default=ROOT / "workspace/agent_genpc_plus_redwood_20260903")
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--python", type=Path,
                        default=Path("/opt/data/private/cr/miniconda3/envs/genpc/bin/python"))
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    samples = tuple(map(str, args.samples))
    manifest = {
        "method": "agent_genpc_plus",
        "mode": "trusted_anchor_audit",
        "strict_zero_shot": True,
        "ground_truth_used_before_metrics": False,
        "frozen_anchor_root": str(FROZEN_FINAL),
        "generation_actions_allowed": ["regenerate_semantic", "regenerate_pixal"],
        "samples": list(samples),
        "actions": stage_anchor(run_root, samples),
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
    if not args.skip_metrics:
        evaluate(run_root, samples, args.python)


if __name__ == "__main__":
    main()
