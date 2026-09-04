#!/usr/bin/env python3
"""Run a bounded, re-observed AgentGenPC+ state loop.

Each successor is produced by ``run_agent_qwen_nano_round.py``.  That round
first attempts a local topology-preserving action and only then considers the
text/image-to-3D edit branch.  This outer loop supplies the missing temporal
contract: every successor is re-scored against the immutable initial state,
the best safe state is retained, and tiny gains or a fixed round budget stop
iteration before local edits can accumulate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
PYTHON = Path("/opt/data/private/cr/miniconda3/envs/genpc/bin/python")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.agent_loop_policy import (
    accepted_against_initial,
    prefers_candidate,
    relative_geometric_gain,
    should_probe_generative_action,
)
from src.bidirectional_cycle_registration import visible_score
from src.multiview_agent_feedback import make_orthographic_reference, measure_multiview_evidence


def run(*args: str) -> None:
    subprocess.run([str(PYTHON), *args], cwd=ROOT, check=True)


def state_evidence(points: np.ndarray, partial: np.ndarray, projector, diagonal: float, reference):
    return (visible_score(partial, points, projector, diagonal, pixel_radius=5.),
            measure_multiview_evidence(partial, points, reference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-mesh", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-relative-geometric-gain", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--skip-nano", action="store_true")
    parser.add_argument("--generative-final-probe", action="store_true",
                        help="After a safe local phase, evaluate exactly one guarded Qwen→Nano 3-D edit.")
    parser.add_argument("--source-encoding-dir", type=Path,
                        help="Optional Nano source-encoding cache tied to the current mesh state.")
    parser.add_argument("--source-voxel-latent", type=Path,
                        help="Matching Nano voxel latent; required with --source-encoding-dir.")
    parser.add_argument("--qwen-steps", type=int, default=40)
    parser.add_argument("--qwen-true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--nano-st-step", type=int, default=12)
    parser.add_argument("--instruction-style", choices=("conservative", "explicit_local"),
                        default="conservative")
    parser.add_argument("--residual-warp-gain", type=float, default=0.)
    args = parser.parse_args()
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be positive")
    if args.generative_final_probe and args.skip_nano:
        raise ValueError("--generative-final-probe cannot be combined with --skip-nano")
    if (args.source_encoding_dir is None) != (args.source_voxel_latent is None):
        raise ValueError("Nano source cache arguments must be supplied together")
    for cache_path in (args.source_encoding_dir, args.source_voxel_latent):
        if cache_path is not None and not cache_path.exists():
            raise FileNotFoundError(cache_path)
    for path in (args.anchor, args.anchor_mesh, args.partial, args.camera, args.semantic):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partial = base.load_points(args.partial)
    initial_points = base.load_points(args.anchor)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=.15, image_shape=(512, 512), device="cpu")
    reference = make_orthographic_reference(partial, initial_points)
    initial_visible, initial_multiview = state_evidence(
        initial_points, partial, projector, diagonal, reference)

    current_points, current_mesh = args.anchor, args.anchor_mesh
    current_visible = initial_visible
    best_points, best_mesh = args.anchor, args.anchor_mesh
    best_visible, best_multiview = initial_visible, initial_multiview
    trace: list[dict] = []
    stop_reason = "max_rounds"
    for round_index in range(args.max_rounds):
        round_dir = args.output_dir / f"round_{round_index:02d}"
        command = ["scripts/run_agent_qwen_nano_round.py", "--anchor", str(current_points),
                   "--anchor-mesh", str(current_mesh), "--sample", str(args.sample),
                   "--partial", str(args.partial), "--camera", str(args.camera), "--semantic",
                   str(args.semantic), "--output-dir", str(round_dir), "--models-dir",
                   str(args.models_dir), "--seed", str(args.seed)]
        if args.skip_nano:
            command.append("--skip-nano")
        run(*command)
        action = json.loads((round_dir / "qwen_nano_round.json").read_text(encoding="utf-8"))
        successor_points = round_dir / "registered_100k.ply"
        successor_mesh = round_dir / "registered_mesh.glb"
        successor = base.load_points(successor_points)
        successor_visible, successor_multiview = state_evidence(
            successor, partial, projector, diagonal, reference)
        global_safe = accepted_against_initial(
            initial_visible, successor_visible, initial_multiview, successor_multiview)
        action_accepted = bool(action.get("accepted", False))
        gain = relative_geometric_gain(current_visible, successor_visible)
        state = {
            "round": round_index, "route": action["route"],
            "action_accepted": action_accepted, "global_safe_against_initial": global_safe,
            "relative_geometric_gain": gain,
            "visible": successor_visible, "multiview": successor_multiview.to_dict(),
            "paths": {"points": str(successor_points), "mesh": str(successor_mesh)},
        }
        trace.append(state)
        if not action_accepted:
            stop_reason = "action_returned_anchor"
            break
        if not global_safe:
            stop_reason = "cross_round_no_harm_gate"
            break
        if prefers_candidate(best_visible, successor_visible):
            best_points, best_mesh = successor_points, successor_mesh
            best_visible, best_multiview = successor_visible, successor_multiview
        current_points, current_mesh, current_visible = successor_points, successor_mesh, successor_visible
        if gain < args.min_relative_geometric_gain:
            stop_reason = "relative_geometric_gain_converged"
            break

    # A text-conditioned mesh edit is deliberately not interleaved with each
    # local solve.  Once local actions have made a safe, re-observed state and
    # stopped, evaluate one distinct global proposal.  The legacy action has
    # its own local gate; the immutable initial-state gate below prevents
    # cumulative drift and the ranker prevents a safe-but-worse replacement.
    generative_probe = None
    if args.generative_final_probe and should_probe_generative_action(trace):
        probe_dir = args.output_dir / "generative_final_probe"
        command = ["scripts/run_agent_qwen_nano_round.py", "--anchor", str(best_points),
                   "--anchor-mesh", str(best_mesh), "--sample", str(args.sample),
                   "--partial", str(args.partial), "--camera", str(args.camera), "--semantic",
                   str(args.semantic), "--output-dir", str(probe_dir), "--models-dir",
                   str(args.models_dir), "--seed", str(args.seed), "--disable-intrinsic-action",
                   "--qwen-steps", str(args.qwen_steps), "--qwen-true-cfg-scale",
                   str(args.qwen_true_cfg_scale), "--nano-st-step", str(args.nano_st_step),
                   "--instruction-style", args.instruction_style]
        if args.residual_warp_gain > 0.:
            command.extend(("--residual-warp-gain", str(args.residual_warp_gain)))
        if args.source_encoding_dir is not None:
            command.extend(("--source-encoding-dir", str(args.source_encoding_dir),
                            "--source-voxel-latent", str(args.source_voxel_latent)))
        run(*command)
        action = json.loads((probe_dir / "qwen_nano_round.json").read_text(encoding="utf-8"))
        successor_points = probe_dir / "registered_100k.ply"
        successor_mesh = probe_dir / "registered_mesh.glb"
        successor = base.load_points(successor_points)
        successor_visible, successor_multiview = state_evidence(
            successor, partial, projector, diagonal, reference)
        global_safe = accepted_against_initial(
            initial_visible, successor_visible, initial_multiview, successor_multiview)
        action_accepted = bool(action.get("accepted", False))
        improves_best = prefers_candidate(best_visible, successor_visible)
        generative_probe = {
            "route": action["route"],
            "action_accepted": action_accepted,
            "global_safe_against_initial": global_safe,
            "improves_current_best": improves_best,
            "visible": successor_visible,
            "multiview": successor_multiview.to_dict(),
            "paths": {"points": str(successor_points), "mesh": str(successor_mesh)},
            "source_cache": {
                "encoding_dir": str(probe_dir / "nano" / "source_encoding"),
                "voxel_latent": str(probe_dir / "nano" / "source_voxel" / "latent.pt"),
            },
        }
        if action_accepted and global_safe and improves_best:
            best_points, best_mesh = successor_points, successor_mesh
            best_visible, best_multiview = successor_visible, successor_multiview
            stop_reason = "generative_final_probe_selected"
        else:
            stop_reason = "generative_final_probe_rejected"

    outputs = {"points": args.output_dir / "registered_100k.ply",
               "mesh": args.output_dir / "registered_mesh.glb",
               "trace": args.output_dir / "agent_loop_trace.json"}
    shutil.copy2(best_points, outputs["points"])
    shutil.copy2(best_mesh, outputs["mesh"])
    record = {
        "method": "bounded_reobserved_agent_loop",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "initial_visible": initial_visible,
        "initial_multiview": initial_multiview.to_dict(),
        "selected_visible": best_visible,
        "selected_multiview": best_multiview.to_dict(),
        "trace": trace,
        "generative_final_probe": generative_probe,
        "stop_reason": stop_reason,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["trace"].write_text(json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps({"stop_reason": stop_reason, "rounds": len(trace),
                      "initial_objective": initial_visible["objective"],
                      "selected_objective": best_visible["objective"]}, indent=2))


if __name__ == "__main__":
    main()
