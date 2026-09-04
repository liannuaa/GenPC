#!/usr/bin/env python3
"""Execute one stateful local-or-generative agent edit round.

The round is deliberately backend-local and stateful: it takes only the
current registered prior/mesh plus the observed partial and saved camera. It
first gives the low-risk intrinsic local geometry action a chance to explain a
compact observed residual.  An accepted local state terminates the round and
is deliberately re-observed on the next call.  If that action rejects, the
round may turn no-GT residual evidence into a camera-locked Qwen target, ask
Nano3D for a mesh-edit proposal, re-register it, and accept a full or
support-locked successor only through shared no-harm gates.  Every rejection
writes the input prior verbatim, making the transition safe to iterate.
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
    # ``run_agent_iterative_loop.py`` invokes this file as a child process.
    # Python otherwise puts only ``scripts/`` on sys.path, which makes the
    # repository's namespace package unavailable despite cwd=ROOT.
    sys.path.insert(0, str(ROOT))


def run(*args: str) -> None:
    subprocess.run([str(PYTHON), *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-mesh", type=Path, required=True)
    parser.add_argument("--sample", required=True,
                        help="Dataset sample ID; binds the candidate to its saved camera and partial.")
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--seed", type=int, default=6145)
    parser.add_argument("--nano-seed", type=int, default=1)
    parser.add_argument("--nano-st-step", type=int, default=12)
    parser.add_argument("--qwen-steps", type=int, default=40)
    parser.add_argument("--qwen-true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--instruction-style", choices=("conservative", "explicit_local"),
                        default="conservative")
    parser.add_argument("--residual-warp-gain", type=float, default=0.,
                        help="Optional measured local 2-D control warp before Qwen; 0 disables it.")
    parser.add_argument("--source-encoding-dir", type=Path,
                        help="Optional Nano canonical source encoding cache for repeated rounds.")
    parser.add_argument("--source-voxel-latent", type=Path,
                        help="Matching Nano source voxel latent; required with --source-encoding-dir.")
    parser.add_argument("--edit-trigger-objective", type=float, default=.10)
    parser.add_argument("--disable-intrinsic-action", action="store_true",
                        help="Skip the topology-preserving local action and use the legacy generative branch.")
    parser.add_argument("--skip-nano", action="store_true",
                        help="Write an explicit preserve transition without loading image/3-D editors.")
    args = parser.parse_args()
    for path in (args.anchor, args.anchor_mesh, args.partial, args.camera, args.semantic):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instruction_dir = args.output_dir / "instruction"
    run("scripts/build_agent_saved_view_instruction.py", "--partial", str(args.partial),
        "--prior", str(args.anchor), "--camera", str(args.camera), "--output-dir", str(instruction_dir),
        "--prompt-style", args.instruction_style)
    feedback = json.loads((instruction_dir / "agent_instruction.json").read_text(encoding="utf-8"))
    import scripts.run_pixal_pca_sim3_ttt_v2 as base
    from src.bidirectional_cycle_registration import visible_score

    anchor_points, partial_points = base.load_points(args.anchor), base.load_points(args.partial)
    diagonal = max(float(np.linalg.norm(np.ptp(partial_points, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial_points, args.camera, padding=.15, image_shape=(512, 512), device="cpu")
    anchor_objective = float(visible_score(
        partial_points, anchor_points, projector, diagonal, pixel_radius=5.)["objective"])
    outputs = {"points": args.output_dir / "registered_100k.ply",
               "mesh": args.output_dir / "registered_mesh.glb",
               "trace": args.output_dir / "qwen_nano_round.json"}
    # One round must make one causal state transition.  Local geometry is
    # evaluated first because it preserves the complete prior exactly outside
    # the observed geodesic support; an accepted successor is re-observed in
    # the next round before any higher-risk generative action is considered.
    if not args.disable_intrinsic_action:
        intrinsic_dir = args.output_dir / "intrinsic_local"
        run("scripts/run_agent_intrinsic_residual_edit.py", "--anchor", str(args.anchor),
            "--anchor-mesh", str(args.anchor_mesh), "--partial", str(args.partial),
            "--camera", str(args.camera), "--semantic", str(args.semantic),
            "--output-dir", str(intrinsic_dir), "--seed", str(args.seed))
        intrinsic = json.loads((intrinsic_dir / "intrinsic_residual_action.json").read_text(
            encoding="utf-8"))
        if intrinsic["accepted"]:
            shutil.copy2(intrinsic_dir / "registered_100k.ply", outputs["points"])
            shutil.copy2(intrinsic_dir / "registered_mesh.glb", outputs["mesh"])
            record = {
                "method": "agent_intrinsic_then_qwen_nano_round",
                "strict_zero_shot": True,
                "ground_truth_cd_emd_used": False,
                "sample_or_category_specific_parameters": False,
                "agent_feedback": feedback,
                "intrinsic_local_action": intrinsic,
                "route": "accepted_intrinsic_local_geometry",
                "accepted": True,
                "anchor_objective": anchor_objective,
                "next_step": "reobserve_successor_before_any_further_action",
                "outputs": {key: str(value) for key, value in outputs.items()},
            }
            outputs["trace"].write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(json.dumps({"route": record["route"], "accepted": True,
                              "next_step": record["next_step"]}, indent=2))
            return
    else:
        intrinsic = {"enabled": False, "reason": "disabled_by_cli"}
    if args.skip_nano or not feedback["eligible"] or anchor_objective < args.edit_trigger_objective:
        shutil.copy2(args.anchor, outputs["points"])
        shutil.copy2(args.anchor_mesh, outputs["mesh"])
        record = {"method": "agent_qwen_nano_round", "strict_zero_shot": True,
                  "ground_truth_cd_emd_used": False, "agent_feedback": feedback,
                  "route": "preserve_prior", "accepted": False,
                  "anchor_objective": anchor_objective,
                  "intrinsic_local_action": intrinsic,
                  "reason": ("skip_nano" if args.skip_nano else
                             "residual_below_shared_trigger"),
                  "outputs": {key: str(value) for key, value in outputs.items()}}
        outputs["trace"].write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"route": record["route"], "reason": record["reason"],
                          "anchor_objective": anchor_objective}, indent=2))
        return

    # Render the current 3-D state in the saved camera before editing. Qwen is
    # therefore never asked to infer a new view or a new object scale.
    from scripts.run_nano3d_agent_prior import camera_locked_source
    from PIL import Image

    source_dir = args.output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    camera_locked_source(
        args.anchor, args.partial, args.camera, args.semantic,
        source_dir / "source_render_white.png", source_dir / "target_identity.png",
        source_dir / "prior_mask.png", source_dir / "partial_mask.png",
        padding=.15, target_mode="partial_supported")
    qwen_source = source_dir / "source_render_white.png"
    qwen_reference = None
    warp = None
    if args.residual_warp_gain > 0.:
        warp_dir = args.output_dir / "residual_warp"
        run("scripts/build_agent_residual_warp.py", "--source-image", str(qwen_source),
            "--partial", str(args.partial), "--prior", str(args.anchor), "--camera", str(args.camera),
            "--output-dir", str(warp_dir), "--gain", str(args.residual_warp_gain))
        warp = json.loads((warp_dir / "residual_warp_info.json").read_text(encoding="utf-8"))
        if warp["evidence"]["actionable"]:
            qwen_reference, qwen_source = qwen_source, warp_dir / "residual_warp_control.png"
    qwen_dir = args.output_dir / "qwen_target"
    qwen_command = ["scripts/run_agent_qwen_image_action.py", "--source-image", str(qwen_source), "--instruction-file",
        str(instruction_dir / "agent_instruction.txt"), "--output-dir", str(qwen_dir),
        "--models-dir", str(args.models_dir), "--seed", str(args.seed),
        "--steps", str(args.qwen_steps), "--true-cfg-scale", str(args.qwen_true_cfg_scale)]
    if qwen_reference is not None:
        qwen_command.extend(("--reference-image", str(qwen_reference)))
    run(*qwen_command)
    qwen = json.loads((qwen_dir / "qwen_agent_action.json").read_text(encoding="utf-8"))
    if not qwen["accepted_for_3d_edit"]:
        shutil.copy2(args.anchor, outputs["points"])
        shutil.copy2(args.anchor_mesh, outputs["mesh"])
        record = {"method": "agent_qwen_nano_round", "strict_zero_shot": True,
                  "ground_truth_cd_emd_used": False, "agent_feedback": feedback,
                  "intrinsic_local_action": intrinsic, "residual_warp": warp, "qwen": qwen,
                  "route": "preserve_prior", "accepted": False,
                  "reason": "camera_locked_qwen_gate", "outputs": {key: str(value) for key, value in outputs.items()}}
        outputs["trace"].write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"route": record["route"], "reason": record["reason"]}, indent=2))
        return

    nano_dir = args.output_dir / "nano"
    nano_command = ["scripts/run_nano3d_agent_prior.py", "--source-mesh", str(args.anchor_mesh),
        "--target-render", str(qwen_dir / "qwen_agent_target.png"), "--registered-prior", str(args.anchor),
        "--partial", str(args.partial), "--camera", str(args.camera),
        "--camera-locked-target-mode", "partial_supported", "--trellis-model",
        str(args.models_dir / "TRELLIS-image-large"), "--output-dir", str(nano_dir),
        "--seed", str(args.nano_seed), "--st-step", str(args.nano_st_step)]
    if (args.source_encoding_dir is None) != (args.source_voxel_latent is None):
        raise ValueError("Nano source cache arguments must be supplied together")
    if args.source_encoding_dir is not None:
        nano_command.extend(("--source-encoding-dir", str(args.source_encoding_dir),
                             "--source-voxel-latent", str(args.source_voxel_latent)))
    run(*nano_command)
    registration_input = args.output_dir / "registration_input" / str(args.sample)
    registration_input.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nano_dir / "nano3d_native_sampled_100k.ply", registration_input / "pixal3d_sampled_100k.ply")
    shutil.copy2(nano_dir / "nano3d_native.glb", registration_input / "pixal3d.glb")
    pca_dir = args.output_dir / "pca"
    run("scripts/run_pixal_pca_sim3_ttt_v2.py", "--gpt-root", str(args.output_dir / "registration_input"),
        "--camera-root", str(args.camera.parent.parent), "--output-root", str(pca_dir), "--samples", str(args.sample),
        "--levels", "5", "--gpu-prescreen", "--gpu-topk", "12", "--gpu-device", "cuda")
    proposal_dir = pca_dir / str(args.sample)
    gate_dir = args.output_dir / "gate"
    run("scripts/run_agent_variant_reregistration.py", "--proposal",
        str(proposal_dir / f"{args.sample}_pca_sim3_ttt_v2_registered_100k.ply"), "--proposal-mesh",
        str(proposal_dir / f"{args.sample}_pca_sim3_ttt_v2_registered_mesh.glb"), "--anchor", str(args.anchor),
        "--multiview-anchor", str(args.anchor), "--partial", str(args.partial), "--camera", str(args.camera),
        "--semantic", str(args.semantic), "--output-dir", str(gate_dir), "--action-name", "qwen_nano3d")
    gate = json.loads((gate_dir / "reregistration_info.json").read_text(encoding="utf-8"))
    if gate["accepted"]:
        shutil.copy2(gate_dir / "registered_100k.ply", outputs["points"])
        shutil.copy2(gate_dir / "registered_mesh.glb", outputs["mesh"])
        route = "accepted_global_qwen_nano3d"
    else:
        support_dir = args.output_dir / "support_locked"
        run("scripts/run_agent_support_locked_edit.py", "--anchor", str(args.anchor), "--edited-proposal",
            str(gate_dir / "proposal_refined_100k.ply"), "--partial", str(args.partial), "--camera",
            str(args.camera), "--semantic", str(args.semantic), "--output-dir", str(support_dir))
        support = json.loads((support_dir / "support_locked_info.json").read_text(encoding="utf-8"))
        shutil.copy2(support_dir / "registered_100k.ply", outputs["points"])
        # A support-locked point transition intentionally preserves the old
        # mesh; it may be used as the source mesh of a later agent round.
        shutil.copy2(args.anchor_mesh, outputs["mesh"])
        route = support["route"]
    record = {"method": "agent_qwen_nano_round", "strict_zero_shot": True,
              "ground_truth_cd_emd_used": False, "sample_or_category_specific_parameters": False,
              "agent_feedback": feedback, "intrinsic_local_action": intrinsic, "residual_warp": warp,
              "qwen": qwen, "nano": json.loads((nano_dir / "nano3d_agent_metadata.json").read_text()),
              "global_gate": gate, "route": route, "accepted": route != "anchor_fallback",
              "outputs": {key: str(value) for key, value in outputs.items()}}
    outputs["trace"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"route": route, "accepted": record["accepted"]}, indent=2))


if __name__ == "__main__":
    main()
