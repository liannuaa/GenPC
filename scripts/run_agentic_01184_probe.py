#!/usr/bin/env python3
"""Run an isolated, state-bound agentic completion probe on one partial scan.

The script deliberately does *not* contain an automatic policy.  At each
state it emits a compact diagnostic board plus an MLLM decision packet; the
next tool call is accepted only when a planner returns a valid JSON decision
bound to that exact state hash.  Qwen, Pixal, proper Sim(3), and Gaussian
editing remain unchanged deterministic executors.

This is a feasibility probe, not a replacement for the frozen mainline.
All artifacts are written below ``--root`` and no existing output is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
import yaml
from munch import Munch


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_probe_state import (
    ACCEPT,
    ADAPT_LOCAL,
    ALIGN_GLOBAL,
    COMPLETE_SEMANTIC,
    GENERATE_PRIOR,
    REFINE_ALIGNMENT,
    SELECT_VIEW,
    AgentDecision,
    ProbeState,
)
from src.bidirectional_cycle_registration import visible_score
from src.depth_prompting import DepthPrompting
from src.mainline_data import load_partial, prompt_label
from src.mainline_paths import GAUSSIAN_PREDICTION_FILENAME, REGISTERED_PRIOR_FILENAME, model_path
from src.moge_pixel_bridge import run_rmbg_mask, save_mask_png
from src.pointcloud_io import load_points
from src.saved_camera import SavedCameraProjector, draw_projection_overlay
from tools.qwen_image_edit import QwenImageEdit, resize_stage1_image_for_output


SAMPLE = "01184"
REDWOOD10 = (
    "01184", "05117", "05452", "06127", "06145",
    "06188", "06830", "07136", "07306", "09639",
)
STATE_FILENAME = "agent_state.json"


def _load_cfg(output_dir: Path) -> Munch:
    cfg = Munch.fromDict(yaml.safe_load((ROOT / "configs" / "mainline_redwood.yaml").read_text(encoding="utf-8")))
    cfg.paths.output_dir = str(output_dir.resolve())
    cfg.paths.data_dir = str((ROOT / "data" / "redwood" / "partial").resolve())
    cfg.paths.models_dir = str((SHARED_ROOT / "models").resolve())
    return cfg


def _paths(root: Path) -> dict[str, Path]:
    camera = root / "inputs" / "camera" / SAMPLE
    pixal = root / "inputs" / "pixal" / SAMPLE
    registration = root / "registration" / SAMPLE
    gaussian = root / "gaussian" / SAMPLE
    return {
        "root": root,
        "partial": root / "inputs" / "partial" / f"{SAMPLE}.ply",
        "candidate_root": root / "view_candidates",
        "camera": camera,
        "pixal": pixal,
        "depth": camera / "depth.png",
        "raw_depth": camera / "raw_depth.png",
        "semantic": camera / "img.png",
        "source_mask": camera / f"{SAMPLE}_moge_to_raw_partial_object_mask.png",
        "point_uv": camera / "point_uv.npy",
        "camera_pth": camera / "camera.pth",
        "pixal_semantic": pixal / "gpt_image.png",
        "prior": pixal / "pixal3d_sampled_100k.ply",
        "pixal_input": pixal / "pixal3d_input.png",
        "pixal_metadata": pixal / "pixal3d_metadata.json",
        "pixal_cache": pixal / "pixal_moge_fp16_observation.npz",
        "registration": registration,
        "native": registration / "native",
        "bridge": registration / "bridge",
        "joint": registration / "joint",
        "amplified": registration / "amplified",
        "wide_tilt": registration / "wide_tilt",
        "final_registration": registration / "final",
        "joint_prior": registration / "joint" / "two_camera_joint_registered_100k.ply",
        "final_prior": registration / "final" / REGISTERED_PRIOR_FILENAME,
        "prediction": gaussian / "decoded" / GAUSSIAN_PREDICTION_FILENAME,
        "diagnostics": root / "diagnostics",
        "final": root / "final" / "agent_selected_100k.ply",
    }


def _state_path(root: Path) -> Path:
    return root / STATE_FILENAME


def _resolve_probe_root(root: Path) -> Path:
    """Accept either a per-sample root or its collection parent for resume."""
    root = root.resolve()
    if _state_path(root).is_file():
        return root
    nested = root / SAMPLE
    if _state_path(nested).is_file():
        return nested
    return root


def _run(command: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[tool]", " ".join(command), flush=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"tool failed with exit status {result.returncode}; see {log}")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[agent-probe elapsed_seconds={time.time() - started:.3f}]\n")


def _copy(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _yaw(viewpoint: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    rotation = np.asarray(((math.cos(radians), 0.0, math.sin(radians)),
                           (0.0, 1.0, 0.0),
                           (-math.sin(radians), 0.0, math.cos(radians))), dtype=np.float32)
    return rotation @ np.asarray(viewpoint, dtype=np.float32).reshape(3)


def _depth_tile(path: Path, title: str, *, size: int = 384) -> Image.Image:
    if path.is_file():
        image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    else:
        image = Image.new("RGB", (size, size), (20, 20, 20))
    tile = Image.new("RGB", (size, size + 30), (20, 20, 20))
    tile.paste(image, (0, 30))
    ImageDraw.Draw(tile).text((8, 8), title, fill=(255, 255, 255))
    return tile


def _write_grid(tiles: list[Image.Image], path: Path) -> None:
    if not tiles:
        return
    width, height = tiles[0].size
    while len(tiles) % 3:
        tiles.append(Image.new("RGB", (width, height), (20, 20, 20)))
    board = Image.new("RGB", (3 * width, (len(tiles) // 3) * height), (20, 20, 20))
    for index, tile in enumerate(tiles):
        board.paste(tile, ((index % 3) * width, (index // 3) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)


def _prepare_view_candidates(root: Path) -> ProbeState:
    paths = _paths(root)
    if _state_path(root).exists():
        raise FileExistsError(f"probe already initialized: {_state_path(root)}")
    source_partial = ROOT / "data" / "redwood" / "partial" / f"{SAMPLE}.ply"
    _copy(source_partial, paths["partial"])
    points, colors = load_partial(paths["partial"])
    xyz = torch.from_numpy(points).to("cuda")
    rgb = torch.from_numpy(colors).to("cuda")

    # Determine the frozen 256-view base camera from the partial alone, then
    # expose a deliberately small, antipodally complete yaw family to the
    # planner.  The legacy saved-view routine contains a depth-sum front/back
    # tie-break.  That score measures only visible geometry and can choose the
    # semantic rear of an otherwise well-covered object.  Retaining its exact
    # result *and* the antipodal view keeps convex-hull coverage as a proposal
    # mechanism while making the front/back decision explicit and auditable.
    # No semantic, prior, GT, or metric is read in this tool.
    base_cfg = _load_cfg(paths["candidate_root"] / "base" / "camera")
    base_prompting = DepthPrompting(base_cfg)
    _, base_eye, _, _, _ = base_prompting._select_saved_view(xyz)
    candidates = {
        "base": None,
        "opposite_180": _yaw(base_eye, 180.0),
        "yaw_plus_90": _yaw(base_eye, 90.0),
        "yaw_minus_90": _yaw(base_eye, -90.0),
    }
    records: dict[str, dict[str, object]] = {}
    tiles: list[Image.Image] = []
    try:
        for name, override in candidates.items():
            output = paths["candidate_root"] / name / "camera"
            cfg = _load_cfg(output)
            prompting = DepthPrompting(cfg)
            raw_depth = prompting._save_depth(
                xyz, rgb, SAMPLE,
                viewpoint_override=None if override is None else np.asarray(override, dtype=np.float32),
            )
            sample_dir = output / SAMPLE
            eye = np.load(sample_dir / "viewpoint.npy").astype(np.float32)
            depth = sample_dir / "depth.png"
            nonzero = int(np.count_nonzero(np.asarray(Image.open(depth).convert("L"))))
            records[name] = {
                "candidate": name,
                "viewpoint": eye.tolist(),
                "raw_depth": str(raw_depth.resolve()),
                "depth": str(depth.resolve()),
                "camera": str((sample_dir / "camera.pth").resolve()),
                "point_uv": str((sample_dir / "point_uv.npy").resolve()),
                "nonzero_depth_pixels": nonzero,
                "source": "partial-only 256-view selection" if name == "base" else "partial-only fixed yaw around base",
                "ground_truth_used": False,
            }
            tiles.append(_depth_tile(depth, f"{name}: {nonzero} occupied px"))
    finally:
        del xyz, rgb
        gc.collect()
        torch.cuda.empty_cache()
    board = paths["diagnostics"] / "view_candidates.png"
    _write_grid(tiles, board)
    candidate_manifest = {
        "method": "partial_only_bounded_camera_candidates",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_id": SAMPLE,
        "candidates": records,
    }
    manifest_path = paths["candidate_root"] / "candidate_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8")
    state = ProbeState(
        sample_id=SAMPLE,
        phase="view_candidates",
        budget_remaining=8,
        artifacts={
            "partial": str(paths["partial"].resolve()),
            "candidate_manifest": str(manifest_path.resolve()),
            "diagnostic_board": str(board.resolve()),
        },
        diagnostics={
            "view_candidates": records,
            "verifier": {
                "available": ["partial_visibility", "depth_occupancy"],
                "ground_truth_cd_emd_used": False,
            },
        },
    )
    return state


def _semantic_iou(depth_path: Path, semantic_path: Path) -> dict[str, float] | None:
    if not depth_path.is_file() or not semantic_path.is_file():
        return None
    depth = np.asarray(Image.open(depth_path).convert("L"), dtype=np.uint8) > 4
    semantic = np.asarray(Image.open(semantic_path).convert("RGB"), dtype=np.uint8)
    foreground = np.min(semantic, axis=-1) < 245
    union = int(np.count_nonzero(depth | foreground))
    return {
        "iou": float(np.count_nonzero(depth & foreground) / max(union, 1)),
        "depth_coverage": float(np.count_nonzero(depth & foreground) / max(int(np.count_nonzero(depth)), 1)),
        "semantic_leakage": float(np.count_nonzero(foreground & ~depth) / max(int(np.count_nonzero(foreground)), 1)),
    }


def _compact_visible(score: dict) -> dict[str, object]:
    projection = score["projection"]
    return {
        "objective": float(score["objective"]),
        "visible_3d": float(score["geometric"]["objective"]),
        "visible_pair_count": int(len(score["geometric"]["partial_ids"])),
        "silhouette_iou": float(projection["iou"]),
        "coverage": float(projection["coverage"]),
        "leakage": float(projection["leakage"]),
        "depth_error_normalized": float(projection["visible_depth_normalized"]),
    }


def _current_prior(paths: dict[str, Path]) -> Path | None:
    for path in (paths["prediction"], paths["final_prior"], paths["joint_prior"]):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _extent_summary(partial: np.ndarray, prior: np.ndarray) -> dict[str, object]:
    partial_extent = np.ptp(partial, axis=0)
    prior_extent = np.ptp(prior, axis=0)
    ratio = prior_extent / np.maximum(partial_extent, 1e-8)
    return {
        "partial_extent": partial_extent.tolist(),
        "prior_extent": prior_extent.tolist(),
        "axis_ratio": ratio.tolist(),
        "log_scale_residual": float(np.mean(np.abs(np.log(np.maximum(ratio, 1e-8))))),
    }


def _observed_surface_support(partial: np.ndarray, prior: np.ndarray) -> dict[str, object]:
    """Summarize one-sided observed-surface support without GT.

    A rendered silhouette can remain stable despite a small normal-direction
    surface offset.  This compact nearest-surface diagnostic is therefore
    deliberately one-sided: every partial point asks whether it is explained
    by the complete prior, while unobserved prior regions remain unconstrained.
    Ratios are normalized by the partial diagonal, so the same evidence can be
    interpreted across objects and datasets.
    """
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    residual = cKDTree(prior).query(partial, k=1, workers=-1)[0] / diagonal
    coverage_1pct = float(np.mean(residual <= .01))
    coverage_2pct = float(np.mean(residual <= .02))
    p90 = float(np.quantile(residual, .90))
    # This is advice for the high-level planner, never a forced action.  A
    # bounded local edit is appropriate only if almost all observed points are
    # already within the fixed 2% trust region; otherwise the failure belongs
    # to view/semantic/prior selection or global alignment instead.
    if coverage_1pct >= .95:
        recommendation = "ACCEPT"
        reason = "The registered complete prior already supports at least 95% of observed partial points within 1% of the partial diagonal."
    elif coverage_2pct >= .90 and p90 <= .02:
        recommendation = "ADAPT_LOCAL"
        reason = "The residual is local and remains inside the fixed 2% local-edit trust region."
    else:
        recommendation = "REPLAN_UPSTREAM"
        reason = "A substantial observed region lies outside the fixed local-edit trust region, so local adaptation would conceal an upstream view, prior, or global-alignment failure."
    return {
        "partial_to_prior_mean_ratio": float(np.mean(residual)),
        "partial_to_prior_p90_ratio": p90,
        "partial_to_prior_p95_ratio": float(np.quantile(residual, .95)),
        "coverage_within_1pct": coverage_1pct,
        "coverage_within_2pct": coverage_2pct,
        "local_action_recommendation": recommendation,
        "recommendation_reason": reason,
        "ground_truth_cd_emd_used": False,
    }


def _write_diagnostics(root: Path, state: ProbeState) -> None:
    paths = _paths(root)
    sequence = len(state.history)
    directory = paths["diagnostics"] / f"step_{sequence:02d}_{state.phase}"
    directory.mkdir(parents=True, exist_ok=True)
    if state.phase == "view_candidates":
        board = paths["diagnostics"] / "view_candidates.png"
        report = {
            "strict_zero_shot": True,
            "ground_truth_cd_emd_used": False,
            "state_phase": state.phase,
            "view_candidates": state.diagnostics.get("view_candidates", {}),
            "diagnostic_board": str(board.resolve()),
            "verifier_energy": None,
        }
        verifier = directory / "verifier.json"
        verifier.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        state.artifacts["diagnostic_board"] = str(board.resolve())
        state.artifacts["verifier"] = str(verifier.resolve())
        state.diagnostics = report
        return
    semantic = _semantic_iou(paths["depth"], paths["semantic"])
    prior_path = _current_prior(paths)
    visible = None
    extent = None
    surface_support = None
    overlay = None
    if prior_path is not None and paths["partial"].is_file() and paths["camera_pth"].is_file():
        partial = load_points(paths["partial"])
        prior = load_points(prior_path)
        diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
        projector = SavedCameraProjector.from_partial(
            partial, paths["camera_pth"], padding=.15, image_shape=(512, 512), device="cpu"
        )
        visible = _compact_visible(visible_score(partial, prior, projector, diagonal, pixel_radius=5.0))
        extent = _extent_summary(partial, prior)
        surface_support = _observed_surface_support(partial, prior)
        if paths["semantic"].is_file():
            overlay = directory / "camera1_overlay.png"
            draw_projection_overlay(overlay, paths["semantic"], partial, prior, projector)
    protection = None
    if prior_path == paths["prediction"] and paths["final_prior"].is_file():
        registered, adapted = load_points(paths["final_prior"]), load_points(prior_path)
        if len(registered) == len(adapted):
            diagonal = max(float(np.linalg.norm(np.ptp(registered, axis=0))), 1e-8)
            displacement = np.linalg.norm(adapted - registered, axis=1) / diagonal
            protection = {
                "mean_displacement_ratio": float(np.mean(displacement)),
                "p90_displacement_ratio": float(np.quantile(displacement, .90)),
                "max_displacement_ratio": float(np.max(displacement)),
                "slot_count_preserved": int(len(adapted)),
            }
    energy = None
    if visible is not None:
        energy = float(
            visible["visible_3d"]
            + .20 * (1.0 - visible["silhouette_iou"])
            + .10 * (1.0 - visible["coverage"])
            + .15 * visible["leakage"]
            + .25 * min(visible["depth_error_normalized"], .20)
        )
    report = {
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "state_phase": state.phase,
        "current_prior": str(prior_path.resolve()) if prior_path is not None else None,
        "semantic_vs_depth": semantic,
        "registered_prior_vs_partial": visible,
        "extent": extent,
        "observed_surface_support": surface_support,
        "prior_protection": protection,
        "verifier_energy": energy,
        "diagnostic_board": None,
    }
    tiles = [
        _depth_tile(paths["depth"], "Camera-1 depth"),
        _depth_tile(paths["semantic"], "semantic observation"),
        _depth_tile(paths["native"] / "pixal_native_moge_projection.png", "native prior / image geometry"),
        _depth_tile(paths["joint"] / "two_camera_joint_saved_view_projection.png", "global alignment"),
        _depth_tile(paths["final_registration"] / "camera1_amplified_saved_view_projection.png", "residual alignment"),
        _depth_tile(overlay if overlay is not None else Path(), "current partial/prior overlay"),
    ]
    board = directory / "diagnostic_board.png"
    _write_grid(tiles, board)
    report["diagnostic_board"] = str(board.resolve())
    (directory / "verifier.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    state.artifacts["diagnostic_board"] = str(board.resolve())
    state.artifacts["verifier"] = str((directory / "verifier.json").resolve())
    state.diagnostics = report


def _write_decision_packet(root: Path, state: ProbeState) -> None:
    path = root / "next_agent_decision.md"
    allowed = ", ".join(state.as_dict()["allowed_actions"]) or "none"
    packet = f"""# Agent decision packet: {SAMPLE}

You are the high-level planner for a bounded zero-shot completion agent.
Read the current state and diagnostic board, then select exactly one permitted
discrete action. Do **not** supply pose, translation, scale, thresholds, or
continuous optimization parameters; deterministic geometry tools own them.
Do **not** use ground truth, CD, or EMD.

- State: `{_state_path(root).resolve()}`
- Diagnostic board: `{state.artifacts.get('diagnostic_board', 'not yet available')}`
- Allowed next actions: `{allowed}`
- Remaining tool budget: `{state.budget_remaining}`

Return JSON in this exact form:

```json
{{
  "state_sha256": "{state.state_hash()}",
  "action": {{"name": "ONE_ALLOWED_ACTION", "arguments": {{}}}},
  "rationale": "Evidence from the current diagnostic board and verifier.",
  "planner": "MLLM model identifier",
  "ground_truth_used": false
}}
```

Argument contract:

- `SELECT_VIEW`: `{{"candidate": "base" | "opposite_180" | "yaw_plus_90" | "yaw_minus_90"}}`
- `COMPLETE_SEMANTIC`: `{{"strategy": "qwen"}}`
- `GENERATE_PRIOR`, `ALIGN_GLOBAL`, `REFINE_ALIGNMENT`, `ADAPT_LOCAL`, and
  `ACCEPT`: `{{}}`.
"""
    path.write_text(packet, encoding="utf-8")


def _commit_view(root: Path, candidate: str) -> dict[str, str]:
    paths = _paths(root)
    source = paths["candidate_root"] / candidate / "camera" / SAMPLE
    if candidate not in {"base", "opposite_180", "yaw_plus_90", "yaw_minus_90"}:
        raise ValueError(f"unknown bounded view candidate: {candidate}")
    for name in ("raw_depth.png", "depth.png", "mask.png", "point_uv.npy", "viewpoint.npy", "camera.pth"):
        _copy(source / name, paths["camera"] / name)
    return {"candidate": candidate, "camera": str((paths["camera"] / "camera.pth").resolve()),
            "depth": str((paths["camera"] / "depth.png").resolve())}


def _complete_semantic(
    root: Path,
    strategy: str,
    *,
    shared_editor: QwenImageEdit | None = None,
) -> dict[str, str]:
    if strategy != "qwen":
        raise ValueError("the first probe supports only the deterministic Qwen executor")
    paths = _paths(root)
    cfg = _load_cfg(paths["camera"].parent)
    raw_depth = Image.open(paths["raw_depth"]).convert("RGB").resize(
        (int(cfg.depth_image_input_res), int(cfg.depth_image_input_res)), Image.Resampling.LANCZOS
    )
    owns_editor = shared_editor is None
    editor = shared_editor or QwenImageEdit(
        device="cuda",
        transformer_path=str(model_path(cfg, "qwen_edit_transformer_path", "nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors")),
        pipeline_path=str(model_path(cfg, "qwen_edit_pipeline_path", "Qwen-Image-Edit-2511")),
        step=int(cfg.qwen_edit_steps), generation_size=int(cfg.qwen_edit_generate_res),
        true_cfg_scale=float(cfg.qwen_edit_true_cfg_scale), negative_prompt=str(cfg.qwen_edit_negative_prompt),
        cpu_offload=True,
    )
    try:
        semantic = editor.generate(raw_depth, prompt_label(SAMPLE, cfg), size=int(cfg.generate_res))
        semantic.save(paths["semantic"])
        resize_stage1_image_for_output(editor.last_stage1_image, int(cfg.generate_res)).save(
            paths["camera"] / "qwen_edit_stage1.png"
        )
        (paths["camera"] / "qwen_edit_prompt.txt").write_text(
            "\n".join((
                "input_image: raw_depth.png", f"prompt: {editor.last_prompt}",
                f"negative_prompt: {editor.negative_prompt!r}",
                f"true_cfg_scale: {editor.true_cfg_scale}", f"num_inference_steps: {editor.step}",
            )) + "\n", encoding="utf-8",
        )
    finally:
        if owns_editor:
            editor.close()
    alpha = run_rmbg_mask(
        paths["semantic"], paths["camera"] / "img_rmbg.png",
        SHARED_ROOT / "models" / "RMBG-2.0",
    )
    save_mask_png(paths["source_mask"], alpha)
    _copy(paths["semantic"], paths["pixal_semantic"])
    return {"semantic": str(paths["semantic"].resolve()), "source_mask": str(paths["source_mask"].resolve()),
            "pixal_input_semantic": str(paths["pixal_semantic"].resolve())}


def _generate_prior(root: Path) -> dict[str, str]:
    paths = _paths(root)
    _run([
        sys.executable, "scripts/run_pixal3d_gpt_batch.py", "--input-root", str(paths["pixal"].parent),
        "--output-root", str(paths["pixal"].parent), "--input-name", "gpt_image.png", "--ids", SAMPLE,
        "--seed", "42", "--resolution", "1024", "--point-count", "100000",
    ], log=paths["pixal"] / "generate_prior.log")
    return {"prior": str(paths["prior"].resolve()), "metadata": str(paths["pixal_metadata"].resolve()),
            "pixal_input": str(paths["pixal_input"].resolve())}


def _align_global(root: Path) -> dict[str, str]:
    paths = _paths(root)
    models = SHARED_ROOT / "models"
    _run([
        sys.executable, "scripts/run_pixal_native_moge_registration.py", "--prior", str(paths["prior"]),
        "--pixal-metadata", str(paths["pixal_metadata"]), "--pixal-input", str(paths["pixal_input"]),
        "--rmbg-model", str(models / "RMBG-2.0"), "--cached-moge-observation", str(paths["pixal_cache"]),
        "--output-dir", str(paths["native"]), "--device", "cuda", "--fp16",
    ], log=paths["native"] / "stage.log")
    _run([
        sys.executable, "scripts/run_pixal_moge_two_camera_bridge.py", "--partial", str(paths["partial"]),
        "--point-uv", str(paths["point_uv"]), "--source-mask", str(paths["source_mask"]),
        "--target-mask", str(paths["native"] / "pixal_input_object_mask.png"),
        "--native-moge", str(paths["native"] / "pixal_native_moge_points.ply"),
        "--native-moge-info", str(paths["native"] / "pixal_native_moge_info.json"),
        "--pixal-prior", str(paths["prior"]), "--partial-camera", str(paths["camera_pth"]),
        "--saved-view-image", str(paths["semantic"]), "--output-dir", str(paths["bridge"]), "--device", "cpu",
    ], log=paths["bridge"] / "stage.log")
    _run([
        sys.executable, "scripts/run_pixal_moge_joint_bundle.py", "--partial", str(paths["partial"]),
        "--pixal-prior", str(paths["prior"]), "--native-moge", str(paths["native"] / "pixal_native_moge_points.ply"),
        "--native-info", str(paths["native"] / "pixal_native_moge_info.json"),
        "--bridge-transform", str(paths["bridge"] / "two_camera_pixal_moge_native_moge_to_partial.npy"),
        "--pixel-matches", str(paths["bridge"] / "two_camera_pixal_moge_partial_to_native_moge_matches.npy"),
        "--partial-camera", str(paths["camera_pth"]), "--semantic", str(paths["semantic"]),
        "--output-dir", str(paths["joint"]), "--coarse-basin-recovery", "--device", "cpu",
    ], log=paths["joint"] / "stage.log")
    return {"global_registered_prior": str(paths["joint_prior"].resolve())}


def _refine_alignment(root: Path) -> dict[str, str]:
    paths = _paths(root)
    common = ["--partial", str(paths["partial"]), "--camera", str(paths["camera_pth"]),
              "--semantic", str(paths["semantic"]), "--search-points", "32000", "--device", "cpu"]
    _run([
        sys.executable, "scripts/run_camera1_amplified_sim3_refine.py", "--registered-prior", str(paths["joint_prior"]),
        "--output-dir", str(paths["amplified"]), *common,
    ], log=paths["amplified"] / "stage.log")
    amplified = paths["amplified"] / REGISTERED_PRIOR_FILENAME
    _run([
        sys.executable, "scripts/run_camera1_amplified_sim3_refine.py", "--registered-prior", str(amplified),
        "--output-dir", str(paths["wide_tilt"]), "--wide-tilt-search", "--max-tilt-degrees", "1.0", *common,
    ], log=paths["wide_tilt"] / "stage.log")
    wide = paths["wide_tilt"] / REGISTERED_PRIOR_FILENAME
    _run([
        sys.executable, "scripts/run_camera1_amplified_sim3_refine.py", "--registered-prior", str(wide),
        "--output-dir", str(paths["final_registration"]), "--wide-tilt-search", "--max-tilt-degrees", "1.0", *common,
    ], log=paths["final_registration"] / "stage.log")
    return {"refined_registered_prior": str(paths["final_prior"].resolve())}


def _adapt_local(root: Path) -> dict[str, str]:
    paths = _paths(root)
    # The agent-facing local tool is deliberately stricter than the frozen
    # mainline ablation.  Camera-1 pixels are only a sparse visible constraint:
    # a .075 cap can erase thin, already-aligned details (e.g., wheels).  The
    # .020 cap is a fixed executor setting, not an MLLM-selected hyperparameter.
    _run([
        sys.executable, "scripts/run_mainline_gaussian.py", "--root", str(root),
        "--registration-root", str(paths["registration"].parent), "--samples", SAMPLE,
        "--max-pixel-distance", "1.0", "--virtual-max-pixel-distance", "1.0",
        "--max-anchor-residual-ratio", ".020", "--max-displacement-ratio", ".020",
    ], log=paths["root"] / "gaussian" / "agent_tool.log")
    return {"adapted_prediction": str(paths["prediction"].resolve())}


def _accept(root: Path) -> dict[str, str]:
    paths = _paths(root)
    source = _current_prior(paths)
    if source is None:
        raise FileNotFoundError("cannot accept before a registered prior exists")
    _copy(source, paths["final"])
    return {"accepted_prediction": str(paths["final"].resolve()), "selected_source": str(source.resolve())}


def _execute(root: Path, state: ProbeState, decision: AgentDecision) -> tuple[str, dict[str, str]]:
    action, arguments = decision.action, decision.arguments
    if action == SELECT_VIEW:
        return "semantic_pending", _commit_view(root, str(arguments.get("candidate", "")))
    if action == COMPLETE_SEMANTIC:
        return "prior_pending", _complete_semantic(root, str(arguments.get("strategy", "")))
    if action == GENERATE_PRIOR:
        return "global_alignment_pending", _generate_prior(root)
    if action == ALIGN_GLOBAL:
        return "alignment_diagnosis", _align_global(root)
    if action == REFINE_ALIGNMENT:
        return "adaptation_diagnosis", _refine_alignment(root)
    if action == ADAPT_LOCAL:
        return "final_diagnosis", _adapt_local(root)
    if action == ACCEPT:
        return "accepted", _accept(root)
    raise AssertionError(action)


def initialize(args: argparse.Namespace) -> None:
    root = _resolve_probe_root(args.root)
    state = _prepare_view_candidates(root)
    _write_diagnostics(root, state)
    state.write(_state_path(root))
    _write_decision_packet(root, state)
    print(json.dumps(state.as_dict(), indent=2))


def apply(args: argparse.Namespace) -> None:
    root = _resolve_probe_root(args.root)
    state = ProbeState.from_file(_state_path(root))
    decision = AgentDecision.from_file(args.decision)
    decision.validate(state)
    if state.budget_remaining <= 0:
        raise RuntimeError("agent tool budget is exhausted")
    previous_hash = state.state_hash()
    next_phase, outputs = _execute(root, state, decision)
    state.history.append({
        "step": len(state.history) + 1,
        "previous_state_sha256": previous_hash,
        "decision": decision.as_dict(),
        "tool_outputs": outputs,
        "ground_truth_cd_emd_used": False,
    })
    state.phase = next_phase
    state.budget_remaining -= 1
    state.artifacts.update(outputs)
    _write_diagnostics(root, state)
    state.write(_state_path(root))
    _write_decision_packet(root, state)
    print(json.dumps(state.as_dict(), indent=2))


def show(args: argparse.Namespace) -> None:
    state = ProbeState.from_file(_state_path(_resolve_probe_root(args.root)))
    print(json.dumps(state.as_dict(), indent=2))


def main() -> None:
    global SAMPLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Per-sample isolated output root for this probe (for example, workspace/probe/01184).")
    parser.add_argument("--sample", default=SAMPLE,
                        help="Redwood partial identifier for this isolated probe.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("initialize", help="Generate partial-only view candidates and first state.")
    init.set_defaults(handler=initialize)
    apply_parser = subparsers.add_parser("apply", help="Execute one state-bound MLLM decision.")
    apply_parser.add_argument("--decision", type=Path, required=True)
    apply_parser.set_defaults(handler=apply)
    show_parser = subparsers.add_parser("show", help="Print the latest state without executing a tool.")
    show_parser.set_defaults(handler=show)
    args = parser.parse_args()
    SAMPLE = str(args.sample)
    if SAMPLE not in REDWOOD10:
        parser.error(f"this feasibility probe is intentionally limited to the Redwood-10 ids: {', '.join(REDWOOD10)}")
    args.handler(args)


if __name__ == "__main__":
    main()
