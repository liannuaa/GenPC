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
import hashlib
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
    ADAPT_AXIS_SCALE,
    ALIGN_GLOBAL,
    COMPLETE_SEMANTIC,
    GENERATE_PRIOR,
    REPLAN_VIEW,
    REFINE_SEMANTIC,
    REFINE_ALIGNMENT,
    RESCUE_GLOBAL,
    RESTORE_ATTEMPT,
    SELECT_CONDITIONING_IMAGE,
    SELECT_VIEW,
    AgentDecision,
    ProbeState,
)
from src.bidirectional_cycle_registration import visible_score
from src.depth_prompting import DepthPrompting
from src.mainline_data import load_partial
from src.mainline_paths import GAUSSIAN_PREDICTION_FILENAME, REGISTERED_PRIOR_FILENAME, model_path
from src.moge_pixel_bridge import run_rmbg_mask, save_mask_png
from src.pointcloud_io import load_points
from src.partial_anchored_gaussian_edit import visible_axis_stretch
from src.saved_camera import SavedCameraProjector, draw_projection_overlay, world_to_camera_axes
from tools.qwen_image_edit import QwenImageEdit, resize_stage1_image_for_output
from src.visible_pixel_sim3_refinement import visible_pixel_pairs


SAMPLE = "01184"
REDWOOD10 = (
    "01184", "05117", "05452", "06127", "06145",
    "06188", "06830", "07136", "07306", "09639",
)
STATE_FILENAME = "agent_state.json"
INPUT_CONTEXT_FILENAME = "agent_input.json"


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
    gaussian_axis = root / "gaussian_axis" / SAMPLE
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
        "pixal_candidate": pixal / "gpt_clarity_candidate.png",
        "prior": pixal / "pixal3d_sampled_100k.ply",
        "pixal_input": pixal / "pixal3d_input.png",
        "pixal_metadata": pixal / "pixal3d_metadata.json",
        "pixal_cache": pixal / "pixal_moge_fp16_observation.npz",
        "registration": registration,
        "gaussian": gaussian,
        "gaussian_axis": gaussian_axis,
        "rescue": registration / "projection_rescue",
        "native": registration / "native",
        "bridge": registration / "bridge",
        "joint": registration / "joint",
        "amplified": registration / "amplified",
        "wide_tilt": registration / "wide_tilt",
        "final_registration": registration / "final",
        "joint_prior": registration / "joint" / "two_camera_joint_registered_100k.ply",
        "final_prior": registration / "final" / REGISTERED_PRIOR_FILENAME,
        "rescue_prior": registration / "projection_rescue" / "camera1_projection_rescue_registered_100k.ply",
        "prediction": gaussian / "decoded" / GAUSSIAN_PREDICTION_FILENAME,
        "axis_prediction": gaussian_axis / "decoded" / GAUSSIAN_PREDICTION_FILENAME,
        "diagnostics": root / "diagnostics",
        "final": root / "final" / "agent_selected_100k.ply",
    }


def _state_path(root: Path) -> Path:
    return root / STATE_FILENAME


def _input_context_path(root: Path) -> Path:
    return root / INPUT_CONTEXT_FILENAME


def _load_input_context(root: Path) -> dict[str, str]:
    """Load the sample-local input contract used by every probe tool.

    The original feasibility study used Redwood identifiers as an implicit
    dataset adapter.  Keeping the source partial and semantic label in the
    isolated state root makes the same bounded controller usable for another
    dataset without changing its geometric executors.
    """
    path = _input_context_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"missing probe input context: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = Path(str(payload.get("partial", "")))
    label = str(payload.get("object_label", "")).strip()
    if not source.is_file() or not label:
        raise ValueError(f"invalid probe input context: {path}")
    camera_root_raw = str(payload.get("camera_root", "")).strip()
    camera_root = Path(camera_root_raw).resolve() if camera_root_raw else None
    if camera_root is not None and not camera_root.is_dir():
        raise ValueError(f"invalid camera root in probe input context: {camera_root}")
    context = {"partial": str(source.resolve()), "object_label": label}
    if camera_root is not None:
        context["camera_root"] = str(camera_root)
    return context


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


def _prepare_view_candidates(root: Path, *, tool_budget: int) -> ProbeState:
    paths = _paths(root)
    if _state_path(root).exists():
        raise FileExistsError(f"probe already initialized: {_state_path(root)}")
    context = _load_input_context(root)
    source_partial = Path(context["partial"])
    _copy(source_partial, paths["partial"])
    points, colors = load_partial(paths["partial"])
    xyz = torch.from_numpy(points).to("cuda")
    rgb = torch.from_numpy(colors).to("cuda")

    # A dataset may provide the actual partial camera.  It is strictly more
    # faithful than re-rasterising the same scan, so preserve it verbatim as
    # the primary candidate.  Otherwise use the frozen partial-only 256-view
    # selection.  The yaw alternatives always derive from this base camera;
    # no semantic, prior, GT, or metric enters the choice.
    provided_camera_root = Path(context["camera_root"]) if "camera_root" in context else None
    required_camera_assets = (
        "raw_depth.png", "depth.png", "mask.png", "camera.pth", "point_uv.npy", "viewpoint.npy",
    )
    if provided_camera_root is not None:
        missing = [name for name in required_camera_assets if not (provided_camera_root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"provided partial camera is incomplete under {provided_camera_root}: {', '.join(missing)}"
            )
        base_eye = np.load(provided_camera_root / "viewpoint.npy").astype(np.float32)
    else:
        base_cfg = _load_cfg(paths["candidate_root"] / "base" / "camera")
        base_prompting = DepthPrompting(base_cfg)
        _, base_eye, _, _, _ = base_prompting._select_saved_view(xyz)
    candidates = {
        "base": None,
        "opposite_180": _yaw(base_eye, 180.0),
        "yaw_plus_30": _yaw(base_eye, 30.0),
        "yaw_minus_30": _yaw(base_eye, -30.0),
        "yaw_plus_60": _yaw(base_eye, 60.0),
        "yaw_minus_60": _yaw(base_eye, -60.0),
        "yaw_plus_90": _yaw(base_eye, 90.0),
        "yaw_minus_90": _yaw(base_eye, -90.0),
    }
    records: dict[str, dict[str, object]] = {}
    tiles: list[Image.Image] = []
    try:
        for name, override in candidates.items():
            output = paths["candidate_root"] / name / "camera"
            sample_dir = output / SAMPLE
            if name == "base" and provided_camera_root is not None:
                for asset in required_camera_assets:
                    _copy(provided_camera_root / asset, sample_dir / asset)
                raw_depth = sample_dir / "raw_depth.png"
                eye = base_eye
                source = "provided partial camera"
            else:
                cfg = _load_cfg(output)
                prompting = DepthPrompting(cfg)
                raw_depth = prompting._save_depth(
                    xyz, rgb, SAMPLE,
                    viewpoint_override=None if override is None else np.asarray(override, dtype=np.float32),
                )
                eye = np.load(sample_dir / "viewpoint.npy").astype(np.float32)
                source = "partial-only 256-view selection" if name == "base" else "partial-only fixed yaw around base"
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
                "source": source,
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
        "object_label": context["object_label"],
        "candidates": records,
    }
    manifest_path = paths["candidate_root"] / "candidate_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8")
    state = ProbeState(
        sample_id=SAMPLE,
        phase="view_candidates",
        budget_remaining=int(tool_budget),
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
    depth_image = Image.open(depth_path).convert("L")
    semantic_image = Image.open(semantic_path).convert("RGB")
    # The external clarity tool can return a different resolution from the
    # Camera-1 semantic observation.  Compare foreground evidence in the
    # fixed Camera-1 raster rather than requiring an accidental image size
    # match.
    if semantic_image.size != depth_image.size:
        semantic_image = semantic_image.resize(depth_image.size, Image.Resampling.LANCZOS)
    depth = np.asarray(depth_image, dtype=np.uint8) > 4
    semantic = np.asarray(semantic_image, dtype=np.uint8)
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


def _verifier_energy(visible: dict[str, object] | None) -> float | None:
    """Use one fixed no-GT score whenever two carriers must be compared."""
    if visible is None or int(visible["visible_pair_count"]) < 1:
        return None
    value = float(
        float(visible["visible_3d"])
        + .20 * (1.0 - float(visible["silhouette_iou"]))
        + .10 * (1.0 - float(visible["coverage"]))
        + .15 * float(visible["leakage"])
        + .25 * min(float(visible["depth_error_normalized"]), .20)
    )
    return value if math.isfinite(value) else None


def _next_untried_view(root: Path, state: ProbeState) -> str | None:
    """Choose a view proposal only from partial-only candidate evidence."""
    manifest_path = Path(str(state.artifacts.get("candidate_manifest", "")))
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tried = {item for item in str(state.artifacts.get("tried_view_candidates", "")).split(",") if item}
    candidates = payload.get("candidates", {})
    available = [
        (int(record.get("nonzero_depth_pixels", 0)), str(name))
        for name, record in candidates.items() if str(name) not in tried
    ]
    return max(available, default=(0, None))[1]


def _foreground_rank(metrics: object) -> tuple[float, float, float]:
    """Lexicographic Camera-1 support rank for semantic candidates."""
    if not isinstance(metrics, dict):
        return (-math.inf, -math.inf, -math.inf)
    try:
        return (
            float(metrics["iou"]),
            float(metrics["depth_coverage"]),
            -float(metrics["semantic_leakage"]),
        )
    except (KeyError, TypeError, ValueError):
        return (-math.inf, -math.inf, -math.inf)


def _agent_recommendation(
    root: Path,
    state: ProbeState,
    *,
    semantic: dict[str, float] | None,
    clarity: dict[str, object] | None,
    visible: dict[str, object] | None,
    surface_support: dict[str, object] | None,
    axis_scale: dict[str, object] | None,
    candidate_comparison: dict[str, object] | None,
) -> dict[str, object]:
    """Return an auditable proposal, not an automatic geometry decision.

    The high-level planner may accept or override this proposal, but can only
    choose an action that is valid for the current state.  Every branch below
    uses the partial, camera-aligned images, and registered carrier only.
    """
    phase = state.phase
    recommendation: dict[str, object] = {
        "source": "fixed_no_gt_diagnostic_policy",
        "must_be_verified_by": "state-bound MLLM or human planner",
        "ground_truth_cd_emd_used": False,
    }
    if phase == "view_candidates":
        candidate = _next_untried_view(root, state) or "base"
        recommendation.update({
            "action": SELECT_VIEW,
            "arguments": {"candidate": candidate},
            "hypothesis": "Choose a partial-derived Camera-1 before any semantic or prior module runs.",
        })
        return recommendation
    if phase == "semantic_pending":
        recommendation.update({
            "action": COMPLETE_SEMANTIC,
            "arguments": {"strategy": "qwen"},
            "hypothesis": "A camera-aligned semantic observation is required before prior generation.",
        })
        return recommendation
    if phase == "prior_pending":
        if semantic is None:
            recommendation.update({
                "action": GENERATE_PRIOR,
                "arguments": {},
                "hypothesis": "The semantic observation is unexpectedly missing; no legal image-repair action exists in this state, so the planner must inspect the recorded tool failure before consuming the prior tool.",
            })
        elif (
            float(semantic["iou"]) < .78
            or float(semantic["depth_coverage"]) < .88
            or float(semantic["semantic_leakage"]) > .18
        ):
            recommendation.update({
                "action": REFINE_SEMANTIC,
                "arguments": {"source": "REQUIRES_POSE_LOCKED_EXTERNAL_CLARITY_IMAGE"},
                "hypothesis": "The semantic foreground does not sufficiently explain the Camera-1 depth; prepare one pose-locked clarity/recompletion candidate before committing a 3-D prior.",
                "semantic_thresholds": {"iou_min": .78, "coverage_min": .88, "leakage_max": .18},
            })
        else:
            recommendation.update({
                "action": GENERATE_PRIOR,
                "arguments": {},
                "hypothesis": "The semantic foreground is camera-consistent enough to instantiate a native complete prior.",
            })
        return recommendation
    if phase == "prior_ready":
        recommendation.update({
            "action": GENERATE_PRIOR,
            "arguments": {},
            "hypothesis": "The one explicit clarity artifact is recorded; consume it rather than opening another image-edit branch.",
            "clarity_check": clarity,
        })
        return recommendation
    if phase == "conditioning_diagnosis":
        candidate_metrics = None
        if isinstance(clarity, dict):
            candidate_metrics = clarity.get("clarity_vs_depth_foreground")
        choose_clarity = _foreground_rank(candidate_metrics) > _foreground_rank(semantic)
        selected = "clarity" if choose_clarity else "qwen"
        recommendation.update({
            "action": SELECT_CONDITIONING_IMAGE,
            "arguments": {"source": selected},
            "hypothesis": (
                "The pose-locked clarity candidate has stronger Camera-1 foreground support than the Qwen observation."
                if choose_clarity else
                "The Qwen observation has equal or stronger Camera-1 foreground support; keep it as the prior-conditioning image."
            ),
            "qwen_vs_depth": semantic,
            "clarity_vs_depth": candidate_metrics,
        })
        return recommendation
    if phase == "global_alignment_pending":
        recommendation.update({
            "action": ALIGN_GLOBAL,
            "arguments": {},
            "hypothesis": "Estimate the native and cross-camera initialization before judging residual error.",
        })
        return recommendation

    pair_count = int(visible["visible_pair_count"]) if visible is not None else 0
    replan = _next_untried_view(root, state)
    if phase in {"alignment_diagnosis", "rescue_diagnosis", "adaptation_diagnosis"} and pair_count < 6:
        if phase == "alignment_diagnosis" and not str(state.artifacts.get("projection_rescued_prior", "")):
            recommendation.update({
                "action": RESCUE_GLOBAL,
                "arguments": {},
                "hypothesis": "The registered carrier has insufficient positive Camera-1 evidence; first attempt the bounded rendered 2-D center/scale recovery.",
            })
        elif replan is not None:
            recommendation.update({
                "action": REPLAN_VIEW,
                "arguments": {"candidate": replan},
                "hypothesis": "Residual geometry has insufficient positive evidence after the bounded recovery; restart from an untried partial-only view.",
            })
        else:
            recommendation.update({
                "action": ACCEPT,
                "arguments": {"candidate": "registered"},
                "hypothesis": "No supported repair action remains inside the fixed tool budget; preserve the complete registered carrier.",
            })
        return recommendation
    if phase in {"alignment_diagnosis", "rescue_diagnosis"}:
        if surface_support and surface_support["local_action_recommendation"] == "REPLAN_UPSTREAM" and replan is not None:
            recommendation.update({
                "action": REPLAN_VIEW,
                "arguments": {"candidate": replan},
                "hypothesis": "A substantial observed surface lies outside the local trust region, indicating an upstream observation or prior mismatch rather than a local edit.",
            })
        else:
            recommendation.update({
                "action": REFINE_ALIGNMENT,
                "arguments": {},
                "hypothesis": "Positive visible support exists, so run the fixed narrow Camera-1 Sim(3) continuation before any local adaptation.",
            })
        return recommendation
    if phase == "adaptation_diagnosis":
        if isinstance(axis_scale, dict) and bool(axis_scale.get("active", False)):
            recommendation.update({
                "action": ADAPT_AXIS_SCALE,
                "arguments": {},
                "hypothesis": "The observed Camera-1 surface supports a relative-axis extent correction; apply its coherent whole-carrier initialization before the fixed local Gaussian edit.",
                "camera_axis_evidence": axis_scale,
            })
        elif surface_support and surface_support["local_action_recommendation"] == "ADAPT_LOCAL":
            recommendation.update({
                "action": ADAPT_LOCAL,
                "arguments": {},
                "hypothesis": "The remaining observed error is localized inside the fixed partial-anchored Gaussian trust region.",
            })
        elif surface_support and surface_support["local_action_recommendation"] == "REPLAN_UPSTREAM" and replan is not None:
            recommendation.update({
                "action": REPLAN_VIEW,
                "arguments": {"candidate": replan},
                "hypothesis": "The visible error exceeds the local adaptation trust region; do not hide it with deformation.",
            })
        else:
            recommendation.update({
                "action": ACCEPT,
                "arguments": {"candidate": "registered"},
                "hypothesis": "The registered prior already explains the observed surface sufficiently; preserve its complete support.",
            })
        return recommendation
    if phase in {"final_diagnosis", "axis_scale_diagnosis"}:
        winner = str((candidate_comparison or {}).get("recommended_candidate", "current"))
        if phase == "axis_scale_diagnosis" and winner == "registered":
            winner = "adapted"
        recommendation.update({
            "action": ACCEPT,
            "arguments": {"candidate": winner},
            "hypothesis": str((candidate_comparison or {}).get(
                "reason", "Compare the edited and registered carriers with the same Camera-1 verifier before accepting either one."
            )),
        })
        return recommendation
    recommendation.update({
        "action": ACCEPT,
        "arguments": {},
        "hypothesis": "No further bounded tool is valid for this state.",
    })
    return recommendation


def _current_prior(paths: dict[str, Path]) -> Path | None:
    # A rescue is an initializer for the normal residual solver, not its final
    # output.  Once a refined Camera-1 candidate exists it must be the active
    # diagnostic/acceptance candidate; otherwise the agent would silently
    # evaluate and export the coarser rescue again.
    for path in (paths["axis_prediction"], paths["prediction"], paths["final_prior"], paths["rescue_prior"], paths["joint_prior"]):
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
    clarity_manifest = paths["pixal"] / "gpt_clarity_action.json"
    clarity = None
    if clarity_manifest.is_file():
        try:
            clarity = json.loads(clarity_manifest.read_text(encoding="utf-8")).get("verification")
        except (OSError, json.JSONDecodeError):
            clarity = {"status": "unreadable_manifest"}
    # Once an ACCEPT action has copied a verifier-selected carrier, inspect
    # that exact frozen output.  During intermediate states use the active
    # executor slot so the next tool can still consume its intended input.
    prior_path = paths["final"] if state.phase == "accepted" and paths["final"].is_file() else _current_prior(paths)
    visible = None
    extent = None
    surface_support = None
    axis_scale = None
    overlay = None
    registered_reference = None
    registered_overlay = None
    partial = None
    projector = None
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
        # Sim(3) has already consumed the global gauge at this stage.  Test
        # only a relative, partial-supported axis residual; this is evidence
        # for the discrete local-adaptation tool, not an unconstrained warp.
        scale_reference = load_points(paths["final_prior"]) if paths["final_prior"].is_file() else prior
        saved_pairs, _ = visible_pixel_pairs(
            partial, scale_reference, projector, max_pixel_distance=1.0,
            max_pairs=min(len(partial), len(scale_reference)),
        )
        _, axis_scale = visible_axis_stretch(
            scale_reference, partial, saved_pairs, max_anchor_residual=.075 * diagonal,
            axis_basis=world_to_camera_axes(projector.camera), minimum_anisotropy=.025,
            minimum_residual_reduction=.03,
        )
        if paths["semantic"].is_file():
            overlay = directory / "camera1_overlay.png"
            draw_projection_overlay(overlay, paths["semantic"], partial, prior, projector)
    protection = None
    applied_axis_scale = None
    axis_info = paths["gaussian_axis"] / "camera_axis_scale" / "camera_axis_scale_info.json"
    if prior_path == paths["axis_prediction"] and axis_info.is_file():
        try:
            applied_axis_scale = json.loads(axis_info.read_text(encoding="utf-8")).get("estimate")
        except (OSError, json.JSONDecodeError):
            applied_axis_scale = {"active": False, "reason": "unreadable_axis_edit_record"}
    if prior_path in {paths["prediction"], paths["axis_prediction"]} and paths["final_prior"].is_file():
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
        energy = _verifier_energy(visible)
    if (
        prior_path in {paths["prediction"], paths["axis_prediction"]}
        and paths["final_prior"].is_file()
        and partial is not None
        and projector is not None
    ):
        # Camera-1 silhouette/depth is an important diagnostic for pose, but
        # it is not the deciding score for a supported relative-axis shape
        # scale: a valid depth-axis correction can intentionally alter the
        # rendered outline while reducing the fixed visible 3-D scale fit.
        # The scale executor already requires sufficient pixel-indexed
        # support, independent axes, anisotropy, and a trimmed residual drop.
        # Therefore this branch uses that direct no-GT scale evidence as the
        # selection contract and retains the rendered score for audit only.
        reference_path = (
            paths["prediction"] if prior_path == paths["axis_prediction"] and paths["prediction"].is_file()
            else paths["final_prior"]
        )
        registered = load_points(reference_path)
        diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
        reference_visible = _compact_visible(
            visible_score(partial, registered, projector, diagonal, pixel_radius=5.0)
        )
        reference_energy = _verifier_energy(reference_visible)
        # A tie intentionally protects the reference 100k carrier.  This is
        # not a GT gate: both candidates are evaluated in the same observed
        # Camera-1 frame.  In particular, an axis-scale candidate must lower
        # the verifier versus the ordinary Gaussian candidate; partial-only
        # evidence never authorizes a merely neutral deformation of the
        # unobserved complete prior.
        axis_supported = bool(
            prior_path == paths["axis_prediction"]
            and isinstance(applied_axis_scale, dict)
            and applied_axis_scale.get("active", False)
        )
        prefer_reference = not axis_supported
        if not axis_supported:
            reason = "The Camera-1 axis estimate is unsupported; retain the ordinary Gaussian carrier without introducing a no-op duplicate."
        else:
            reason = "The partial-supported relative-axis scale satisfies its direct Camera-1 visible 3-D evidence contract; retain the coherent 100k carrier and use the rendered score as an auditable diagnostic."
        registered_reference = {
            "registered_prior_vs_partial": reference_visible,
            "reference_candidate": "ordinary_gaussian" if reference_path == paths["prediction"] else "registered_prior",
            "verifier_energy": reference_energy,
            "recommended_candidate": "registered" if prefer_reference else "current",
            "reason": reason,
            "selection_evidence": "camera_axis_scale_fit" if axis_supported else "ordinary_gaussian",
            "axis_scale_evidence": applied_axis_scale,
            "ground_truth_cd_emd_used": False,
        }
        if paths["semantic"].is_file():
            registered_overlay = directory / "registered_camera1_overlay.png"
            draw_projection_overlay(registered_overlay, paths["semantic"], partial, registered, projector)
    recommendation = _agent_recommendation(
        root,
        state,
        semantic=semantic,
        clarity=clarity,
        visible=visible,
        surface_support=surface_support,
        axis_scale=axis_scale,
        candidate_comparison=registered_reference,
    )
    report = {
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "state_phase": state.phase,
        "current_prior": str(prior_path.resolve()) if prior_path is not None else None,
        "semantic_vs_depth": semantic,
        "pixal_clarity_check": clarity,
        "registered_prior_vs_partial": visible,
        "extent": extent,
        "observed_surface_support": surface_support,
        "candidate_visible_axis_scale": axis_scale,
        "applied_visible_axis_scale": applied_axis_scale,
        "prior_protection": protection,
        "edited_vs_registered": registered_reference,
        "agent_recommendation": recommendation,
        "verifier_energy": energy,
        "diagnostic_board": None,
    }
    tiles = [
        _depth_tile(paths["depth"], "Camera-1 depth"),
        _depth_tile(paths["semantic"], "semantic observation"),
        _depth_tile(paths["pixal_semantic"], "prior conditioning image"),
        _depth_tile(paths["pixal_candidate"], "clarity candidate"),
        _depth_tile(paths["native"] / "pixal_native_moge_projection.png", "native prior / image geometry"),
        _depth_tile(paths["joint"] / "two_camera_joint_saved_view_projection.png", "global alignment"),
        _depth_tile(paths["final_registration"] / "camera1_amplified_saved_view_projection.png", "residual alignment"),
        _depth_tile(overlay if overlay is not None else Path(), "current partial/prior overlay"),
        _depth_tile(registered_overlay if registered_overlay is not None else Path(), "registered reference overlay"),
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
- No-GT diagnostic recommendation (may be overridden only with evidence from this state):
  `{json.dumps(state.diagnostics.get('agent_recommendation', {}), ensure_ascii=False)}`

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

- `SELECT_VIEW` and `REPLAN_VIEW`: `{{"candidate": "<one name from candidate_manifest.json>"}}`
- `RESCUE_GLOBAL`, `RESTORE_ATTEMPT`, `GENERATE_PRIOR`, `ALIGN_GLOBAL`, `REFINE_ALIGNMENT`, `ADAPT_LOCAL`, and `ADAPT_AXIS_SCALE`:
  `{{}}`.
- `ACCEPT`: `{{}}`, or `{{"candidate": "rescue"}}` / `{{"candidate":
  "registered"}}` / `{{"candidate": "baseline"}}` / `{{"candidate":
  "adapted"}}` / `{{"candidate": "axis"}}` when the fresh verifier finds
  that a later tool result is weaker than the bounded rescue / pre-edit
  registered candidate / replayed initial candidate / ordinary Gaussian
  candidate / camera-axis candidate.
- `COMPLETE_SEMANTIC`: `{{"strategy": "qwen"}}`
- `REFINE_SEMANTIC`: `{{"source": "/absolute/path/to/pose_locked_external_clarity.png"}}`.
  The supplied image is recorded and checked against the Camera-1 semantic
  observation before it becomes a candidate prior-conditioning image.
- `SELECT_CONDITIONING_IMAGE`: `{{"source": "qwen" | "clarity"}}`.
  The selected image is copied to the fixed Pixal input slot; no geometry is
  changed by this action.
"""
    path.write_text(packet, encoding="utf-8")


def _commit_view(root: Path, candidate: str) -> dict[str, str]:
    paths = _paths(root)
    source = paths["candidate_root"] / candidate / "camera" / SAMPLE
    manifest = json.loads((paths["candidate_root"] / "candidate_manifest.json").read_text(encoding="utf-8"))
    if candidate not in manifest.get("candidates", {}):
        raise ValueError(f"unknown bounded view candidate: {candidate}")
    for name in ("raw_depth.png", "depth.png", "mask.png", "point_uv.npy", "viewpoint.npy", "camera.pth"):
        _copy(source / name, paths["camera"] / name)
    return {"candidate": candidate, "camera": str((paths["camera"] / "camera.pth").resolve()),
            "depth": str((paths["camera"] / "depth.png").resolve())}


def _archive_and_reset_for_replan(root: Path, state: ProbeState, candidate: str) -> dict[str, str]:
    """Archive an unsupported attempt before switching its bounded view.

    A replan never invents a new camera. It selects an untried candidate
    prepared from the partial alone at initialization, then removes stale
    semantic/prior/registration artifacts so no executor can resume the
    rejected observation. The compact archive makes the failure auditable.
    """
    paths = _paths(root)
    manifest = json.loads((paths["candidate_root"] / "candidate_manifest.json").read_text(encoding="utf-8"))
    allowed = set(manifest.get("candidates", {}))
    current = str(state.artifacts.get("candidate", ""))
    if candidate not in allowed:
        raise ValueError(f"unknown bounded view candidate: {candidate}")
    if candidate == current:
        raise ValueError("REPLAN_VIEW must select a previously untried view candidate")
    tried = [item for item in str(state.artifacts.get("tried_view_candidates", "")).split(",") if item]
    if candidate in tried:
        raise ValueError(f"view candidate already attempted: {candidate}")

    archive = root / "replan_attempts" / f"attempt_{len(tried):02d}_{current or 'unknown'}"
    archive.mkdir(parents=True, exist_ok=True)
    _copy(_state_path(root), archive / "state_before_replan.json")
    verifier = paths["diagnostics"] / f"step_{len(state.history):02d}_{state.phase}" / "verifier.json"
    if verifier.is_file():
        _copy(verifier, archive / "verifier.json")
    board = Path(str(state.artifacts.get("diagnostic_board", "")))
    if board.is_file():
        _copy(board, archive / "diagnostic_board.png")
    carrier = _current_prior(paths)
    if carrier is not None:
        _copy(carrier, archive / carrier.name)
    (archive / "replan.json").write_text(json.dumps({
        "from_candidate": current or None,
        "to_candidate": candidate,
        "reason": state.diagnostics.get("observed_surface_support", {}).get("recommendation_reason"),
        "ground_truth_cd_emd_used": False,
    }, indent=2) + "\n", encoding="utf-8")

    for path in (paths["camera"], paths["pixal"], paths["registration"], paths["gaussian"]):
        if path.exists():
            shutil.rmtree(path)
    if paths["final"].exists():
        paths["final"].unlink()
    outputs = _commit_view(root, candidate)
    tried.append(candidate)
    outputs.update({
        "candidate": candidate,
        "tried_view_candidates": ",".join(tried),
        "replan_archive": str(archive.resolve()),
    })
    return outputs


def _restore_last_replan_attempt(root: Path, state: ProbeState) -> dict[str, str]:
    """Select the preceding finite attempt rather than accepting a worse retry."""
    paths = _paths(root)
    archive = Path(str(state.artifacts.get("replan_archive", "")))
    if not archive.is_dir():
        raise FileNotFoundError("RESTORE_ATTEMPT requires an archived replan attempt")
    carriers = sorted(archive.glob("*registered_100k.ply"))
    if len(carriers) != 1:
        raise RuntimeError(f"expected one archived registered carrier under {archive}, found {len(carriers)}")
    _copy(carriers[0], paths["final"])
    restored_candidate = None
    state_before = archive / "state_before_replan.json"
    if state_before.is_file():
        restored_candidate = json.loads(state_before.read_text(encoding="utf-8")).get("artifacts", {}).get("candidate")
    return {
        "accepted_prediction": str(paths["final"].resolve()),
        "selected_source": str(carriers[0].resolve()),
        "restored_attempt": str(archive.resolve()),
        "candidate": str(restored_candidate or "archived"),
    }


def _rescue_global(root: Path) -> dict[str, str]:
    """Recover only translation/scale when Camera-1 has no positive pairs."""
    paths = _paths(root)
    source = _current_prior(paths)
    if source is None:
        raise FileNotFoundError("RESCUE_GLOBAL requires a registered prior")
    _run([
        sys.executable, "scripts/run_camera1_projection_sim3_rescue.py",
        "--partial", str(paths["partial"]), "--registered-prior", str(source),
        "--camera", str(paths["camera_pth"]), "--semantic", str(paths["semantic"]),
        "--output-dir", str(paths["rescue"]), "--device", "cpu",
    ], log=paths["rescue"] / "stage.log")
    return {"projection_rescued_prior": str(paths["rescue_prior"].resolve())}


def _complete_semantic(
    root: Path,
    strategy: str,
    *,
    shared_editor: QwenImageEdit | None = None,
) -> dict[str, str]:
    if strategy != "qwen":
        raise ValueError("the first probe supports only the deterministic Qwen executor")
    paths = _paths(root)
    # Every expensive tool is an idempotent transaction at the agent boundary.
    # If an external scheduler interrupts after Qwen/RMBG wrote their durable
    # artifacts but before the state file is committed, replaying the same
    # state-bound decision must resume rather than spend a second generation.
    if paths["semantic"].is_file() and paths["source_mask"].is_file():
        try:
            with Image.open(paths["semantic"]) as image:
                image.convert("RGB").load()
            with Image.open(paths["source_mask"]) as mask:
                mask.convert("L").load()
        except Exception:
            pass
        else:
            _copy(paths["semantic"], paths["pixal_semantic"])
            return {
                "semantic": str(paths["semantic"].resolve()),
                "source_mask": str(paths["source_mask"].resolve()),
                "pixal_input_semantic": str(paths["pixal_semantic"].resolve()),
                "reused_interrupted_tool_artifacts": "true",
            }
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
        semantic = editor.generate(raw_depth, _load_input_context(root)["object_label"], size=int(cfg.generate_res))
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


def _clarity_prompt(label: str) -> str:
    return "\n".join((
        "Use case: precise-object-edit",
        "Asset type: Pixal3D conditioning image",
        "",
        f"Improve only visual clarity, plausible material detail, sharpness, and the clean pure-white background of the displayed {label}.",
        "",
        "Strict geometry constraints: preserve exactly the input camera view, object orientation, apparent scale, center/crop, silhouette, perspective, and every visible component. "
        f"Do not rotate, mirror, recrop, resize, add/remove/reshape any part, or alter the pose. Keep one complete {label}.",
        "",
        "Avoid: text, labels, logos, extra objects, duplicate parts, changed viewpoint, or shadows obscuring the outline.",
    ))


def _refine_semantic(root: Path, source_value: str) -> dict[str, str]:
    """Install an external pose-locked GPT clarity artifact for Pixal only.

    The image-generation service remains an explicit agent tool rather than a
    hidden credential-bearing Python dependency.  Camera-1 keeps the Qwen
    observation and its original mask; only Camera-2/Pixal receives the
    clarity image, exactly as in the frozen mainline contract.
    """
    paths = _paths(root)
    source = Path(str(source_value)).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"REFINE_SEMANTIC requires a non-empty external image: {source}")
    if not paths["semantic"].is_file():
        raise FileNotFoundError("REFINE_SEMANTIC requires the Qwen semantic observation")
    try:
        with Image.open(source) as image:
            image.convert("RGB").load()
    except Exception as exc:
        raise ValueError(f"invalid external clarity image: {source}") from exc
    _copy(source, paths["pixal_candidate"])
    prompt = _clarity_prompt(_load_input_context(root)["object_label"])
    prompt_path = paths["pixal"] / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    semantic_metrics = _semantic_iou(paths["semantic"], paths["pixal_candidate"])
    depth_metrics = _semantic_iou(paths["depth"], paths["pixal_candidate"])
    manifest = {
        "method": "external_pose_locked_gpt_clarity_agent_tool",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "camera1_observation": str(paths["semantic"].resolve()),
        "external_clarity_source": str(source),
        "clarity_candidate_image": str(paths["pixal_candidate"].resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompt": prompt,
        "verification": {
            "camera1_vs_clarity_foreground": semantic_metrics,
            "clarity_vs_depth_foreground": depth_metrics,
            "policy": "recorded for the next state-bound planner decision; no ground truth or metric is used",
        },
    }
    manifest_path = paths["pixal"] / "gpt_clarity_action.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "clarity_candidate_semantic": str(paths["pixal_candidate"].resolve()),
        "gpt_clarity_manifest": str(manifest_path.resolve()),
        "gpt_clarity_prompt": str(prompt_path.resolve()),
    }


def _select_conditioning_image(root: Path, source_name: str) -> dict[str, str]:
    """Commit one observed-image candidate to the fixed Pixal input slot."""
    paths = _paths(root)
    options = {"qwen": paths["semantic"], "clarity": paths["pixal_candidate"]}
    if source_name not in options:
        raise ValueError(f"unknown conditioning-image source: {source_name}")
    source = options[source_name]
    if not source.is_file():
        raise FileNotFoundError(source)
    _copy(source, paths["pixal_semantic"])
    selection = {
        "selected_source": source_name,
        "source": str(source.resolve()),
        "pixal_conditioning_image": str(paths["pixal_semantic"].resolve()),
        "ground_truth_cd_emd_used": False,
    }
    selection_path = paths["pixal"] / "conditioning_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return {
        "selected_conditioning_source": source_name,
        "pixal_input_semantic": str(paths["pixal_semantic"].resolve()),
        "conditioning_selection": str(selection_path.resolve()),
    }


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
    source = paths["rescue_prior"] if paths["rescue_prior"].is_file() else paths["joint_prior"]
    common = ["--partial", str(paths["partial"]), "--camera", str(paths["camera_pth"]),
              "--semantic", str(paths["semantic"]), "--search-points", "32000", "--device", "cpu"]
    _run([
        sys.executable, "scripts/run_camera1_amplified_sim3_refine.py", "--registered-prior", str(source),
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
    # Keep the same fixed .075 trust region as the accepted mainline.  The
    # agent decides whether local adaptation is warranted; it must not silently
    # weaken the deterministic executor and thereby lose observed geometry.
    _run([
        sys.executable, "scripts/run_mainline_gaussian.py", "--root", str(root),
        "--registration-root", str(paths["registration"].parent), "--samples", SAMPLE,
    ], log=paths["root"] / "gaussian" / "agent_tool.log")
    return {"adapted_prediction": str(paths["prediction"].resolve())}


def _adapt_axis_scale(root: Path) -> dict[str, str]:
    """Apply one supported whole-carrier scale, then the fixed local edit."""
    paths = _paths(root)
    _run([
        sys.executable, "scripts/run_mainline_gaussian.py", "--root", str(root),
        "--registration-root", str(paths["registration"].parent), "--samples", SAMPLE,
        "--gaussian-subdir", "gaussian_axis", "--camera-axis-scale",
    ], log=paths["root"] / "gaussian_axis" / "agent_tool.log")
    return {"axis_scale_prediction": str(paths["axis_prediction"].resolve())}


def _accept(root: Path, candidate: str = "current") -> dict[str, str]:
    paths = _paths(root)
    if candidate == "current":
        source = _current_prior(paths)
    elif candidate == "rescue":
        source = paths["rescue_prior"]
    elif candidate == "registered":
        source = paths["final_prior"]
    elif candidate == "baseline":
        source = paths["joint_prior"]
    elif candidate == "adapted":
        source = paths["prediction"]
    elif candidate == "axis":
        source = paths["axis_prediction"]
    else:
        raise ValueError(f"unsupported ACCEPT candidate: {candidate}")
    if source is None:
        raise FileNotFoundError("cannot accept before a registered prior exists")
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    _copy(source, paths["final"])
    return {
        "accepted_prediction": str(paths["final"].resolve()),
        "selected_source": str(source.resolve()),
        "accepted_candidate": candidate,
    }


def _execute(root: Path, state: ProbeState, decision: AgentDecision) -> tuple[str, dict[str, str]]:
    action, arguments = decision.action, decision.arguments
    if action == SELECT_VIEW:
        outputs = _commit_view(root, str(arguments.get("candidate", "")))
        outputs["tried_view_candidates"] = str(arguments.get("candidate", ""))
        return "semantic_pending", outputs
    if action == REPLAN_VIEW:
        return "semantic_pending", _archive_and_reset_for_replan(root, state, str(arguments.get("candidate", "")))
    if action == RESTORE_ATTEMPT:
        return "accepted", _restore_last_replan_attempt(root, state)
    if action == COMPLETE_SEMANTIC:
        return "prior_pending", _complete_semantic(root, str(arguments.get("strategy", "")))
    if action == REFINE_SEMANTIC:
        return "conditioning_diagnosis", _refine_semantic(root, str(arguments.get("source", "")))
    if action == SELECT_CONDITIONING_IMAGE:
        return "prior_ready", _select_conditioning_image(root, str(arguments.get("source", "")))
    if action == GENERATE_PRIOR:
        return "global_alignment_pending", _generate_prior(root)
    if action == ALIGN_GLOBAL:
        return "alignment_diagnosis", _align_global(root)
    if action == RESCUE_GLOBAL:
        return "rescue_diagnosis", _rescue_global(root)
    if action == REFINE_ALIGNMENT:
        return "adaptation_diagnosis", _refine_alignment(root)
    if action == ADAPT_LOCAL:
        return "final_diagnosis", _adapt_local(root)
    if action == ADAPT_AXIS_SCALE:
        return "axis_scale_diagnosis", _adapt_axis_scale(root)
    if action == ACCEPT:
        return "accepted", _accept(root, str(arguments.get("candidate", "current")))
    raise AssertionError(action)


def initialize(args: argparse.Namespace) -> None:
    root = _resolve_probe_root(args.root)
    if int(args.tool_budget) <= 0:
        raise ValueError("--tool-budget must be positive")
    source = args.partial.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    label = str(args.object_label).strip()
    if not label:
        raise ValueError("--object-label must be non-empty")
    context_path = _input_context_path(root)
    if context_path.exists():
        raise FileExistsError(f"probe input context already exists: {context_path}")
    context_path.parent.mkdir(parents=True, exist_ok=True)
    camera_root = args.camera_root.resolve() if args.camera_root is not None else None
    if camera_root is not None and not camera_root.is_dir():
        raise FileNotFoundError(camera_root)
    context = {
        "partial": str(source), "object_label": label,
        "dataset_adapter": "explicit_sample_local_input_contract",
        "ground_truth_cd_emd_used": False,
    }
    if camera_root is not None:
        context["camera_root"] = str(camera_root)
    context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    state = _prepare_view_candidates(root, tool_budget=int(args.tool_budget))
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


def recommend(args: argparse.Namespace) -> None:
    """Print the current no-GT diagnostic proposal without changing state."""
    state = ProbeState.from_file(_state_path(_resolve_probe_root(args.root)))
    print(json.dumps({
        "state_sha256": state.state_hash(),
        "phase": state.phase,
        "allowed_actions": state.as_dict()["allowed_actions"],
        "agent_recommendation": state.diagnostics.get("agent_recommendation"),
    }, indent=2, ensure_ascii=False))


def refresh(args: argparse.Namespace) -> None:
    """Recompute only verifier evidence after an interrupted external tool."""
    root = _resolve_probe_root(args.root)
    state = ProbeState.from_file(_state_path(root))
    _write_diagnostics(root, state)
    state.write(_state_path(root))
    _write_decision_packet(root, state)
    print(json.dumps(state.as_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    global SAMPLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Per-sample isolated output root for this probe (for example, workspace/probe/01184).")
    parser.add_argument("--sample", default=SAMPLE,
                        help="Stable identifier for this isolated probe.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("initialize", help="Generate partial-only view candidates and first state.")
    init.add_argument("--partial", type=Path, required=True,
                      help="Input partial PLY; never replaced by a dataset-specific default.")
    init.add_argument("--object-label", required=True,
                      help="Dataset-provided semantic label used only by the completion prompt.")
    init.add_argument("--camera-root", type=Path,
                      help="Optional directory containing this partial scan's raw_depth.png, depth.png, camera.pth, point_uv.npy, and viewpoint.npy. When supplied, the agent preserves this Camera-1 exactly as its base candidate.")
    init.add_argument("--tool-budget", type=int, default=14,
                      help="Maximum state-bound tool calls for this isolated probe (default: 14).")
    init.set_defaults(handler=initialize)
    apply_parser = subparsers.add_parser("apply", help="Execute one state-bound MLLM decision.")
    apply_parser.add_argument("--decision", type=Path, required=True)
    apply_parser.set_defaults(handler=apply)
    show_parser = subparsers.add_parser("show", help="Print the latest state without executing a tool.")
    show_parser.set_defaults(handler=show)
    recommend_parser = subparsers.add_parser(
        "recommend", help="Print the state-bound no-GT action proposal without executing it.",
    )
    recommend_parser.set_defaults(handler=recommend)
    refresh_parser = subparsers.add_parser(
        "refresh", help="Recompute no-GT diagnostics without consuming an agent tool budget.",
    )
    refresh_parser.set_defaults(handler=refresh)
    args = parser.parse_args()
    SAMPLE = str(args.sample)
    args.handler(args)


if __name__ == "__main__":
    main()
