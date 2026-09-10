"""Orchestrate the scene wrapper without altering object-level runners."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

from PIL import Image, ImageOps

from src.moge_pixel_bridge import run_rmbg_mask, save_mask_png
from src.scene_completion.contracts import SceneManifest


def write_gpt_action_manifest(root: Path, manifest: SceneManifest) -> Path:
    """Persist the direct agent-mediated GPT semantic-completion contract.

    The Codex built-in image-generation tool is intentionally not imported by
    a Python batch runner: it is an interactive image agent, not a hidden
    model dependency.  This manifest makes every image action reproducible and
    lets the runner refuse to continue until the saved `gpt_image.png` assets
    exist.
    """
    root = Path(root).resolve()
    actions = []
    for item in manifest.instances:
        instance_root = root / "instances" / item.instance_id
        direct_output = root / "gpt_outputs" / f"{item.instance_id}.png"
        semantic_target = root / "inputs" / "camera" / item.instance_id / "img.png"
        pixal_target = root / "inputs" / "pixal" / item.instance_id / "gpt_image.png"
        actions.append({
            "id": item.instance_id,
            "label": item.label,
            "mask_input": str((root / "masks" / f"{item.instance_id}.png").resolve()),
            "mask_task": {
                "input": str(manifest.source_image),
                "output": str((root / "masks" / f"{item.instance_id}.png").resolve()),
                "prompt": (
                    f"Segment exactly one visible {item.label} in this scene. Preserve the source image "
                    "pixel grid and output a binary mask at the exact same resolution: pure white for "
                    f"pixels belonging to this {item.label}, pure black for everything else. Do not include "
                    "shadows, floor, wall, adjacent objects, or background. Do not redraw the object."
                ),
            },
            "completion_task": {
                "inputs": {
                    "masked_scene_crop": str((instance_root / "masked_crop.png").resolve()),
                },
                "output": str(direct_output.resolve()),
                "install_targets": {
                    "camera1_semantic": str(semantic_target.resolve()),
                    "pixal_input": str(pixal_target.resolve()),
                },
                "prompt": (
                    f"Starting only from this isolated RGB scene crop, produce exactly one complete {item.label} "
                    "on a pure white square background. POSE LOCK: the crop is the only camera reference. Preserve "
                    "the exact observed object direction, left/right relation, yaw, pitch, roll, image-plane angle, "
                    "perspective/foreshortening, silhouette, aspect ratio, material, colour, and visible structural "
                    "details. Complete only genuinely occluded or missing portions behind the observed view. Never "
                    "canonicalize it into a frontal, side, top-down, symmetric, catalog, or product-shot viewpoint; do "
                    "not rotate, mirror, re-pose, resize disproportionately, or redesign it. Do not output a depth map "
                    "or add props, text, people, floor, wall, shadows, or a second object."
                ),
            },
        })
    payload = {
        "method": "codex_gpt_image2_instance_mask_and_direct_semantic_completion",
        "interactive_agent_stage": True,
        "depth_to_semantic_used": False,
        "source_image": str(manifest.source_image),
        "instances": actions,
        "continuation_contract": (
            "After each GPT action, save the direct semantic completion to the declared gpt_outputs path. "
            "The wrapper copies the same frozen image into both Camera-1 and Pixal contracts before "
            "deterministic Pixal generation, registration, and textured-mesh scene composition."
        ),
    }
    path = root / "gpt_image_actions.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _square_semantic_image(source: Path, target: Path, *, size: int) -> None:
    """Preserve aspect ratio while materialising the shared square image contract."""
    if int(size) < 64:
        raise ValueError("semantic image size must be at least 64")
    image = Image.open(source).convert("RGB")
    content = ImageOps.contain(image, (int(size), int(size)), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (int(size), int(size)), "white")
    canvas.paste(content, ((int(size) - content.width) // 2, (int(size) - content.height) // 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def install_gpt_images(
    root: Path,
    sample_ids: Iterable[str],
    source_root: Path,
    *,
    rmbg_model: Path,
    overwrite: bool,
    semantic_size: int = 512,
) -> None:
    """Install one direct GPT completion as both Camera-1 and Pixal semantic input."""
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    for sample in sample_ids:
        source = source_root / f"{sample}.png"
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing GPT image for {sample}: {source}. See {root / 'gpt_image_actions.json'}"
            )
        camera_root = root / "inputs" / "camera" / sample
        semantic = camera_root / "img.png"
        target = root / "inputs" / "pixal" / sample / "gpt_image.png"
        source_mask = camera_root / f"{sample}_moge_to_raw_partial_object_mask.png"
        existing = (semantic.exists(), target.exists(), source_mask.exists())
        if any(existing) and not overwrite:
            if all(existing):
                continue
            raise FileExistsError(
                f"Refusing to mix a partial direct-GPT install for {sample}; use --overwrite-gpt-images"
            )
        _square_semantic_image(source, semantic, size=semantic_size)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(semantic, target)
        alpha = run_rmbg_mask(semantic, camera_root / "img_rmbg.png", Path(rmbg_model))
        save_mask_png(source_mask, alpha)


def run_command(command: list[str], *, root: Path, log: Path, dry_run: bool) -> None:
    """Run an unchanged object-level public entry point with a scene log."""
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"scene substage failed; inspect {log}")


def pixal_command(
    *, root: Path, manifest: SceneManifest, decimation_target: int, overwrite: bool = False,
) -> tuple[list[str], Path]:
    if int(decimation_target) < 1_000:
        raise ValueError("scene Pixal decimation target must be at least 1,000 faces")
    command = [
        sys.executable, "scripts/run_pixal3d_gpt_batch.py",
        "--input-root", str(root / "inputs" / "pixal"),
        "--output-root", str(root / "inputs" / "pixal"),
        "--input-name", "gpt_image.png", "--ids", *manifest.instance_ids,
        "--decimation-target", str(int(decimation_target)),
    ]
    if overwrite:
        command.append("--overwrite")
    return command, root / "logs" / "pixal.log"


def registration_command(
    *, root: Path, manifest: SceneManifest, sample_workers: int = 1, bridge_only: bool = False,
) -> tuple[list[str], Path]:
    if int(sample_workers) < 1:
        raise ValueError("sample_workers must be positive")
    command = [
        sys.executable, "scripts/run_fixed_pixal_moge_registration.py",
        "--samples", *manifest.instance_ids,
        "--pixal-root", str(root / "inputs" / "pixal"),
        "--camera-root", str(root / "inputs" / "camera"),
        "--partial-root", str(root / "inputs" / "partial"),
        "--output-root", str(root / "registration"),
        "--sample-workers", str(int(sample_workers)),
    ]
    if bridge_only:
        command.append("--bridge-only")
    return command, root / "logs" / "registration.log"
