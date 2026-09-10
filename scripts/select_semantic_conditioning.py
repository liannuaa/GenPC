#!/usr/bin/env python3
"""Select a camera-consistent semantic image without using ground truth.

Each sample exposes the freshly generated Qwen observation and, optionally,
one pose-locked clarity candidate.  The selector compares their foregrounds
against the saved Camera-1 depth support and installs the lexicographically
better image as ``gpt_image.png`` for prior generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _foreground_metrics(depth_path: Path, image_path: Path) -> dict[str, float]:
    depth_image = Image.open(depth_path).convert("L")
    semantic_image = Image.open(image_path).convert("RGB")
    if semantic_image.size != depth_image.size:
        semantic_image = semantic_image.resize(depth_image.size, Image.Resampling.LANCZOS)
    depth = np.asarray(depth_image, dtype=np.uint8) > 4
    semantic = np.asarray(semantic_image, dtype=np.uint8)
    foreground = np.min(semantic, axis=-1) < 245
    union = int(np.count_nonzero(depth | foreground))
    return {
        "iou": float(np.count_nonzero(depth & foreground) / max(union, 1)),
        "depth_coverage": float(
            np.count_nonzero(depth & foreground) / max(int(np.count_nonzero(depth)), 1)
        ),
        "semantic_leakage": float(
            np.count_nonzero(foreground & ~depth) / max(int(np.count_nonzero(foreground)), 1)
        ),
    }


def _rank(metrics: dict[str, float]) -> tuple[float, float, float]:
    return metrics["iou"], metrics["depth_coverage"], -metrics["semantic_leakage"]


def _clarity_prompt(label: str) -> str:
    return f"""Use case: precise-object-edit
Asset type: Pixal3D conditioning image
Primary request: Make the displayed {label} visually clearer, sharper, and fully coherent as a realistic complete object while preserving its geometry.
Input images: Image 1 is the edit target and sole geometry authority.
Scene/backdrop: pure white studio background.
Constraints: preserve exactly the input camera viewpoint, object orientation, articulated pose, apparent scale, image-plane center, crop, silhouette, proportions, and every visible component; do not rotate, mirror, recrop, rescale, recenter, add, remove, bend, or reposition any part. Improve only edge clarity, material coherence, and fine visible detail. No text, logo, or watermark."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Optional JSON sample-to-label mapping.")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    samples = [str(sample) for sample in config["sample_ids"]]
    labels = {str(key): str(value) for key, value in config.get("prompt_overrides", {}).items()}
    if args.labels is not None:
        labels.update(json.loads(args.labels.read_text(encoding="utf-8")))

    records: dict[str, object] = {}
    for sample in samples:
        camera_dir = run_root / "inputs" / "camera" / sample
        pixal_dir = run_root / "inputs" / "pixal" / sample
        depth = camera_dir / "depth.png"
        qwen = camera_dir / "img.png"
        clarity = pixal_dir / "gpt_clarity_candidate.png"
        if not depth.is_file() or not qwen.is_file():
            raise FileNotFoundError(f"missing fresh Camera-1 assets for {sample}")
        pixal_dir.mkdir(parents=True, exist_ok=True)

        candidate_paths = {"qwen": qwen}
        if clarity.is_file():
            candidate_paths["gpt_clarity"] = clarity
        metrics = {
            name: _foreground_metrics(depth, path)
            for name, path in candidate_paths.items()
        }
        selected = max(candidate_paths, key=lambda name: _rank(metrics[name]))
        selected_source = candidate_paths[selected]
        destination = pixal_dir / "gpt_image.png"
        shutil.copy2(selected_source, destination)

        label = labels.get(sample, sample.replace("_", " "))
        prompt_path = pixal_dir / "gpt_clarity_prompt.txt"
        prompt_path.write_text(_clarity_prompt(label) + "\n", encoding="utf-8")
        record = {
            "sample": sample,
            "selection": selected,
            "selection_rule": "lexicographic(iou, depth_coverage, -semantic_leakage)",
            "ground_truth_used": False,
            "depth": str(depth),
            "candidates": {
                name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "foreground_vs_depth": metrics[name],
                }
                for name, path in candidate_paths.items()
            },
            "installed_path": str(destination),
            "installed_sha256": _sha256(destination),
            "prompt_path": str(prompt_path),
        }
        (pixal_dir / "conditioning_selection.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records[sample] = record
        print(f"[{sample}] selected {selected}: {metrics[selected]}", flush=True)

    manifest = {
        "method": "camera1_depth_supported_semantic_selection",
        "ground_truth_used": False,
        "samples": records,
    }
    (run_root / "conditioning_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
