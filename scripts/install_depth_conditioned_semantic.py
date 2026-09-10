#!/usr/bin/env python3
"""Install and audit a direct-GPT semantic image against its saved depth mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.moge_pixel_bridge import run_rmbg_mask, save_mask_png


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bbox(mask: np.ndarray) -> list[int]:
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("foreground mask is empty")
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def _framing_measurements(depth_mask: np.ndarray, semantic_mask: np.ndarray) -> dict:
    depth_bbox, semantic_bbox = _bbox(depth_mask), _bbox(semantic_mask)
    depth_centre = np.array([
        .5 * (depth_bbox[0] + depth_bbox[2]), .5 * (depth_bbox[1] + depth_bbox[3]),
    ])
    semantic_centre = np.array([
        .5 * (semantic_bbox[0] + semantic_bbox[2]), .5 * (semantic_bbox[1] + semantic_bbox[3]),
    ])
    depth_size = np.array([
        depth_bbox[2] - depth_bbox[0], depth_bbox[3] - depth_bbox[1],
    ], dtype=np.float64)
    semantic_size = np.array([
        semantic_bbox[2] - semantic_bbox[0], semantic_bbox[3] - semantic_bbox[1],
    ], dtype=np.float64)
    intersection = depth_mask & semantic_mask
    return {
        "depth_bbox": depth_bbox,
        "semantic_bbox": semantic_bbox,
        "depth_centre": depth_centre,
        "semantic_centre": semantic_centre,
        "bbox_scale": semantic_size / np.maximum(depth_size, 1.),
        "intersection": intersection,
        "mask_iou": float(intersection.sum() / max(int((depth_mask | semantic_mask).sum()), 1)),
        "depth_coverage": float(intersection.sum() / max(int(depth_mask.sum()), 1)),
        "semantic_leakage": float((semantic_mask & ~depth_mask).sum() / max(int(semantic_mask.sum()), 1)),
    }


def _normalise_framing(
    image: Image.Image,
    semantic_bbox: list[int],
    depth_bbox: list[int],
    maximum_scale_error: float,
) -> tuple[Image.Image, dict]:
    """Apply the smallest global similarity that enters the Camera-1 bounds."""
    semantic_size = np.array([
        semantic_bbox[2] - semantic_bbox[0], semantic_bbox[3] - semantic_bbox[1],
    ], dtype=np.float64)
    depth_size = np.array([
        depth_bbox[2] - depth_bbox[0], depth_bbox[3] - depth_bbox[1],
    ], dtype=np.float64)
    ratios = semantic_size / np.maximum(depth_size, 1.)
    lower = float(np.max((1. - maximum_scale_error) / np.maximum(ratios, 1e-8)))
    upper = float(np.min((1. + maximum_scale_error) / np.maximum(ratios, 1e-8)))
    if lower <= upper:
        scale = float(np.clip(1., lower, upper))
    else:
        scale = float(1. / np.sqrt(np.prod(ratios)))
    source_centre = np.array([
        .5 * (semantic_bbox[0] + semantic_bbox[2]), .5 * (semantic_bbox[1] + semantic_bbox[3]),
    ])
    target_centre = np.array([
        .5 * (depth_bbox[0] + depth_bbox[2]), .5 * (depth_bbox[1] + depth_bbox[3]),
    ])
    inverse = (
        1. / scale, 0., float(source_centre[0] - target_centre[0] / scale),
        0., 1. / scale, float(source_centre[1] - target_centre[1] / scale),
    )
    corrected = image.transform(
        image.size, Image.Transform.AFFINE, inverse,
        resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255),
    )
    return corrected, {
        "type": "global_image_plane_similarity",
        "isotropic_scale": scale,
        "source_centre_xy": source_centre.tolist(),
        "target_centre_xy": target_centre.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument(
        "--prompt-file", type=Path, default=None,
        help="Exact direct-GPT prompt; defaults to gpt_depth_completion_prompt.txt beside --semantic.",
    )
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--rmbg-model", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--maximum-center-offset-ratio", type=float, default=.04)
    parser.add_argument("--maximum-bbox-scale-error", type=float, default=.125)
    parser.add_argument(
        "--minimum-depth-coverage", type=float, default=.835,
        help="Shared foreground-coverage floor; tolerant to thin-part RMBG boundary noise.",
    )
    parser.add_argument(
        "--normalize-framing", action=argparse.BooleanOptionalAction, default=True,
        help="Apply one global 2D similarity when GPT drifts from the saved Camera-1 bbox.",
    )
    args = parser.parse_args()

    prompt_path = args.prompt_file
    if prompt_path is None:
        candidate = args.semantic.parent / "gpt_depth_completion_prompt.txt"
        prompt_path = candidate if candidate.is_file() else None
    prompt = None if prompt_path is None else prompt_path.read_text(encoding="utf-8").strip()

    output = args.camera_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    semantic_path = output / "img.png"
    semantic = Image.open(args.semantic).convert("RGB").resize(
        (int(args.resolution), int(args.resolution)), Image.Resampling.LANCZOS,
    )
    semantic.save(semantic_path)
    alpha = run_rmbg_mask(
        semantic_path, output / "img_rmbg.png", args.rmbg_model.resolve(),
    )
    object_mask_path = output / f"{args.sample}_moge_to_raw_partial_object_mask.png"
    save_mask_png(object_mask_path, alpha)

    depth = np.asarray(
        Image.open(args.depth).convert("L").resize(semantic.size, Image.Resampling.NEAREST),
    )
    depth_mask = depth > 8
    semantic_mask = np.asarray(Image.open(object_mask_path).convert("L")) > 127
    initial = _framing_measurements(depth_mask, semantic_mask)
    centre_offset_ratio = float(
        np.linalg.norm(initial["semantic_centre"] - initial["depth_centre"]) / int(args.resolution)
    )
    bbox_error = float(np.max(np.abs(initial["bbox_scale"] - 1.)))
    framing_transform = None
    if bool(args.normalize_framing) and (
        centre_offset_ratio > float(args.maximum_center_offset_ratio)
        or bbox_error > float(args.maximum_bbox_scale_error)
    ):
        (output / "img_pre_framing.png").write_bytes(semantic_path.read_bytes())
        semantic, framing_transform = _normalise_framing(
            semantic, initial["semantic_bbox"], initial["depth_bbox"],
            float(args.maximum_bbox_scale_error),
        )
        semantic.save(semantic_path)
        alpha = run_rmbg_mask(
            semantic_path, output / "img_rmbg.png", args.rmbg_model.resolve(),
        )
        save_mask_png(object_mask_path, alpha)
        semantic_mask = np.asarray(Image.open(object_mask_path).convert("L")) > 127
    measured = _framing_measurements(depth_mask, semantic_mask)
    depth_bbox, semantic_bbox = measured["depth_bbox"], measured["semantic_bbox"]
    intersection = measured["intersection"]
    semantic_rgb = np.asarray(semantic, dtype=np.uint8).copy()
    audit_overlay = semantic_rgb.copy()
    depth_only = depth_mask & ~semantic_mask
    semantic_only = semantic_mask & ~depth_mask
    audit_overlay[depth_only] = np.array([0, 230, 255], dtype=np.uint8)
    audit_overlay[semantic_only] = np.array([255, 0, 220], dtype=np.uint8)
    audit_overlay_path = output / "semantic_depth_audit.png"
    Image.fromarray(audit_overlay).save(audit_overlay_path)
    centre_offset_ratio = float(
        np.linalg.norm(measured["semantic_centre"] - measured["depth_centre"]) / int(args.resolution)
    )
    bbox_scale = measured["bbox_scale"]
    depth_coverage = measured["depth_coverage"]
    passed = bool(
        centre_offset_ratio <= float(args.maximum_center_offset_ratio)
        and np.max(np.abs(bbox_scale - 1.)) <= float(args.maximum_bbox_scale_error)
        and depth_coverage >= float(args.minimum_depth_coverage)
    )
    record = {
        "method": "depth_mask_conditioned_direct_gpt_semantic_install",
        "ground_truth_used": False,
        "sample": str(args.sample),
        "source_semantic": str(args.semantic.resolve()),
        "source_sha256": _sha256(args.semantic),
        "prompt_file": None if prompt_path is None else str(prompt_path.resolve()),
        "prompt": prompt,
        "depth": str(args.depth.resolve()),
        "installed_semantic": str(semantic_path),
        "object_mask": str(object_mask_path),
        "audit_overlay": str(audit_overlay_path),
        "audit_colours": {
            "cyan": "observed depth not covered by semantic foreground",
            "magenta": "semantic foreground outside observed depth",
        },
        "resolution": int(args.resolution),
        "initial_framing": {
            "semantic_bbox_xyxy": initial["semantic_bbox"],
            "bbox_scale_xy": initial["bbox_scale"].tolist(),
            "centre_offset_ratio": float(
                np.linalg.norm(initial["semantic_centre"] - initial["depth_centre"])
                / int(args.resolution)
            ),
            "mask_iou": initial["mask_iou"],
            "depth_coverage": initial["depth_coverage"],
        },
        "framing_transform": framing_transform,
        "depth_bbox_xyxy": depth_bbox,
        "semantic_bbox_xyxy": semantic_bbox,
        "centre_offset_ratio": centre_offset_ratio,
        "bbox_scale_xy": bbox_scale.tolist(),
        "mask_iou": measured["mask_iou"],
        "depth_coverage": depth_coverage,
        "semantic_leakage": measured["semantic_leakage"],
        "thresholds": {
            "maximum_center_offset_ratio": float(args.maximum_center_offset_ratio),
            "maximum_bbox_scale_error": float(args.maximum_bbox_scale_error),
            "minimum_depth_coverage": float(args.minimum_depth_coverage),
        },
        "passed": passed,
    }
    manifest = output / "semantic_install_manifest.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not passed:
        raise SystemExit("semantic image does not preserve the saved depth framing")


if __name__ == "__main__":
    main()
