#!/usr/bin/env python3
"""Align and confidence-gate a MoGe point map as a soft partial extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.moge_soft_observation import (_unique_pairs, confidence_gated_moge_extension,
                                       moge_visible_soft_completion, robust_moge_to_partial)
from src.ray_consistent_registration import apply_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--moge", type=Path, required=True)
    parser.add_argument("--moge-pixels", type=Path, required=True)
    parser.add_argument("--pixel-matches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    partial, moge = base.load_points(args.partial), base.load_points(args.moge)
    pixels, matches = np.load(args.moge_pixels), np.load(args.pixel_matches)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    transform, fit = robust_moge_to_partial(moge, partial, matches, diagonal=diagonal)
    aligned = apply_transform(moge, transform)
    pairs = _unique_pairs(matches)
    residual = np.linalg.norm(aligned[pairs[:, 1].astype(np.int64)] - partial[pairs[:, 0].astype(np.int64)], axis=1)
    extension, confidence, support = confidence_gated_moge_extension(
        aligned, pixels, pairs, residual, image_size=args.image_size, diagonal=diagonal)
    visible_completion, visible_confidence, visible_support = moge_visible_soft_completion(
        aligned, pixels, pairs, residual, image_size=args.image_size, diagonal=diagonal)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(args.output_dir / "moge_aligned_soft_extension.ply", extension)
    combined = np.concatenate((partial, extension), axis=0)
    colors = np.concatenate((np.tile((145, 145, 145), (len(partial), 1)),
                             np.tile((45, 125, 230), (len(extension), 1))), axis=0)
    trimesh.points.PointCloud(combined, colors=colors).export(args.output_dir / "partial_gray_moge_soft_blue.ply")
    np.savez_compressed(args.output_dir / "moge_soft_observation.npz", points=extension,
                        confidence=confidence, hard_observation_count=len(partial),
                        transform=transform, matched_residual=residual)
    np.savez_compressed(args.output_dir / "moge_visible_soft_completion.npz", points=visible_completion,
                        confidence=visible_confidence, hard_observation_count=len(partial),
                        transform=transform, matched_residual=residual)
    base.write_points(args.output_dir / "moge_visible_soft_completion.ply", visible_completion)
    record = {
        "method": "scan_anchored_confidence_gated_moge_soft_observation",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "hard_observation": str(args.partial.resolve()),
        "soft_observation": "MoGe points only in real-scan-supported low-residual image cells",
        "soft_points_may_not_replace_or_move_hard_partial": True,
        "fit": fit, "support": support,
        "visible_soft_completion": visible_support,
        "input": {"moge": str(args.moge.resolve()), "moge_pixels": str(args.moge_pixels.resolve()),
                  "pixel_matches": str(args.pixel_matches.resolve())},
    }
    (args.output_dir / "moge_soft_observation_info.json").write_text(
        json.dumps(base.jsonable(record), indent=2), encoding="utf-8")
    print(json.dumps(base.jsonable({"fit": fit, "support": support,
                                    "visible_soft_completion": visible_support}), indent=2))


if __name__ == "__main__":
    main()
