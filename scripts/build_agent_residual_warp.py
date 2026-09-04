#!/usr/bin/env python3
"""Build an observation-driven local 2-D control image for one agent action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from src.residual_image_warp import warp_residual_support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=.5)
    parser.add_argument("--support-radius-pixels", type=float, default=48.)
    args = parser.parse_args()
    for path in (args.source_image, args.partial, args.prior, args.camera):
        if not path.exists():
            raise FileNotFoundError(path)
    image = np.asarray(Image.open(args.source_image).convert("RGB"))
    partial, prior = base.load_points(args.partial), base.load_points(args.prior)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    projector = base.SavedCameraProjector.from_partial(
        partial, args.camera, padding=.15, image_shape=image.shape[:2], device="cpu")
    warped, evidence = warp_residual_support(
        image, partial, prior, projector, diagonal=diagonal, gain=args.gain,
        support_radius_pixels=args.support_radius_pixels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "residual_warp_control.png"
    Image.fromarray(warped).save(output)
    record = {
        "method": "partial_prior_residual_camera_locked_2d_warp",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_or_category_specific_parameters": False,
        "source": str(args.source_image), "partial": str(args.partial), "prior": str(args.prior),
        "camera": str(args.camera), "evidence": evidence.to_dict(), "output": str(output),
    }
    (args.output_dir / "residual_warp_info.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
