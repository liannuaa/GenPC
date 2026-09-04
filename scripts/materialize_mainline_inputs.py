#!/usr/bin/env python3
"""Materialize the exact retained ten-sample inputs for an auditable rerun.

This is intentionally a *rebuild*, not another stochastic image/3-D
generation attempt.  It copies the previously accepted Qwen/GPT/Pixal assets
and the matching saved-camera files into one self-contained experiment root;
the following registration and Gaussian stages then execute from those copies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


import sys

ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mainline_paths import redwood_partial_root

SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830", "07136", "07306", "09639")

CAMERA_FILES = (
    "depth.png", "raw_depth.png", "img.png", "img_sam.png", "camera.pth", "point_uv.npy",
    "qwen_edit_prompt.txt", "qwen_edit_stage1.png", "qwen_edit_stage1_prompt.txt",
)
PIXAL_FILES = (
    "qwen_img.png", "gpt_image.png", "prompt.txt", "pixal3d_input.png", "pixal3d.glb",
    "pixal3d_sampled_100k.ply", "pixal3d_metadata.json",
)


def _copy(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "workspace" / "single_view_boundary_gaussian_redwood10_20260904")
    parser.add_argument("--camera-source", type=Path,
                        default=SHARED_ROOT / "workspace" / "redwood_onestage_rawdepth_512_stage2_20260714")
    parser.add_argument("--pixal-source", type=Path,
                        default=SHARED_ROOT / "workspace" / "redwood_qwen_gpt_pixal_bidirectional_mainline_20260823")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest: dict[str, object] = {
        "method": "materialized_accepted_qwen_gpt_pixal_inputs",
        "regenerated": False,
        "note": "Exact accepted assets are copied to make the registration/edit experiment self-contained.",
        "camera_source": str(args.camera_source.resolve()),
        "pixal_source": str(args.pixal_source.resolve()),
        "samples": {},
    }
    for sample in SAMPLES:
        camera_source = args.camera_source / sample
        pixal_source = args.pixal_source / sample
        camera_target = root / "inputs" / "camera" / sample
        pixal_target = root / "inputs" / "pixal" / sample
        partial_target = root / "inputs" / "partial" / f"{sample}.ply"
        copied: list[str] = []
        for name in CAMERA_FILES:
            source = camera_source / name
            if source.exists():
                _copy(source, camera_target / name); copied.append(f"camera/{name}")
            elif name in ("depth.png", "img.png", "camera.pth", "point_uv.npy"):
                raise FileNotFoundError(source)
        mask = camera_source / f"{sample}_moge_to_raw_partial_object_mask.png"
        _copy(mask, camera_target / mask.name); copied.append(f"camera/{mask.name}")
        for name in PIXAL_FILES:
            _copy(pixal_source / name, pixal_target / name); copied.append(f"pixal/{name}")
        _copy(redwood_partial_root(ROOT) / f"{sample}.ply", partial_target); copied.append("partial")
        manifest["samples"][sample] = {"state": "materialized", "files": copied}
    (root / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(root), "samples": list(SAMPLES)}, indent=2))


if __name__ == "__main__":
    main()
