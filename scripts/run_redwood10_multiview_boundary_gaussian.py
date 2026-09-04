#!/usr/bin/env python3
"""Run one fixed multi-view positive-overlap Gaussian edit on Redwood-10.

The runner consumes only a frozen registered Pixal body, the original partial,
and saved-camera assets. It never invokes GT/CD/EMD or per-sample routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830", "07136", "07306", "09639")


def _run(command: list[str], *, log: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"stage failed; see {log}")


def _complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT / "workspace" / "single_view_boundary_gaussian_redwood10_20260904")
    parser.add_argument("--registration-root", type=Path,
                        help="Optional frozen registration root. Defaults to <root>/registration.")
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=list(SAMPLES))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    registration_root = (root / "registration" if args.registration_root is None
                         else args.registration_root.resolve())
    python = sys.executable
    manifest: dict[str, object] = {
        "method": "multiview_positive_overlap_boundary_conditioned_gaussian_edit",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "registration_root": str(registration_root),
        "parameters": {
            "saved_camera_max_pixel_distance": 2.0, "virtual_positive_views": 6,
            "virtual_render_size": 384, "virtual_max_pixel_distance": 2.0,
            "max_anchor_residual_ratio": .06, "max_displacement_ratio": .06,
            "graph_neighbours": 8, "graph_edge_ratio": 1.8, "graph_screening": .003,
            "prior_protection_views": 6, "prior_protection_weight": .02,
            "protection_exclusion_ratio": .08,
        },
        "samples": {},
    }
    for sample in args.samples:
        registered = registration_root / sample / "camera1_amplified_registered_100k.ply"
        if not registered.is_file():
            registered = registration_root / sample / "final" / "camera1_amplified_registered_100k.ply"
        partial = root / "inputs" / "partial" / f"{sample}.ply"
        camera = root / "inputs" / "camera" / sample / "camera.pth"
        semantic = root / "inputs" / "camera" / sample / "img.png"
        edit_root = root / "gaussian" / sample / "edit"
        decode_root = root / "gaussian" / sample / "decoded"
        edited = edit_root / "partial_anchored_gaussian_edit_editable_prior_100k.ply"
        prediction = decode_root / "partial_anchored_gaussian_decoded_100k.ply"
        try:
            required = (registered, partial, camera, semantic)
            missing = [str(path) for path in required if not _complete(path)]
            if missing:
                raise FileNotFoundError("missing required input: " + ", ".join(missing))
            if not (args.resume and _complete(edited)):
                _run([
                    python, "scripts/run_partial_anchored_gaussian_edit.py",
                    "--prior", str(registered), "--partial", str(partial), "--camera", str(camera),
                    "--semantic", str(semantic), "--output-dir", str(edit_root), "--device", "cpu",
                ], log=edit_root / "stage.log", dry_run=args.dry_run)
            if not (args.resume and _complete(prediction)):
                _run([
                    python, "scripts/run_partial_anchored_gaussian_decode.py",
                    "--edited-prior", str(edited), "--partial", str(partial), "--camera", str(camera),
                    "--semantic", str(semantic), "--view-reference", str(registered),
                    "--output-dir", str(decode_root), "--device", "cpu",
                ], log=decode_root / "stage.log", dry_run=args.dry_run)
            manifest["samples"][sample] = {"state": "complete", "prediction": str(prediction)}
        except Exception as exc:
            manifest["samples"][sample] = {"state": "failed", "error": str(exc)}
        (root / "gaussian" / "batch_manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "gaussian" / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
