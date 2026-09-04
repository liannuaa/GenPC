#!/usr/bin/env python3
"""Rebuild the shared Pixal--MoGe--partial registration route for Redwood.

The runner intentionally reconstructs only registration intermediates from
retained Qwen/GPT/Pixal assets.  It never regenerates semantic images, GLBs, or
surface priors, and it never reads GT/CD/EMD.  For every sample it applies one
fixed route:

Pixal-native MoGe -> two-camera bridge -> coupled residual -> pixel-indexed
visible 3-D Sim(3) -> amplified Camera-1 -> 1-degree wide tilt -> 0.5-degree
continuation.  All stages are applied; score records are diagnostic rather
than proposal gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
SAMPLES = ("01184", "05117", "05452", "06127", "06145", "06188", "06830", "07306", "09639")


def _paths(sample: str, *, pixal_root: Path, camera_root: Path, output_root: Path) -> dict[str, Path]:
    pixal = pixal_root / sample
    camera = camera_root / sample
    output = output_root / sample
    return {
        "partial": ROOT / "data" / f"{sample}.ply",
        "prior": pixal / "pixal3d_sampled_100k.ply",
        "pixal_input": pixal / "pixal3d_input.png",
        "pixal_metadata": pixal / "pixal3d_metadata.json",
        "camera": camera / "camera.pth",
        "point_uv": camera / "point_uv.npy",
        "source_mask": camera / f"{sample}_moge_to_raw_partial_object_mask.png",
        "semantic": camera / "img.png",
        "native": output / "native",
        "bridge": output / "bridge",
        "joint": output / "joint",
        "amplified": output / "amplified",
        "wide_tilt": output / "wide_tilt",
        "final": output / "final",
    }


def _require(paths: dict[str, Path], keys: Iterable[str]) -> None:
    missing = [str(paths[key]) for key in keys if not paths[key].is_file() or paths[key].stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing required retained input(s): " + ", ".join(missing))


def _run(command: list[str], *, cwd: Path, log: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"stage failed (see {log})")


def _exists(path: Path, *, resume: bool) -> bool:
    return bool(resume and path.is_file() and path.stat().st_size > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=list(SAMPLES))
    parser.add_argument("--pixal-root", type=Path,
                        default=SHARED_ROOT / "workspace" / "redwood_qwen_gpt_pixal_bidirectional_mainline_20260823")
    parser.add_argument("--camera-root", type=Path,
                        default=SHARED_ROOT / "workspace" / "redwood_onestage_rawdepth_512_stage2_20260714")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "workspace" / "pixal_moge_full9_rebuilt_20260904")
    parser.add_argument("--moge-model", type=Path, default=SHARED_ROOT / "models" / "moge-2-vitl" / "model.pt")
    parser.add_argument("--rmbg-model", type=Path, default=SHARED_ROOT / "models" / "RMBG-2.0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    manifest: dict[str, object] = {
        "method": "fixed_pixal_moge_two_camera_pixel_sim3_camera1_continuation",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "samples": list(args.samples), "status": {},
        "parameters": {
            "proposal_gates_used": False,
            "amplified_levels": [[.006, .30, .006], [.002, .10, .002], [.0005, .025, .0005]],
            "wide_tilt_levels": [[.010, 1.00, .010], [.004, .35, .004], [.001, .10, .001]],
            "final_tilt_levels": [[.010, .50, .010], [.004, .175, .004], [.001, .05, .001]],
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample in args.samples:
        paths = _paths(sample, pixal_root=args.pixal_root, camera_root=args.camera_root,
                       output_root=args.output_root)
        try:
            _require(paths, ("partial", "prior", "pixal_input", "pixal_metadata", "camera", "point_uv", "source_mask", "semantic"))
            native_ply = paths["native"] / "pixal_native_moge_points.ply"
            native_info = paths["native"] / "pixal_native_moge_info.json"
            if not (_exists(native_ply, resume=args.resume) and _exists(native_info, resume=args.resume)):
                _run([
                    python, "scripts/run_agent_pixal_native_moge_registration.py",
                    "--prior", str(paths["prior"]), "--pixal-metadata", str(paths["pixal_metadata"]),
                    "--pixal-input", str(paths["pixal_input"]), "--moge-model", str(args.moge_model),
                    "--rmbg-model", str(args.rmbg_model), "--output-dir", str(paths["native"]),
                    "--device", "cuda", "--fp16",
                ], cwd=ROOT, log=paths["native"] / "stage.log", dry_run=args.dry_run)
            target_mask = paths["native"] / "pixal_input_object_mask.png"
            bridge_transform = paths["bridge"] / "two_camera_pixal_moge_native_moge_to_partial.npy"
            pixel_matches = paths["bridge"] / "two_camera_pixal_moge_partial_to_native_moge_matches.npy"
            bridge_prior = paths["bridge"] / "two_camera_pixal_moge_registered_100k.ply"
            if not (_exists(bridge_transform, resume=args.resume) and _exists(pixel_matches, resume=args.resume)
                    and _exists(bridge_prior, resume=args.resume)):
                _run([
                    python, "scripts/run_agent_pixal_moge_two_camera_bridge.py",
                    "--partial", str(paths["partial"]), "--point-uv", str(paths["point_uv"]),
                    "--source-mask", str(paths["source_mask"]), "--target-mask", str(target_mask),
                    "--native-moge", str(native_ply), "--native-moge-info", str(native_info),
                    "--pixal-prior", str(paths["prior"]), "--partial-camera", str(paths["camera"]),
                    "--saved-view-image", str(paths["semantic"]), "--output-dir", str(paths["bridge"]),
                    "--device", "cpu",
                ], cwd=ROOT, log=paths["bridge"] / "stage.log", dry_run=args.dry_run)
            joint_prior = paths["joint"] / "two_camera_joint_registered_100k.ply"
            if not _exists(joint_prior, resume=args.resume):
                _run([
                    python, "scripts/run_agent_pixal_moge_joint_bundle.py",
                    "--partial", str(paths["partial"]), "--pixal-prior", str(paths["prior"]),
                    "--native-moge", str(native_ply), "--native-info", str(native_info),
                    "--bridge-transform", str(bridge_transform), "--pixel-matches", str(pixel_matches),
                    "--partial-camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["joint"]), "--device", "cpu",
                ], cwd=ROOT, log=paths["joint"] / "stage.log", dry_run=args.dry_run)
            amplified_prior = paths["amplified"] / "camera1_amplified_registered_100k.ply"
            if not _exists(amplified_prior, resume=args.resume):
                _run([
                    python, "scripts/run_camera1_amplified_sim3_refine.py",
                    "--partial", str(paths["partial"]), "--registered-prior", str(joint_prior),
                    "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["amplified"]), "--device", "cpu",
                ], cwd=ROOT, log=paths["amplified"] / "stage.log", dry_run=args.dry_run)
            wide_prior = paths["wide_tilt"] / "camera1_amplified_registered_100k.ply"
            if not _exists(wide_prior, resume=args.resume):
                _run([
                    python, "scripts/run_camera1_amplified_sim3_refine.py",
                    "--partial", str(paths["partial"]), "--registered-prior", str(amplified_prior),
                    "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["wide_tilt"]), "--wide-tilt-search", "--device", "cpu",
                ], cwd=ROOT, log=paths["wide_tilt"] / "stage.log", dry_run=args.dry_run)
            final_prior = paths["final"] / "camera1_amplified_registered_100k.ply"
            if not _exists(final_prior, resume=args.resume):
                _run([
                    python, "scripts/run_camera1_amplified_sim3_refine.py",
                    "--partial", str(paths["partial"]), "--registered-prior", str(wide_prior),
                    "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["final"]), "--wide-tilt-search", "--max-tilt-degrees", ".5",
                    "--device", "cpu",
                ], cwd=ROOT, log=paths["final"] / "stage.log", dry_run=args.dry_run)
            manifest["status"][sample] = {"state": "complete", "final": str(final_prior.resolve())}
        except Exception as exc:  # keep the remaining shared samples running
            manifest["status"][sample] = {"state": "failed", "error": str(exc)}
        (args.output_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
