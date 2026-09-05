#!/usr/bin/env python3
"""Rebuild the shared Pixal--MoGe--partial registration route for Redwood.

The runner intentionally reconstructs only registration intermediates from
retained Qwen/GPT/Pixal assets.  It never regenerates semantic images, GLBs, or
surface priors, and it never reads GT/CD/EMD.  For every sample it applies one
fixed route:

Pixal-native MoGe -> two-camera bridge -> coupled residual -> broad
pixel-indexed visible 3-D Sim(3) capture -> narrow visible 3-D Sim(3) ->
amplified Camera-1 -> 1-degree wide tilt -> 1-degree continuation. All stages
are applied; fixed objectives choose among the shared candidates without
proposal gates.
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mainline_paths import (
    REDWOOD10_SAMPLE_IDS,
    REGISTERED_PRIOR_FILENAME,
    redwood_partial_root,
)


def _paths(sample: str, *, pixal_root: Path, camera_root: Path, partial_root: Path,
           output_root: Path) -> dict[str, Path]:
    pixal = pixal_root / sample
    camera = camera_root / sample
    output = output_root / sample
    return {
        "partial": partial_root / f"{sample}.ply",
        "prior": pixal / "pixal3d_sampled_100k.ply",
        "pixal_input": pixal / "pixal3d_input.png",
        "pixal_metadata": pixal / "pixal3d_metadata.json",
        "pixal_moge_cache": pixal / "pixal_moge_fp16_observation.npz",
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


def _run(command: list[str], *, cwd: Path, log: Path, dry_run: bool,
         in_process: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    if in_process:
        script = Path(command[1])
        if command[0] != sys.executable or script.suffix != ".py":
            raise ValueError("in-process registration requires a Python script command")
        previous_argv, previous_cwd = sys.argv[:], Path.cwd()
        try:
            os.chdir(cwd)
            with log.open("w", encoding="utf-8") as handle, \
                    contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                sys.argv = command[1:]
                runpy.run_path(str((cwd / script).resolve()), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise RuntimeError(f"stage failed (see {log})") from error
        finally:
            sys.argv = previous_argv
            os.chdir(previous_cwd)
        return
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"stage failed (see {log})")


def _exists(path: Path, *, resume: bool) -> bool:
    return bool(resume and path.is_file() and path.stat().st_size > 0)


def _manifest(args: argparse.Namespace) -> dict[str, object]:
    """Build the deterministic, GT-free registration record."""
    return {
        "method": "fixed_pixal_moge_two_camera_pixel_sim3_camera1_continuation",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "samples": list(args.samples), "status": {},
        "parameters": {
            "proposal_gates_used": False,
            "amplified_levels": [[.006, .30, .006], [.002, .10, .002], [.0005, .025, .0005]],
            "wide_tilt_levels": [[.010, float(args.camera1_wide_tilt_degrees), .010],
                                 [.004, .35 * float(args.camera1_wide_tilt_degrees), .004],
                                 [.001, .10 * float(args.camera1_wide_tilt_degrees), .001]],
            "final_tilt_levels": [[.010, float(args.camera1_final_tilt_degrees), .010],
                                  [.004, .35 * float(args.camera1_final_tilt_degrees), .004],
                                  [.001, .10 * float(args.camera1_final_tilt_degrees), .001]],
            "camera1_search_points": int(args.camera1_search_points),
            "camera1_wide_tilt_degrees": float(args.camera1_wide_tilt_degrees),
            "camera1_final_tilt_degrees": float(args.camera1_final_tilt_degrees),
            "coarse_basin_recovery": bool(args.coarse_basin_recovery),
            "sample_workers": int(args.sample_workers),
            "bridge_only": bool(args.bridge_only),
        },
    }


def _write_json(path: Path, contents: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2), encoding="utf-8")


def _parallel_child_command(args: argparse.Namespace, *, sample: str, status: Path) -> list[str]:
    """Dispatch one independent sample without sharing Python/GPU state.

    Each child retains the existing within-sample in-process stage reuse.  A
    separate process is essential: ``runpy`` temporarily changes ``sys.argv``
    and is intentionally not thread-safe.  The only change from serial mode is
    that independent object directories are processed at the same time.
    """
    command = [
        sys.executable, str(Path(__file__).resolve()), "--samples", sample,
        "--pixal-root", str(args.pixal_root), "--camera-root", str(args.camera_root),
        "--partial-root", str(args.partial_root), "--output-root", str(args.output_root),
        "--moge-model", str(args.moge_model), "--rmbg-model", str(args.rmbg_model),
        "--camera1-search-points", str(args.camera1_search_points),
        "--camera1-wide-tilt-degrees", str(args.camera1_wide_tilt_degrees),
        "--camera1-final-tilt-degrees", str(args.camera1_final_tilt_degrees),
        "--sample-workers", "1", "--sample-status", str(status), "--no-write-batch-manifest",
    ]
    if not args.coarse_basin_recovery:
        command.append("--no-coarse-basin-recovery")
    if args.bridge_only:
        command.append("--bridge-only")
    if not args.resume:
        command.append("--no-resume")
    if not args.in_process:
        command.append("--no-in-process")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _run_parallel_child(command: list[str], *, sample: str, output_root: Path) -> tuple[str, int]:
    """Run one sample child and keep its scheduler transcript beside outputs."""
    log = output_root / sample / "scheduler.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return sample, int(result.returncode)


def _run_samples_parallel(args: argparse.Namespace) -> dict[str, object]:
    """Parallelize only independent samples; preserve each frozen sample route."""
    manifest = _manifest(args)
    status_root = args.output_root / ".scheduler_status"
    status_root.mkdir(parents=True, exist_ok=True)
    status_by_sample: dict[str, dict[str, object]] = {}
    commands = {
        sample: _parallel_child_command(args, sample=sample, status=status_root / f"{sample}.json")
        for sample in args.samples
    }
    with ThreadPoolExecutor(max_workers=int(args.sample_workers)) as executor:
        futures = {
            executor.submit(_run_parallel_child, command, sample=sample, output_root=args.output_root): sample
            for sample, command in commands.items()
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                _, returncode = future.result()
                status_path = status_root / f"{sample}.json"
                if returncode == 0 and status_path.is_file():
                    child_status = json.loads(status_path.read_text(encoding="utf-8"))
                    status_by_sample[sample] = dict(child_status.get("status", {}).get(sample, {}))
                else:
                    status_by_sample[sample] = {
                        "state": "failed",
                        "error": f"parallel child exited with status {returncode}; see {args.output_root / sample / 'scheduler.log'}",
                    }
            except Exception as exc:
                status_by_sample[sample] = {"state": "failed", "error": str(exc)}
    manifest["status"] = {sample: status_by_sample.get(sample, {
        "state": "failed", "error": "parallel child produced no status"}) for sample in args.samples}
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=list(REDWOOD10_SAMPLE_IDS),
                        help="Sample identifiers with matching partial, camera, and Pixal assets.")
    parser.add_argument("--pixal-root", type=Path,
                        default=SHARED_ROOT / "workspace" / "redwood_qwen_gpt_pixal_bidirectional_mainline_20260823")
    parser.add_argument("--camera-root", type=Path,
                        default=SHARED_ROOT / "workspace" / "redwood_onestage_rawdepth_512_stage2_20260714")
    parser.add_argument("--partial-root", type=Path, default=redwood_partial_root(ROOT))
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "workspace" / "pixal_moge_full9_rebuilt_20260904")
    parser.add_argument("--moge-model", type=Path, default=SHARED_ROOT / "models" / "moge-2-vitl" / "model.pt")
    parser.add_argument("--rmbg-model", type=Path, default=SHARED_ROOT / "models" / "RMBG-2.0")
    parser.add_argument("--camera1-search-points", type=int, default=32_000,
                        help="Shared visible Sim(3) subset size for all three Camera-1 continuations.")
    parser.add_argument("--camera1-wide-tilt-degrees", type=float, default=1.0,
                        help="Shared wide Camera-1 tilt trust region in degrees.")
    parser.add_argument("--camera1-final-tilt-degrees", type=float, default=1.0,
                        help="Shared final Camera-1 tilt trust region in degrees.")
    parser.add_argument("--coarse-basin-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the fixed globally shared broad Camera-1 pixel-Sim(3) capture before refinement.")
    parser.add_argument(
        "--bridge-only", action="store_true",
        help=(
            "Stop after the camera-2 native-MoGe to camera-1 partial bridge, without joint or Camera-1 "
            "residual stages. This is the scene route: the bridge maps the complete Pixal prior into the "
            "shared scene-MoGe coordinate frame."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--in-process", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse imports between fixed stages; --no-in-process restores subprocess execution.")
    parser.add_argument("--sample-workers", type=int, default=1,
                        help="Independent samples to register concurrently; one preserves the serial baseline.")
    parser.add_argument("--sample-status", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--write-batch-manifest", action=argparse.BooleanOptionalAction, default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.camera1_search_points < 3:
        raise ValueError("--camera1-search-points must be at least three")
    if min(args.camera1_wide_tilt_degrees, args.camera1_final_tilt_degrees) <= 0.:
        raise ValueError("Camera-1 tilt trust regions must be positive")
    if args.sample_workers < 1:
        raise ValueError("--sample-workers must be positive")
    if len(set(args.samples)) != len(args.samples):
        raise ValueError("--samples must not contain duplicate identifiers")

    args.pixal_root = args.pixal_root.resolve()
    args.camera_root = args.camera_root.resolve()
    args.partial_root = args.partial_root.resolve()
    args.output_root = args.output_root.resolve()
    args.moge_model = args.moge_model.resolve()
    args.rmbg_model = args.rmbg_model.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.sample_workers > 1 and len(args.samples) > 1:
        manifest = _run_samples_parallel(args)
        if args.write_batch_manifest:
            _write_json(args.output_root / "batch_manifest.json", manifest)
        if args.sample_status is not None:
            _write_json(args.sample_status.resolve(), manifest)
        print(json.dumps(manifest, indent=2))
        return

    python = sys.executable
    manifest = _manifest(args)
    for sample in args.samples:
        paths = _paths(sample, pixal_root=args.pixal_root, camera_root=args.camera_root,
                       partial_root=args.partial_root,
                       output_root=args.output_root)
        try:
            _require(paths, ("partial", "prior", "pixal_input", "pixal_metadata", "camera", "point_uv", "source_mask", "semantic"))
            native_ply = paths["native"] / "pixal_native_moge_points.ply"
            native_info = paths["native"] / "pixal_native_moge_info.json"
            native_registered = paths["native"] / "pixal_native_moge_registered_100k.ply"
            if not (
                _exists(native_ply, resume=args.resume)
                and _exists(native_info, resume=args.resume)
                and _exists(native_registered, resume=args.resume)
            ):
                native_command = [
                    python, "scripts/run_pixal_native_moge_registration.py",
                    "--prior", str(paths["prior"]), "--pixal-metadata", str(paths["pixal_metadata"]),
                    "--pixal-input", str(paths["pixal_input"]), "--rmbg-model", str(args.rmbg_model),
                    "--output-dir", str(paths["native"]), "--device", "cuda", "--fp16",
                ]
                if paths["pixal_moge_cache"].is_file():
                    native_command.extend(["--cached-moge-observation", str(paths["pixal_moge_cache"])])
                else:
                    native_command.extend(["--moge-model", str(args.moge_model)])
                _run(native_command, cwd=ROOT, log=paths["native"] / "stage.log", dry_run=args.dry_run,
                     in_process=args.in_process)
            target_mask = paths["native"] / "pixal_input_object_mask.png"
            bridge_transform = paths["bridge"] / "two_camera_pixal_moge_native_moge_to_partial.npy"
            pixel_matches = paths["bridge"] / "two_camera_pixal_moge_partial_to_native_moge_matches.npy"
            bridge_prior = paths["bridge"] / "two_camera_pixal_moge_registered_100k.ply"
            bridge_info = paths["bridge"] / "two_camera_pixal_moge_info.json"
            bridge_ready = (
                _exists(bridge_transform, resume=args.resume)
                and _exists(pixel_matches, resume=args.resume)
                and _exists(bridge_prior, resume=args.resume)
            )
            if args.bridge_only and bridge_ready:
                if not bridge_info.is_file():
                    bridge_ready = False
                else:
                    prior_refinement = json.loads(bridge_info.read_text(encoding="utf-8")).get(
                        "pixal_partial_micro_refinement", {}
                    )
                    bridge_ready = not bool(prior_refinement.get("enabled", True))
            if not bridge_ready:
                bridge_command = [
                    python, "scripts/run_pixal_moge_two_camera_bridge.py",
                    "--partial", str(paths["partial"]), "--point-uv", str(paths["point_uv"]),
                    "--source-mask", str(paths["source_mask"]), "--target-mask", str(target_mask),
                    "--native-moge", str(native_ply), "--native-moge-info", str(native_info),
                    "--pixal-prior", str(paths["prior"]), "--partial-camera", str(paths["camera"]),
                    "--saved-view-image", str(paths["semantic"]), "--output-dir", str(paths["bridge"]),
                    "--device", "cpu",
                ]
                if args.bridge_only:
                    # The bridge itself is the necessary Camera-2 -> Camera-1
                    # coordinate conversion.  Do not alter the already-good
                    # native Pixal alignment with its optional local residual.
                    bridge_command.append("--no-refine")
                _run(bridge_command, cwd=ROOT, log=paths["bridge"] / "stage.log", dry_run=args.dry_run,
                     in_process=args.in_process)
            if args.bridge_only:
                manifest["status"][sample] = {
                    "state": "complete",
                    "final": str(bridge_prior.resolve()),
                    "route": "pixal_native_moge_then_camera2_to_camera1_bridge",
                }
                if args.write_batch_manifest:
                    _write_json(args.output_root / "batch_manifest.json", manifest)
                continue
            joint_prior = paths["joint"] / "two_camera_joint_registered_100k.ply"
            if not _exists(joint_prior, resume=args.resume):
                joint_command = [
                    python, "scripts/run_pixal_moge_joint_bundle.py",
                    "--partial", str(paths["partial"]), "--pixal-prior", str(paths["prior"]),
                    "--native-moge", str(native_ply), "--native-info", str(native_info),
                    "--bridge-transform", str(bridge_transform), "--pixel-matches", str(pixel_matches),
                    "--partial-camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["joint"]), "--device", "cpu",
                ]
                if args.coarse_basin_recovery:
                    joint_command.append("--coarse-basin-recovery")
                _run(joint_command, cwd=ROOT, log=paths["joint"] / "stage.log", dry_run=args.dry_run,
                     in_process=args.in_process)
            amplified_prior = paths["amplified"] / REGISTERED_PRIOR_FILENAME
            if not _exists(amplified_prior, resume=args.resume):
                _run([
                    python, "scripts/run_camera1_amplified_sim3_refine.py",
                    "--partial", str(paths["partial"]), "--registered-prior", str(joint_prior),
                    "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["amplified"]),
                    "--search-points", str(args.camera1_search_points), "--device", "cpu",
                ], cwd=ROOT, log=paths["amplified"] / "stage.log", dry_run=args.dry_run,
                     in_process=args.in_process)
            wide_prior = paths["wide_tilt"] / REGISTERED_PRIOR_FILENAME
            if not _exists(wide_prior, resume=args.resume):
                _run([
                    python, "scripts/run_camera1_amplified_sim3_refine.py",
                    "--partial", str(paths["partial"]), "--registered-prior", str(amplified_prior),
                    "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["wide_tilt"]), "--search-points", str(args.camera1_search_points),
                    "--wide-tilt-search", "--max-tilt-degrees", str(args.camera1_wide_tilt_degrees), "--device", "cpu",
                ], cwd=ROOT, log=paths["wide_tilt"] / "stage.log", dry_run=args.dry_run,
                     in_process=args.in_process)
            final_prior = paths["final"] / REGISTERED_PRIOR_FILENAME
            if not _exists(final_prior, resume=args.resume):
                _run([
                    python, "scripts/run_camera1_amplified_sim3_refine.py",
                    "--partial", str(paths["partial"]), "--registered-prior", str(wide_prior),
                    "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
                    "--output-dir", str(paths["final"]), "--search-points", str(args.camera1_search_points),
                    "--wide-tilt-search", "--max-tilt-degrees", str(args.camera1_final_tilt_degrees),
                    "--device", "cpu",
                ], cwd=ROOT, log=paths["final"] / "stage.log", dry_run=args.dry_run,
                     in_process=args.in_process)
            manifest["status"][sample] = {"state": "complete", "final": str(final_prior.resolve())}
        except Exception as exc:  # keep the remaining shared samples running
            manifest["status"][sample] = {"state": "failed", "error": str(exc)}
        if args.write_batch_manifest:
            _write_json(args.output_root / "batch_manifest.json", manifest)
    if args.sample_status is not None:
        _write_json(args.sample_status.resolve(), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
