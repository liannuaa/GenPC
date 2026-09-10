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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mainline_paths import (
    GAUSSIAN_PREDICTION_FILENAME,
    REDWOOD10_SAMPLE_IDS,
    locate_registered_prior,
)


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
    parser.add_argument("--gaussian-subdir", default="gaussian",
                        help="Output child directory for a comparable Gaussian candidate (default: gaussian).")
    parser.add_argument("--samples", nargs="+", default=list(REDWOOD10_SAMPLE_IDS),
                        help="Sample identifiers with materialized partial, camera, Pixal, and registration assets.")
    parser.add_argument("--max-pixel-distance", type=float, default=1.0)
    parser.add_argument("--virtual-positive-views", type=int, default=6)
    parser.add_argument("--virtual-render-size", type=int, default=384)
    parser.add_argument("--virtual-max-pixel-distance", type=float, default=1.0)
    parser.add_argument("--max-anchor-residual-ratio", type=float, default=.075)
    parser.add_argument("--support-radius-ratio", type=float, default=.08)
    parser.add_argument("--max-displacement-ratio", type=float, default=.075)
    parser.add_argument("--graph-neighbours", type=int, default=8)
    parser.add_argument("--graph-edge-ratio", type=float, default=1.8)
    parser.add_argument("--graph-screening", type=float, default=.0015)
    parser.add_argument("--prior-protection-views", type=int, default=6)
    parser.add_argument("--prior-protection-weight", type=float, default=.01)
    parser.add_argument("--protection-exclusion-ratio", type=float, default=.10)
    parser.add_argument("--remote-gain", type=float, default=1.0)
    parser.add_argument("--remote-gain-radius-ratio", type=float, default=.08)
    parser.add_argument("--remote-displacement-cap-multiplier", type=float, default=1.0)
    parser.add_argument("--camera-axis-scale", action="store_true",
                        help="Generate one whole-carrier Camera-1 relative-axis scale candidate before local Gaussian editing.")
    parser.add_argument("--axis-scale-max-log", type=float, default=.16)
    parser.add_argument("--axis-scale-minimum-pairs", type=int, default=512)
    parser.add_argument("--axis-scale-minimum-anisotropy", type=float, default=.025)
    parser.add_argument("--axis-scale-minimum-residual-reduction", type=float, default=.03)
    parser.add_argument("--axis-scale-trim-quantile", type=float, default=.75)
    parser.add_argument("--axis-scale-iterations", type=int, default=3)
    parser.add_argument("--axis-scale-retain-isotropic-component", action="store_true")
    parser.add_argument("--graph-cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--graph-cg-max-iterations", type=int, default=240)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    gaussian_subdir = Path(str(args.gaussian_subdir))
    if gaussian_subdir.is_absolute() or len(gaussian_subdir.parts) != 1 or gaussian_subdir.name in {"", ".", ".."}:
        raise ValueError("--gaussian-subdir must be one safe child directory name")
    registration_root = (root / "registration" if args.registration_root is None
                         else args.registration_root.resolve())
    python = sys.executable
    parameters = {
        "saved_camera_max_pixel_distance": args.max_pixel_distance,
        "virtual_positive_views": args.virtual_positive_views,
        "virtual_render_size": args.virtual_render_size,
        "virtual_max_pixel_distance": args.virtual_max_pixel_distance,
        "max_anchor_residual_ratio": args.max_anchor_residual_ratio,
        "support_radius_ratio": args.support_radius_ratio,
        "max_displacement_ratio": args.max_displacement_ratio,
        "graph_neighbours": args.graph_neighbours,
        "graph_edge_ratio": args.graph_edge_ratio,
        "graph_screening": args.graph_screening,
        "prior_protection_views": args.prior_protection_views,
        "prior_protection_weight": args.prior_protection_weight,
        "protection_exclusion_ratio": args.protection_exclusion_ratio,
        "remote_gain": args.remote_gain,
        "remote_gain_radius_ratio": args.remote_gain_radius_ratio,
        "remote_displacement_cap_multiplier": args.remote_displacement_cap_multiplier,
        "camera_axis_scale": args.camera_axis_scale,
        "axis_scale_max_log": args.axis_scale_max_log,
        "axis_scale_minimum_pairs": args.axis_scale_minimum_pairs,
        "axis_scale_minimum_anisotropy": args.axis_scale_minimum_anisotropy,
        "axis_scale_minimum_residual_reduction": args.axis_scale_minimum_residual_reduction,
        "axis_scale_trim_quantile": args.axis_scale_trim_quantile,
        "axis_scale_iterations": args.axis_scale_iterations,
        "axis_scale_retain_isotropic_component": args.axis_scale_retain_isotropic_component,
        "graph_cg_tolerance": args.graph_cg_tolerance,
        "graph_cg_max_iterations": args.graph_cg_max_iterations,
    }
    manifest_path = root / gaussian_subdir / "batch_manifest.json"
    manifest: dict[str, object] = {
        "method": "multiview_positive_overlap_boundary_conditioned_gaussian_edit",
        "strict_zero_shot": True, "ground_truth_cd_emd_used": False,
        "registration_root": str(registration_root),
        "parameters": parameters,
        "samples": {},
    }
    if args.resume and manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (prior_manifest.get("method") != manifest["method"]
                or prior_manifest.get("registration_root") != manifest["registration_root"]
                or prior_manifest.get("parameters") != parameters):
            raise ValueError(
                "existing Gaussian manifest has different inputs or parameters; "
                "use a new output root rather than mixing runs"
            )
        manifest["samples"] = dict(prior_manifest.get("samples", {}))
    for sample in args.samples:
        registered = locate_registered_prior(registration_root, sample)
        partial = root / "inputs" / "partial" / f"{sample}.ply"
        camera = root / "inputs" / "camera" / sample / "camera.pth"
        semantic = root / "inputs" / "camera" / sample / "img.png"
        scale_root = root / gaussian_subdir / sample / "camera_axis_scale"
        edit_root = root / gaussian_subdir / sample / "edit"
        decode_root = root / gaussian_subdir / sample / "decoded"
        scaled_prior = scale_root / "camera_axis_scaled_prior_100k.ply"
        edited = edit_root / "partial_anchored_gaussian_edit_editable_prior_100k.ply"
        prediction = decode_root / GAUSSIAN_PREDICTION_FILENAME
        try:
            required = (registered, partial, camera, semantic)
            missing = [str(path) for path in required if not _complete(path)]
            if missing:
                raise FileNotFoundError("missing required input: " + ", ".join(missing))
            edit_prior = registered
            if args.camera_axis_scale:
                if not (args.resume and _complete(scaled_prior)):
                    _run([
                        python, "scripts/run_camera_axis_scale.py",
                        "--prior", str(registered), "--partial", str(partial), "--camera", str(camera),
                        "--output-dir", str(scale_root), "--device", "cpu",
                        "--max-pixel-distance", str(args.max_pixel_distance),
                        "--max-anchor-residual-ratio", str(args.max_anchor_residual_ratio),
                        "--max-log-stretch", str(args.axis_scale_max_log),
                        "--minimum-pairs", str(args.axis_scale_minimum_pairs),
                        "--minimum-anisotropy", str(args.axis_scale_minimum_anisotropy),
                        "--minimum-residual-reduction", str(args.axis_scale_minimum_residual_reduction),
                        "--trim-quantile", str(args.axis_scale_trim_quantile),
                        "--iterations", str(args.axis_scale_iterations),
                        *(["--retain-isotropic-component"]
                          if args.axis_scale_retain_isotropic_component else []),
                    ], log=scale_root / "stage.log", dry_run=args.dry_run)
                edit_prior = scaled_prior
            if not (args.resume and _complete(edited)):
                _run([
                    python, "scripts/run_partial_anchored_gaussian_edit.py",
                    "--prior", str(edit_prior), "--partial", str(partial), "--camera", str(camera),
                    "--semantic", str(semantic), "--output-dir", str(edit_root), "--device", "cpu",
                    "--max-pixel-distance", str(args.max_pixel_distance),
                    "--virtual-positive-views", str(args.virtual_positive_views),
                    "--virtual-render-size", str(args.virtual_render_size),
                    "--virtual-max-pixel-distance", str(args.virtual_max_pixel_distance),
                    "--max-anchor-residual-ratio", str(args.max_anchor_residual_ratio),
                    "--support-radius-ratio", str(args.support_radius_ratio),
                    "--max-displacement-ratio", str(args.max_displacement_ratio),
                    "--neighbours", str(args.graph_neighbours),
                    "--graph-edge-ratio", str(args.graph_edge_ratio),
                    "--graph-screening", str(args.graph_screening),
                    "--prior-protection-views", str(args.prior_protection_views),
                    "--prior-protection-weight", str(args.prior_protection_weight),
                    "--protection-exclusion-ratio", str(args.protection_exclusion_ratio),
                    "--remote-gain", str(args.remote_gain),
                    "--remote-gain-radius-ratio", str(args.remote_gain_radius_ratio),
                    "--remote-displacement-cap-multiplier", str(args.remote_displacement_cap_multiplier),
                    "--graph-cg-tolerance", str(args.graph_cg_tolerance),
                    "--graph-cg-max-iterations", str(args.graph_cg_max_iterations),
                ], log=edit_root / "stage.log", dry_run=args.dry_run)
            if not (args.resume and _complete(prediction)):
                _run([
                    python, "scripts/run_partial_anchored_gaussian_decode.py",
                    "--edited-prior", str(edited), "--partial", str(partial), "--camera", str(camera),
                    "--semantic", str(semantic), "--view-reference", str(edit_prior),
                    "--output-dir", str(decode_root), "--device", "cpu",
                    "--max-pixel-distance", str(args.max_pixel_distance),
                    "--virtual-positive-views", str(args.virtual_positive_views),
                    "--virtual-render-size", str(args.virtual_render_size),
                    "--virtual-max-pixel-distance", str(args.virtual_max_pixel_distance),
                    "--max-anchor-residual-ratio", str(args.max_anchor_residual_ratio),
                ], log=decode_root / "stage.log", dry_run=args.dry_run)
            manifest["samples"][sample] = {
                "state": "complete", "prediction": str(prediction),
                "edit_prior": str(edit_prior),
            }
        except Exception as exc:
            manifest["samples"][sample] = {"state": "failed", "error": str(exc)}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
