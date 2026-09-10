#!/usr/bin/env python3
"""Run the fusion-free GenPC++ object pipeline with resumable image checkpoints.

The deterministic stages are executed locally. Two image-edit checkpoints are
explicit because this repository does not embed credentials or an opaque GPT
client: clarity editing before Pixal, and residual-guided multi-view editing
before TRELLIS. Re-running ``--stage all`` resumes after either checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SHARED_ROOT = ROOT.parents[1] if ROOT.parent.name == ".codex-worktrees" else ROOT

from src.object_mainline import (
    ExternalArtifactRequired,
    ObjectMainlineLayout,
    VIEWS,
    copy_input_once,
    materialize_camera_conditions,
    publish_prediction,
    write_manifest,
)


def _run(command: list[str], *, commands: list[list[str]], dry_run: bool) -> None:
    commands.append(command)
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _resolved_config(
    base: Path, layout: ObjectMainlineLayout, object_type: str, *, write: bool = True,
) -> Path:
    config = yaml.safe_load(base.read_text(encoding="utf-8"))
    config["paths"]["data_dir"] = str(layout.partial_root.resolve())
    config["paths"]["output_dir"] = str(layout.camera_root.resolve())
    config["paths"]["models_dir"] = str((SHARED_ROOT / "models").resolve())
    config["sample_ids"] = [layout.sample]
    config.setdefault("prompt_overrides", {})[layout.sample] = object_type
    path = layout.root / "manifests" / f"{layout.sample}_resolved_config.yaml"
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _prepare_upstream(
    args: argparse.Namespace, layout: ObjectMainlineLayout, commands: list[list[str]],
) -> None:
    if not args.dry_run:
        copy_input_once(args.partial, layout.partial)
    config = _resolved_config(
        args.config, layout, args.object_type, write=not args.dry_run,
    )
    if not (layout.camera.is_file() and layout.semantic.is_file()):
        _run([
            args.genpc_python, "scripts/run_semantic_stage.py",
            "--config", str(config),
            "--partial-root", str(layout.partial_root),
            "--output-root", str(layout.camera_root),
            "--models-root", str((SHARED_ROOT / "models").resolve()),
            "--samples", layout.sample,
        ], commands=commands, dry_run=args.dry_run)
    if not (layout.clarity_image.is_file() and layout.clarity_prompt.is_file()):
        raise ExternalArtifactRequired(
            "save a camera/pose-preserving clarity edit of inputs/camera/<sample>/img.png "
            "and its exact prompt before resuming",
            (layout.clarity_image, layout.clarity_prompt),
        )
    if not (layout.pixal_mesh.is_file() and layout.pixal_carrier.is_file()):
        _run([
            args.genpc_python, "scripts/run_pixal3d_gpt_batch.py",
            "--input-root", str(layout.pixal_root),
            "--output-root", str(layout.pixal_root),
            "--model", str(args.pixal_model),
            "--dino", str(args.dino_model),
            "--moge", str(args.moge_model),
            "--rmbg", str(args.rmbg_model),
            "--ids", layout.sample,
        ], commands=commands, dry_run=args.dry_run)
    if not layout.registered_pixal.is_file():
        _run([
            args.genpc_python, "scripts/run_fixed_pixal_moge_registration.py",
            "--samples", layout.sample,
            "--pixal-root", str(layout.pixal_root),
            "--camera-root", str(layout.camera_root),
            "--partial-root", str(layout.partial_root),
            "--output-root", str(layout.pixal_registration_root),
        ], commands=commands, dry_run=args.dry_run)


def _prepare_multiview(
    args: argparse.Namespace, layout: ObjectMainlineLayout, commands: list[list[str]],
) -> None:
    required = (
        layout.pixal_mesh, layout.pixal_carrier, layout.registered_pixal,
        layout.partial, layout.camera,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not layout.render_manifest.is_file():
        _run([
            args.genpc_python, "scripts/prepare_trellis_multiview_probe.py", "render",
            "--glb", str(layout.pixal_mesh),
            "--source-carrier", str(layout.pixal_carrier),
            "--registered-carrier", str(layout.registered_pixal),
            "--camera", str(layout.camera),
            "--output-dir", str(layout.render_dir),
            "--resolution", "768", "--fov", "38",
        ], commands=commands, dry_run=args.dry_run)
    if not layout.evidence_board.is_file():
        _run([
            args.genpc_python, "scripts/build_shared_local_residual_cards.py",
            "--manifest", str(layout.render_manifest),
            "--registered-prior", str(layout.registered_pixal),
            "--partial", str(layout.partial),
            "--object-type", args.object_type,
            "--output-dir", str(layout.evidence_dir),
            "--resolution", "512",
        ], commands=commands, dry_run=args.dry_run)


def _materialize_multiview(layout: ObjectMainlineLayout, *, dry_run: bool) -> None:
    if dry_run:
        return
    materialize_camera_conditions(layout)


def _generate_trellis(
    args: argparse.Namespace, layout: ObjectMainlineLayout, commands: list[list[str]],
) -> None:
    for view in VIEWS:
        if not layout.condition(view).is_file():
            raise FileNotFoundError(layout.condition(view))
    if not (layout.trellis_mesh.is_file() and layout.trellis_carrier.is_file()):
        _run([
            args.trellis_python, "scripts/run_trellis_multiview_regeneration.py",
            "--images", *(str(layout.condition(view)) for view in VIEWS),
            "--model", str(args.trellis_model),
            "--trellis-repo", str(args.trellis_repo),
            "--output-dir", str(layout.trellis_dir),
            "--seed", "42", "--mode", "stochastic",
            "--sparse-steps", "12", "--sparse-cfg", "7.5",
            "--slat-steps", "12", "--slat-cfg", "3.0",
            "--sample-points", "100000", "--preview-frames", "24",
        ], commands=commands, dry_run=args.dry_run)


def _select_views(
    args: argparse.Namespace, layout: ObjectMainlineLayout, commands: list[list[str]],
) -> None:
    for view in VIEWS:
        if layout.selection_json(view).is_file():
            continue
        _run([
            args.genpc_python, "scripts/select_trellis_camera1_view.py",
            "--mesh", str(layout.trellis_mesh),
            "--colored-points", str(layout.trellis_carrier),
            "--condition", str(layout.local_view(view)),
            "--output-dir", str(layout.selection_dir(view)),
        ], commands=commands, dry_run=args.dry_run)


def _raster_command(
    args: argparse.Namespace,
    layout: ObjectMainlineLayout,
    *,
    output: Path,
    steps: int,
    initial_info: Path | None = None,
    inverse_rounds: int = 0,
) -> list[str]:
    command = [
        args.genpc_python, "scripts/run_nvdiffrast_multiview_registration.py",
        "--mesh", str(layout.trellis_mesh),
        "--prior", str(layout.trellis_carrier),
        "--partial", str(layout.partial),
    ]
    if initial_info is None:
        command.extend(["--selection-json", *(str(layout.selection_json(view)) for view in VIEWS)])
    else:
        command.extend(["--initial-info", str(initial_info)])
    command.extend([
        "--render-manifest", str(layout.render_manifest),
        "--front", str(layout.condition("front")),
        "--side", str(layout.condition("side")),
        "--back", str(layout.condition("back")),
        "--camera", str(layout.camera),
        "--semantic", str(layout.semantic),
        "--output-dir", str(output),
        "--resolution", "128" if initial_info is None else "256",
        "--steps", str(steps),
    ])
    if inverse_rounds:
        command.extend(["--partial-inverse-rounds", str(inverse_rounds)])
    return command


def _register_trellis(
    args: argparse.Namespace, layout: ObjectMainlineLayout, commands: list[list[str]],
) -> None:
    _select_views(args, layout, commands)
    semantic_info = layout.semantic_capture_dir / "registration_info.json"
    if not semantic_info.is_file():
        _run(
            _raster_command(args, layout, output=layout.semantic_capture_dir, steps=420),
            commands=commands, dry_run=args.dry_run,
        )
    camera_info = layout.camera_capture_dir / "registration_info.json"
    if not camera_info.is_file():
        _run(
            _raster_command(
                args, layout, output=layout.camera_capture_dir, steps=400,
                initial_info=semantic_info, inverse_rounds=1,
            ),
            commands=commands, dry_run=args.dry_run,
        )
    polish_info = layout.camera_polish_dir / "registration_info.json"
    if not polish_info.is_file():
        _run(
            _raster_command(
                args, layout, output=layout.camera_polish_dir, steps=500,
                initial_info=camera_info,
            ),
            commands=commands, dry_run=args.dry_run,
        )
    final_prediction = layout.final_registration_dir / "trellis_registered_100k.ply"
    if not final_prediction.is_file():
        _run([
            args.genpc_python, "scripts/run_partial_supported_sim3.py",
            "--initial-prior", str(layout.camera_polish_dir / "trellis_registered_100k.ply"),
            "--partial", str(layout.partial),
            "--camera", str(layout.camera),
            "--semantic", str(layout.semantic),
            "--camera1-condition", str(layout.condition("front")),
            "--initial-transform", str(polish_info),
            "--output-dir", str(layout.final_registration_dir),
            "--rotation-degrees", "22", "--scale-ratio", "1.14",
            "--translation-ratio", "0.18", "--iterations", "14",
            "--population", "5", "--seed", "6145",
            "--silhouette-weight", "0", "--visible-weight", "1.2",
        ], commands=commands, dry_run=args.dry_run)
    if not args.dry_run:
        publish_prediction(layout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--object-type", required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("upstream", "prepare-edits", "materialize-edits", "trellis", "register", "all", "status"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "mainline_redwood.yaml")
    parser.add_argument("--genpc-python", default=sys.executable)
    parser.add_argument(
        "--trellis-python",
        default="/opt/data/private/cr/miniconda3/envs/las-comp/bin/python",
    )
    parser.add_argument("--pixal-model", type=Path, default=SHARED_ROOT / "models" / "Pixal3D-weights")
    parser.add_argument("--dino-model", type=Path, default=SHARED_ROOT / "models" / "dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--moge-model", type=Path, default=SHARED_ROOT / "models" / "moge-2-vitl")
    parser.add_argument("--rmbg-model", type=Path, default=SHARED_ROOT / "models" / "RMBG-2.0")
    parser.add_argument("--trellis-model", type=Path, default=SHARED_ROOT / "models" / "TRELLIS-image-large")
    parser.add_argument("--trellis-repo", type=Path, default=Path("/opt/data/private/cr/lab/LaS-Comp"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    layout = ObjectMainlineLayout(args.run_root.resolve(), args.sample)
    commands: list[list[str]] = []
    try:
        if args.stage == "status":
            pass
        elif args.stage == "upstream":
            _prepare_upstream(args, layout, commands)
        elif args.stage == "prepare-edits":
            _prepare_multiview(args, layout, commands)
        elif args.stage == "materialize-edits":
            _materialize_multiview(layout, dry_run=args.dry_run)
        elif args.stage == "trellis":
            _generate_trellis(args, layout, commands)
        elif args.stage == "register":
            _register_trellis(args, layout, commands)
        else:
            _prepare_upstream(args, layout, commands)
            _prepare_multiview(args, layout, commands)
            _materialize_multiview(layout, dry_run=args.dry_run)
            _generate_trellis(args, layout, commands)
            _register_trellis(args, layout, commands)
    except ExternalArtifactRequired as exc:
        write_manifest(layout, object_type=args.object_type, commands=commands)
        print(json.dumps({
            "state": "waiting_for_external_image_edit",
            "message": str(exc),
            "required_paths": [str(path.resolve()) for path in exc.paths],
            "status": layout.status(),
        }, indent=2))
        return

    manifest = write_manifest(layout, object_type=args.object_type, commands=commands)
    print(json.dumps({
        "state": "complete" if layout.prediction.is_file() else "ready",
        "prediction": str(layout.prediction.resolve()) if layout.prediction.is_file() else None,
        "manifest": str(manifest.resolve()),
        "status": layout.status(),
    }, indent=2))


if __name__ == "__main__":
    main()
