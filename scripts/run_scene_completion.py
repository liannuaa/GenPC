#!/usr/bin/env python3
"""Scene-level wrapper around the frozen single-object GenPC+ mainline.

Inputs are an RGB scene image and an auditable JSON list of GPT binary instance
masks.  The wrapper uses Pixal's MoGe-2 on the whole scene, extracts every
masked object as a partial in that one camera frame, and dispatches the
unchanged object-level public runners.  It never changes object-level defaults
or moves a completed object after its own registration.

GPT image editing is deliberately represented by saved files plus
``gpt_image_actions.json``: Codex's built-in image tool is an interactive
agent capability, not a hidden Python dependency.  This makes the scene run
resumable and fully inspectable after GPT masks/direct semantic images are
produced.  Scene mode never converts depth into a semantic image.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PIXAL3D_ROOT = Path(os.environ.get("PIXAL3D_SOURCE", str(SHARED_ROOT / "models" / "Pixal3D"))).resolve()
if PIXAL3D_ROOT.is_dir() and str(PIXAL3D_ROOT) not in sys.path:
    # MoGe is supplied by Pixal3D's pinned source checkout, just as it is in
    # run_pixal3d_gpt_batch.py.  No alternate scene-depth model is introduced.
    sys.path.insert(0, str(PIXAL3D_ROOT))

from src.scene_completion.contracts import load_scene_manifest
from src.scene_completion.instances import extract_scene_instances
from src.scene_completion.meshes import (
    export_registered_scene_meshes,
    materialize_scene_bridge_placements,
)
from src.scene_completion.orchestrator import (
    install_gpt_images,
    pixal_command,
    registration_command,
    run_command,
    write_gpt_action_manifest,
)
from src.scene_completion.scene_camera import write_scene_camera_assets
from src.scene_completion.scene_moge import infer_scene_moge, load_scene_moge


_STAGES = ("prepare", "gpt-actions", "pixal", "registration", "meshes", "all")


def _record(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def _need_gpt_images(root: Path, ids: tuple[str, ...]) -> None:
    missing = [str(root / "inputs" / "pixal" / sample / "gpt_image.png") for sample in ids
               if not (root / "inputs" / "pixal" / sample / "gpt_image.png").is_file()]
    if missing:
        raise FileNotFoundError(
            "Direct GPT semantic images are required before Pixal generation. Produce the declared completion_task "
            f"assets in {root / 'gpt_image_actions.json'}; missing: " + ", ".join(missing)
        )


def _stages_to_run(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return _STAGES[:-1]
    return (stage,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-image", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True,
                        help="GPT-produced scene-instance JSON with one binary mask per object.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=_STAGES, default="all")
    parser.add_argument("--models-root", type=Path, default=SHARED_ROOT / "models")
    parser.add_argument("--moge-model", type=Path, default=SHARED_ROOT / "models" / "moge-2-vitl" / "model.pt")
    parser.add_argument("--gpt-image-source", type=Path,
                        help="Directory containing <instance-id>.png generated from gpt_image_actions.json.")
    parser.add_argument("--overwrite-gpt-images", action="store_true")
    parser.add_argument("--overwrite-scene-moge", action="store_true")
    parser.add_argument(
        "--mask-erosion-pixels", type=int, default=5,
        help="Protect scene-MoGe instance partials from mask-boundary depth noise (fixed default: 5 px).",
    )
    parser.add_argument("--crop-padding-pixels", type=int, default=24)
    parser.add_argument("--scene-camera-padding", type=float, default=.15)
    parser.add_argument("--semantic-size", type=int, default=512)
    parser.add_argument(
        "--pixal-decimation-target", type=int, default=100_000,
        help="Maximum face target for each generated scene-instance Pixal mesh (default: 100k).",
    )
    parser.add_argument(
        "--overwrite-pixal", action="store_true",
        help="Regenerate completed scene Pixal assets, e.g. after changing --pixal-decimation-target.",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=512)
    parser.add_argument("--min-moge-points", type=int, default=128)
    parser.add_argument(
        "--registration-workers", type=int, default=2,
        help="Independent scene objects to register concurrently (default: conservative 2-worker budget on 24GB GPU).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32-moge", action="store_true")
    parser.add_argument(
        "--write-instance-meshes", action="store_true",
        help="Also persist one transformed GLB per instance; disabled by default to avoid duplicating scene storage.",
    )
    parser.add_argument(
        "--collision-clearance", type=float, default=.0015,
        help="Final-scene mesh gap in scene-MoGe units; 0 disables collision refinement (default: .0015).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.mask_erosion_pixels, args.crop_padding_pixels) < 0:
        parser.error("mask erosion and crop padding must be non-negative")
    if min(args.min_mask_pixels, args.min_moge_points) < 3:
        parser.error("minimum mask/point counts must be at least three")
    if args.registration_workers < 1:
        parser.error("--registration-workers must be positive")
    if args.pixal_decimation_target < 1_000:
        parser.error("--pixal-decimation-target must be at least 1,000")
    if not 0.0 <= args.scene_camera_padding < 0.5:
        parser.error("--scene-camera-padding must be in [0, .5)")
    if args.collision_clearance < 0.:
        parser.error("--collision-clearance must be non-negative")

    root = args.output_root.resolve()
    scene_image = args.scene_image.resolve()
    manifest = load_scene_manifest(args.instances, expected_source_image=scene_image)
    stages = _stages_to_run(args.stage)
    run_state = {
        "method": "scene_moge_partials_direct_gpt_semantics_textured_pixal_meshes",
        "scene_image": str(scene_image),
        "instance_manifest": str(manifest.path),
        "instances": [{"id": item.instance_id, "label": item.label, "layer": item.layer}
                      for item in manifest.instances],
        "requested_stages": list(stages),
        "object_mainline_modified": False,
        "scene_moge_contract": "same Pixal MoGe-2 model and RGB tensor convention as native Pixal registration",
        "partial_contract": "every instance partial is the GPT-mask-indexed subset of the visible scene MoGe cloud",
        "depth_to_semantic_used": False,
        "final_representation": "registered_textured_pixal_meshes_without_point_cloud_fusion",
    }
    root.mkdir(parents=True, exist_ok=True)
    _record(root / "scene_run_manifest.json", **run_state)

    observation = None
    if "prepare" in stages:
        cache = root / "scene_moge" / "pixal_moge_scene_observation.npz"
        if cache.is_file() and not args.overwrite_scene_moge:
            observation = load_scene_moge(root / "scene_moge")
            if observation.image_path != scene_image:
                raise ValueError("existing scene MoGe cache belongs to a different scene image; use a new output root")
        else:
            observation = infer_scene_moge(
                scene_image, moge_model=args.moge_model, output_dir=root / "scene_moge",
                device=args.device, fp16=not args.fp32_moge,
            )
        extract_scene_instances(
            manifest, observation, output_root=root,
            erosion_pixels=args.mask_erosion_pixels, crop_padding=args.crop_padding_pixels,
            min_mask_pixels=args.min_mask_pixels, min_moge_points=args.min_moge_points,
        )
        write_scene_camera_assets(
            manifest, observation, output_root=root, padding=args.scene_camera_padding,
        )

    if "gpt-actions" in stages:
        action_path = write_gpt_action_manifest(root, manifest)
        print(f"[GPT actions] {action_path}", flush=True)

    if args.gpt_image_source is not None:
        install_gpt_images(
            root, manifest.instance_ids, args.gpt_image_source,
            rmbg_model=args.models_root / "RMBG-2.0", overwrite=args.overwrite_gpt_images,
            semantic_size=args.semantic_size,
        )

    if "pixal" in stages:
        _need_gpt_images(root, manifest.instance_ids)
        command, log = pixal_command(
            root=root, manifest=manifest, decimation_target=args.pixal_decimation_target,
            overwrite=args.overwrite_pixal,
        )
        run_command(command, root=ROOT, log=log, dry_run=args.dry_run)

    if "registration" in stages:
        command, log = registration_command(
            root=root, manifest=manifest, sample_workers=args.registration_workers, bridge_only=True,
        )
        run_command(command, root=ROOT, log=log, dry_run=args.dry_run)
        if not args.dry_run:
            materialize_scene_bridge_placements(
                manifest,
                pixal_root=root / "inputs" / "pixal",
                partial_root=root / "inputs" / "partial",
                registration_root=root / "registration",
            )

    if "meshes" in stages:
        scene_context = root / "scene_context_unmasked_visible.ply"
        export_registered_scene_meshes(
            manifest, pixal_root=root / "inputs" / "pixal", registration_root=root / "registration",
            output_dir=root / "scene_meshes", write_instance_meshes=args.write_instance_meshes,
            scene_context_path=scene_context if scene_context.is_file() else None,
            collision_clearance=args.collision_clearance,
        )

    run_state["completed_stages"] = list(stages)
    _record(root / "scene_run_manifest.json", **run_state)
    print(json.dumps({"output_root": str(root), "stages": list(stages), "instances": list(manifest.instance_ids)}, indent=2))


if __name__ == "__main__":
    main()
