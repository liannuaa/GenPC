#!/usr/bin/env python3
"""Prepare camera-consistent multi-view conditions for TRELLIS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trellis_multiview_probe import (
    build_residual_view_refinement_prompt,
    compose_equal_view_board,
    normalise_edited_view_framing,
    render_registered_multiview,
    split_equal_view_board,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    render = subparsers.add_parser("render")
    render.add_argument("--glb", type=Path, required=True)
    render.add_argument("--source-carrier", type=Path, required=True)
    render.add_argument("--registered-carrier", type=Path, required=True)
    render.add_argument("--camera", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--resolution", type=int, default=768)
    render.add_argument("--fov", type=float, default=38.0)
    render.add_argument(
        "--informative-partial", type=Path,
        help="Keep Camera-1 and select three auxiliary views by partial information gain.",
    )
    render.add_argument("--candidate-yaw-step", type=float, default=15.0)
    render.add_argument("--min-yaw-separation", type=float, default=45.0)
    render.add_argument("--selection-resolution", type=int, default=256)
    render.add_argument("--num-views", type=int, default=4)

    split = subparsers.add_parser("split")
    split.add_argument("--board", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--names", nargs="+", default=["front", "side", "back"])

    normalise = subparsers.add_parser("normalise-view")
    normalise.add_argument("--edited", type=Path, required=True)
    normalise.add_argument("--reference", type=Path, required=True)
    normalise.add_argument("--output", type=Path, required=True)
    normalise.add_argument("--white-threshold", type=int, default=245)

    compose = subparsers.add_parser("compose")
    compose.add_argument("--views", type=Path, nargs="+", required=True)
    compose.add_argument("--output", type=Path, required=True)

    prompt = subparsers.add_parser("view-prompt")
    prompt.add_argument("--object-type", required=True)
    prompt.add_argument("--target-view", required=True)
    prompt.add_argument("--reference-view", required=True)
    prompt.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.operation == "render":
        # Must be set before importing pyrender/OpenGL.
        # Prefer pyrender's pyglet backend when an X server is available
        # (including xvfb). Some headless hosts expose CUDA but no EGL
        # devices, so forcing EGL there makes otherwise portable rendering
        # fail before the geometry is loaded.
        if not os.environ.get("DISPLAY"):
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        result = render_registered_multiview(
            glb_path=args.glb.resolve(),
            source_carrier_path=args.source_carrier.resolve(),
            registered_carrier_path=args.registered_carrier.resolve(),
            camera_path=args.camera.resolve(),
            output_dir=args.output_dir.resolve(),
            resolution=args.resolution,
            field_of_view_degrees=args.fov,
            informative_partial_path=(
                None if args.informative_partial is None
                else args.informative_partial.resolve()
            ),
            candidate_yaw_step_degrees=args.candidate_yaw_step,
            min_yaw_separation_degrees=args.min_yaw_separation,
            selection_resolution=args.selection_resolution,
            num_views=args.num_views,
        )
    elif args.operation == "split":
        paths = split_equal_view_board(args.board.resolve(), args.output_dir.resolve(), args.names)
        result = {"board": str(args.board.resolve()), "views": [str(path.resolve()) for path in paths]}
        (args.output_dir.resolve() / "condition_manifest.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    elif args.operation == "normalise-view":
        result = normalise_edited_view_framing(
            args.edited.resolve(),
            args.reference.resolve(),
            args.output.resolve(),
            args.white_threshold,
        )
    elif args.operation == "compose":
        output = compose_equal_view_board(
            [path.resolve() for path in args.views], args.output.resolve(),
        )
        result = {"views": [str(path.resolve()) for path in args.views], "board": str(output.resolve())}
    else:
        prompt_text = build_residual_view_refinement_prompt(
            args.object_type, args.target_view, args.reference_view,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt_text, encoding="utf-8")
        result = {
            "object_type": args.object_type,
            "target_view": args.target_view,
            "reference_view": args.reference_view,
            "prompt": str(args.output.resolve()),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
