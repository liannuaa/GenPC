#!/usr/bin/env python3
"""Persist the exact GPT refinement prompts used by the scratch agent run."""

from __future__ import annotations

import argparse
from pathlib import Path


COMMON = (
    "Use case: product-mockup\n"
    "Asset type: zero-shot 3-D completion semantic input\n"
    "Input image: Image 1 is the edit target and geometric reference.\n"
)


PROMPTS = {
    "01184": COMMON + "Primary request: Refine this realistic rubbish bin image into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, silhouette, lid shape, handle placement, body proportions, and the two visible wheel locations/orientations from Image 1. Improve only sharpness, surface coherence, and clearly implied details. Do not rotate, mirror, recenter, crop, resize, change wheel orientation, add/remove parts, or alter object geometry. No text, watermark, or extra objects.\n",
    "05117": COMMON + "Primary request: Refine this realistic red upholstered chair image into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, silhouette, backrest tilt, seat proportions, armrest placement, and every visible leg from Image 1. Improve only sharpness, material coherence, and clearly implied details. Do not rotate, mirror, recenter, crop, resize, add/remove parts, or alter object geometry. No text, watermark, or extra objects.\n",
    "05452": COMMON + "Primary request: Refine this side-view armchair into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, silhouette, thin curved metal front legs, curved arm support, seat and backrest proportions, and the original slanted/curved backrest shape from Image 1. Do not introduce a pointed corner, an extra leg, or a new armrest. Improve only sharpness, material coherence, and clearly implied details. Do not rotate, mirror, recenter, crop, resize, add/remove parts, or alter object geometry. No text, watermark, or extra objects.\n",
    "06127": COMMON + "Primary request: Refine this terracotta flower pot with leafy plant into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, pot silhouette and rim/base proportions, stem layout, and visible leaf count, placement, and contour from Image 1. Improve only sharpness, natural material coherence, and clearly implied detail. Do not rotate, mirror, recenter, crop, resize, add/remove leaves or other parts, or alter object geometry. No text, watermark, or extra objects.\n",
    "06145": COMMON + "Primary request: Refine this pedestal table into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, rectangular tabletop aspect ratio and orientation, tabletop thickness, central support location, and round base footprint from Image 1. In particular do not swap the tabletop length and width or change its perspective orientation. Improve only sharpness, realistic material coherence, and clearly implied details. Do not rotate, mirror, recenter, crop, resize, add/remove parts, or alter object geometry. No text, watermark, or extra objects.\n",
    "06188": COMMON + "Primary request: Refine this red motor scooter into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, body silhouette, steering direction, front-wheel yaw and lean, front fork, headlight, handlebar, mirrors, seat, and rear-wheel placement from Image 1. The front wheel must retain the exact direction and tilt visible in Image 1; do not straighten it, reverse it, or swap its orientation. Improve only sharpness, material coherence, and clearly implied detail. Do not rotate, mirror, recenter, crop, resize, add/remove parts, or alter geometry. No text, watermark, or extra objects.\n",
    "06830": COMMON + "Primary request: Refine this three-wheeled children's tricycle/scooter into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, tall black rear handle, blue frame, seat, rear-wheel spacing, and front-wheel orientation from Image 1. The front wheel and fork must keep the slight existing steering angle; do not straighten, mirror, reverse, or relocate it. Improve only sharpness, surface coherence, and clearly implied details. Do not rotate, recenter, crop, resize, add/remove parts, or alter geometry. No text, watermark, or extra objects.\n",
    "07136": COMMON + "Primary request: Refine this dark leather sofa into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, full sofa length, width, seat depth, backrest height, left armrest contour, right-side seat edge, and the three visible cushion divisions from Image 1. Do not shorten, elongate, rotate, mirror, or reshape the sofa. Improve only sharpness, leather coherence, and clearly implied details. Do not recenter, crop, resize, add/remove cushions or parts, or alter object geometry. No text, watermark, or extra objects.\n",
    "07306": COMMON + "Primary request: Refine this red cylindrical office trash can into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, circular rim, top opening, body taper, height, and base silhouette from Image 1. Improve only sharpness, metal/plastic material coherence, and clearly implied details. Do not rotate, mirror, recenter, crop, resize, add/remove parts, or alter object geometry. No text, watermark, logo, or extra objects.\n",
    "09639": COMMON + "Primary request: Refine this ergonomic office chair into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, tall backrest silhouette, seat, both armrests, central stem, and all five visible star-base legs with their wheel directions from Image 1. Keep the chair proportions and local armrest/leg placement; do not widen, shorten, rotate, mirror, or replace the base. Improve only sharpness, upholstery/material coherence, and clearly implied detail. Do not recenter, crop, resize, add/remove parts, or alter geometry. No text, watermark, or extra objects.\n",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=sorted(PROMPTS))
    args = parser.parse_args()
    for sample in args.samples:
        (args.root / sample / "gpt_image_prompt.txt").write_text(
            PROMPTS[sample], encoding="utf-8")


if __name__ == "__main__":
    main()
