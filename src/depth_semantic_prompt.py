"""Prompt contract for direct-GPT semantic completion from a depth raster."""

from __future__ import annotations


def build_depth_semantic_prompt(
    object_description: str,
    *,
    structural_constraints: tuple[str, ...] = (),
) -> str:
    """Build a category-independent, mask-locked completion instruction.

    The depth raster, rather than a language description, remains the source of
    camera, framing, silhouette, and visible geometry.  Optional constraints
    describe genuine object structure (for example, parallel paired wheels),
    never a sample-specific image transform.
    """
    subject = object_description.strip()
    if not subject:
        raise ValueError("object_description must not be empty")
    constraints = " ".join(item.strip() for item in structural_constraints if item.strip())
    if constraints:
        constraints = f" Additional structural constraints: {constraints}"
    return (
        "Generate one complete, realistic RGB object image from the supplied incomplete depth map. "
        f"The object is {subject}. "
        "Treat every non-background depth pixel and its foreground mask as a hard ControlNet-like "
        "spatial constraint. Preserve the input camera viewpoint, perspective, image-plane center, "
        "apparent width and height, crop, orientation, visible silhouette, depth ordering, and all "
        "observed part locations. Do not rotate, mirror, recenter, rescale, widen, shorten, tilt, or "
        "redesign the observed object. Complete only genuinely missing or occluded regions, connect "
        "them naturally to the observed structure, and keep a single coherent object on a pure white "
        "background. Produce sharp boundaries and realistic materials suitable for image-to-3D "
        "reconstruction. Do not add text, logos, extra parts, secondary objects, or cast shadows that "
        f"change the foreground mask.{constraints}"
    )
