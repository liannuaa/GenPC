"""Translate saved-view residual patches into a bounded image-edit action.

Unlike a PCA-axis-only instruction, this module tells a text-conditioned image
editor *where in the registered camera view* an observed mismatch occurs. It
uses only the partial cloud, current complete prior, and the fixed saved
camera. No object category, sample ID, GT surface, or metric is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.hierarchical_residual_registration import visible_residual_components


@dataclass(frozen=True)
class SavedViewTextFeedback:
    eligible: bool
    correction: str
    region: str
    component_fraction: float
    median_residual_ratio: float
    prompt_style: str
    suggested_image_shift_percent: int
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _region_name(pixel: np.ndarray, image_shape: tuple[int, int]) -> str:
    height, width = map(float, image_shape)
    x, y = float(pixel[0]) / max(width, 1.), float(pixel[1]) / max(height, 1.)
    horizontal = "left" if x < 1. / 3. else "right" if x > 2. / 3. else "central"
    vertical = "upper" if y < 1. / 3. else "lower" if y > 2. / 3. else "middle"
    return f"{vertical}-{horizontal}" if horizontal != "central" and vertical != "middle" else (
        horizontal if vertical == "middle" else vertical)


def _bounded_image_shift_percent(residual_ratio: float) -> int:
    """Convert a normalized measured residual to a visible, bounded 2-D cue."""
    return int(np.clip(np.rint(100. * float(residual_ratio)), 2, 8))


def build_saved_view_text_feedback(
    partial: np.ndarray,
    registered_prior: np.ndarray,
    projector,
    *,
    diagonal: float,
    prompt_style: str = "conservative",
) -> SavedViewTextFeedback:
    """Describe the largest supported residual component in saved-view words."""
    if prompt_style not in {"conservative", "explicit_local"}:
        raise ValueError(f"unsupported prompt style: {prompt_style}")
    components, summary = visible_residual_components(
        partial, registered_prior, projector, diagonal=diagonal,
        pixel_radius=8., residual_quantile=.70, residual_min_ratio=.018,
        residual_max_ratio=.14, min_points=48, max_components=8,
    )
    if not components:
        return SavedViewTextFeedback(
            eligible=False, correction="conform", region="visible surface",
            component_fraction=0., median_residual_ratio=0.,
            prompt_style=prompt_style, suggested_image_shift_percent=0,
            prompt=("Keep the exact saved-camera view, object identity, global pose, global "
                    "proportions, structure, and hidden geometry unchanged. Do not add or remove "
                    "parts. Make no geometric edit because no compact observed residual is supported."),
        )
    (source, target), info = components[0]
    pixel, _ = projector.project(target)
    region = _region_name(np.median(pixel, axis=0), projector.image_shape)
    correction = "expand" if info["target_span_ratio"] > info["source_span_ratio"] else "contract"
    verb = "gently expand outward" if correction == "expand" else "gently contract inward"
    shift_percent = _bounded_image_shift_percent(info["median_residual_ratio"])
    if prompt_style == "conservative":
        prompt = (
            "Keep the exact saved-camera view, object identity, global pose, global proportions, "
            "structural parts, and hidden geometry unchanged. Do not rotate, crop, mirror, recenter, "
            "add, or remove parts. Only on the visible " + region + " surface region, " + verb +
            " so that the local outer surface follows the observed scan. Keep every other visible "
            "and hidden region unchanged; make the correction local and smooth."
        )
    else:
        prompt = (
            "This is a camera-locked local geometry edit, not a request to preserve the input exactly. "
            "Keep object identity, camera, outer object extent, global pose, global proportions, "
            "structural parts, and hidden geometry unchanged. Do not rotate, crop, mirror, recenter, "
            "add, or remove parts. In the visible " + region + " surface support only, clearly " +
            ("pull the local surface and its shading inward" if correction == "contract" else
             "push the local surface and its shading outward") + " by about " + str(shift_percent) +
            "% of the object image diagonal, with a smooth transition. Make this local correction "
            "visibly apparent while leaving every other visible and hidden region unchanged."
        )
    return SavedViewTextFeedback(
        eligible=True, correction=correction, region=region,
        component_fraction=float(info["component_fraction"]),
        median_residual_ratio=float(info["median_residual_ratio"]), prompt_style=prompt_style,
        suggested_image_shift_percent=shift_percent, prompt=prompt,
    )
