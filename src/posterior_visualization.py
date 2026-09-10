"""Diagnostic renderers for the observation-conditioned posterior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.zbuffer import zbuffer_depth_with_indices


def save_support_overlay(
    path: Path,
    semantic: Path,
    prior: np.ndarray,
    masks: dict[str, np.ndarray],
    projector,
) -> None:
    """Render stable, residual, protected, and propagated carrier states."""
    image = Image.open(semantic).convert("RGB")
    height, width = projector.image_shape
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    uv, depth = projector.project(prior)
    _, visible, indices = zbuffer_depth_with_indices(
        uv, depth, projector.image_shape, splat_radius=0
    )
    draw = ImageDraw.Draw(image, "RGBA")
    styles = (
        ("protected", (110, 110, 110, 60), 1),
        ("moved", (42, 130, 255, 150), 1),
        ("locked", (35, 220, 110, 210), 2),
        ("editable", (255, 175, 20, 255), 2),
    )
    yy, xx = np.where(visible)
    ids = indices[yy, xx]
    for name, colour, radius in styles:
        keep = masks[name][ids]
        for x, y in zip(xx[keep], yy[keep]):
            draw.ellipse(
                (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius),
                fill=colour,
            )
    image.save(path)
