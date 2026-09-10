from pathlib import Path

import numpy as np
from PIL import Image

from src.trellis_multiview_probe import (
    build_residual_view_refinement_prompt,
    foreground_bbox,
    normalise_edited_view_framing,
)


def _rectangle(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    image[box[1]:box[3], box[0]:box[2]] = (40, 60, 80)
    Image.fromarray(image).save(path)


def test_normalise_edited_view_restores_foreground_frame(tmp_path):
    reference = tmp_path / "reference.png"
    edited = tmp_path / "edited.png"
    output = tmp_path / "normalised.png"
    _rectangle(reference, (120, 100), (30, 20, 90, 80))
    _rectangle(edited, (200, 160), (20, 10, 180, 150))

    record = normalise_edited_view_framing(edited, reference, output)

    restored = foreground_bbox(Image.open(output))
    target = foreground_bbox(Image.open(reference))
    restored_centre = ((restored[0] + restored[2]) / 2, (restored[1] + restored[3]) / 2)
    target_centre = ((target[0] + target[2]) / 2, (target[1] + target[3]) / 2)
    assert np.allclose(restored_centre, target_centre, atol=1)
    assert output.is_file()
    assert record["isotropic_scale"] < 1.0


def test_residual_view_prompt_only_parameterises_category_and_views():
    prompt = build_residual_view_refinement_prompt("office chair", "back", "side")
    assert "office chair" in prompt
    assert "BACK-view diagnostic evidence" in prompt
    assert "accepted SIDE view" in prompt
    assert "leg" not in prompt.lower()
    assert "front half" not in prompt.lower()
