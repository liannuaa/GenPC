from pathlib import Path

import numpy as np
from PIL import Image

from src.object_mainline import (
    FINAL_PREDICTION_FILENAME,
    ObjectMainlineLayout,
    VIEWS,
    materialize_camera_conditions,
)
from src.trellis_multiview_probe import (
    build_local_residual_prompt,
    build_shared_low_frequency_prompt,
)


def _object_image(size: int, shift: int = 0) -> Image.Image:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[12:52, 18 + shift:46 + shift] = (70, 90, 120)
    return Image.fromarray(image)


def test_layout_has_one_fusion_free_prediction_contract(tmp_path: Path) -> None:
    layout = ObjectMainlineLayout(tmp_path, "sample")
    assert layout.prediction == tmp_path / "final" / "sample" / FINAL_PREDICTION_FILENAME
    assert "prediction" in layout.status()
    assert not layout.status()["prediction"]


def test_prompts_are_category_parameterized_and_part_agnostic() -> None:
    shared = build_shared_low_frequency_prompt("generic articulated object")
    local = build_local_residual_prompt("generic articulated object", "side")
    assert "generic articulated object" in shared
    assert "SIDE" in local
    for forbidden in ("pig", "front leg", "rear leg", "tail", "wing"):
        assert forbidden not in shared.lower()
        assert forbidden not in local.lower()


def test_external_edits_are_normalized_to_the_render_camera(tmp_path: Path) -> None:
    layout = ObjectMainlineLayout(tmp_path, "sample")
    layout.render_dir.mkdir(parents=True)
    originals = []
    for view in VIEWS:
        path = layout.render_view(view)
        _object_image(64).save(path)
        originals.append(path)

    board = Image.new("RGB", (192, 64), "white")
    for index in range(3):
        board.paste(_object_image(64, shift=2), (index * 64, 0))
    layout.stage1_board.parent.mkdir(parents=True)
    board.save(layout.stage1_board)
    layout.local_raw_dir.mkdir(parents=True)
    for view in VIEWS:
        _object_image(80, shift=4).save(layout.local_raw_view(view))

    result = materialize_camera_conditions(layout)
    assert result["ground_truth_used"] is False
    assert layout.condition_board.is_file()
    for view in VIEWS:
        assert Image.open(layout.condition(view)).size == (64, 64)
