from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_spar3d_agent_prior import semantic_to_rgba


def test_white_background_semantic_mask_preserves_foreground(tmp_path: Path):
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    image[1, 1] = [200, 20, 10]
    path = tmp_path / "semantic.png"
    Image.fromarray(image).save(path)
    rgba = np.asarray(semantic_to_rgba(path))
    assert rgba[0, 0, 3] == 0
    assert rgba[1, 1, 3] == 255
