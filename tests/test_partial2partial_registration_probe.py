import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_partial2partial_registration_probe as probe


def test_image_view_bbox_transform_uses_uniform_scale_only():
    source = np.array(
        [
            [-2.0, -1.0, 6.0],
            [2.0, 1.0, 8.0],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [-2.0, -0.5, -0.25],
            [2.0, 0.5, 0.25],
        ],
        dtype=np.float64,
    )

    transform = probe.image_view_bbox_transform(
        source,
        target,
        axis_map=(0, 2, 1),
        signs=(-1, -1, -1),
        percentile=0,
    )

    singular_values = np.linalg.svd(transform[:3, :3], compute_uv=False)

    np.testing.assert_allclose(singular_values, np.full(3, singular_values[0]))


if __name__ == "__main__":
    test_image_view_bbox_transform_uses_uniform_scale_only()
