import json

import numpy as np
import torch

from src.saved_camera import SavedCameraProjector


def test_saved_camera_uses_exact_pinhole_sidecar_when_available(tmp_path):
    camera_path = tmp_path / "camera.pth"
    torch.save({"legacy_camera": "unused"}, camera_path)
    metadata = {
        "image_size": 101,
        "intrinsic": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        "extrinsic_world_to_camera": np.eye(4).tolist(),
    }
    (tmp_path / "camera.json").write_text(json.dumps(metadata), encoding="utf-8")
    points = np.array(((0.0, 0.0, 2.0), (1.0, -1.0, 2.0)))
    projector = SavedCameraProjector.from_partial(
        points, camera_path, padding=.15, image_shape=(201, 201),
    )
    pixels, depth = projector.project(points)
    assert projector.projection_model == "pinhole_sidecar"
    assert np.allclose(pixels, ((100.0, 100.0), (200.0, 0.0)))
    assert np.allclose(depth, (2.0, 2.0))
