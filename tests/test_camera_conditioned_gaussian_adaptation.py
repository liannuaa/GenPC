import numpy as np

from src.camera_conditioned_gaussian_adaptation import camera_conditioned_gaussian_adaptation


class _GridProjector:
    image_shape = (64, 64)

    @staticmethod
    def project(points):
        points = np.asarray(points, dtype=np.float64)
        # Distinct integer pixels retain a one-to-one Camera-1 correspondence.
        return points[:, :2] * 32. + 8., points[:, 2]


def test_camera_conditioned_field_locks_aligned_region_and_uses_depth_residual():
    x, y = np.meshgrid(np.arange(32), np.arange(32), indexing="xy")
    prior = np.c_[x.reshape(-1), y.reshape(-1), np.full(x.size, 32.)] / 32.
    partial = prior.copy()
    right = prior[:, 0] > .72
    partial[right, 2] = .80
    edited, info, masks = camera_conditioned_gaussian_adaptation(
        prior, partial, _GridProjector(), camera_axes=np.eye(3),
        residual_ratio=.08, stable_ratio=.04, minimum_component_pairs=12,
        influence_geodesic_radius_ratio=.25, maximum_influence_fraction=.9,
        minimum_camera_axis_dominance=.8, minimum_depth_improvement=.02,
    )
    assert info["active"]
    assert info["depth_residual_improvement"] > .02
    assert np.median(edited[right, 2] - prior[right, 2]) < -.05
    # The selected low-residual visible support is an exact zero-displacement
    # boundary. Non-locked neighbours may interpolate the continuous field.
    assert masks["locked"][~right].any()
    assert np.max(np.abs(edited[masks["locked"]] - prior[masks["locked"]])) < 1e-10
    assert masks["moved"].sum() > 0


def test_camera_conditioned_field_preserves_inherited_locked_slots():
    x, y = np.meshgrid(np.arange(32), np.arange(32), indexing="xy")
    prior = np.c_[x.reshape(-1), y.reshape(-1), np.full(x.size, 32.)] / 32.
    partial = prior.copy()
    right = prior[:, 0] > .72
    partial[right, 2] = .80
    inherited = (prior[:, 0] < .35) & (prior[:, 1] > .35)
    edited, info, masks = camera_conditioned_gaussian_adaptation(
        prior, partial, _GridProjector(), camera_axes=np.eye(3),
        residual_ratio=.08, stable_ratio=.04, minimum_component_pairs=12,
        influence_geodesic_radius_ratio=.25, maximum_influence_fraction=.9,
        minimum_camera_axis_dominance=.8, minimum_depth_improvement=.02,
        locked_prior_mask=inherited,
    )
    assert info["active"]
    assert info["inherited_locked_gaussians"] == int(inherited.sum())
    assert np.all(masks["locked"][inherited])
    assert np.allclose(edited[inherited], prior[inherited])
