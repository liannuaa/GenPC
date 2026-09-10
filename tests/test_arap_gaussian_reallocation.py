import numpy as np

from src.arap_gaussian_reallocation import partial_depth_arap_gaussian_adaptation


class _GridProjector:
    image_shape = (96, 96)

    @staticmethod
    def project(points):
        points = np.asarray(points, dtype=np.float64)
        return points[:, :2] * 48. + 8., points[:, 2]


def test_partial_depth_arap_keeps_stable_surface_and_exactly_observes_residual_patch():
    x, y = np.meshgrid(np.arange(34), np.arange(34), indexing="xy")
    prior = np.c_[x.reshape(-1), y.reshape(-1), np.full(x.size, 34.)] / 34.
    partial = prior.copy()
    appendage = prior[:, 0] > .74
    partial[appendage, 2] -= .07
    edited, info, masks = partial_depth_arap_gaussian_adaptation(
        prior, partial, _GridProjector(), maximum_screen_distance=6., residual_ratio=.04,
        minimum_component_pixels=18, maximum_anchor_residual_ratio=.25,
        influence_radius_ratios=(.16, .24, .32), maximum_influence_fraction=.8,
        maximum_edge_stretch=3.5, minimum_edge_compression=.35,
    )
    assert info["active"]
    assert info["anchor_residual_after_median"] < 1e-10
    stable_left = masks["stable"] & (prior[:, 0] < .55)
    assert stable_left.any()
    assert np.allclose(edited[stable_left], prior[stable_left])
    assert (masks["moved"] & ~masks["anchors"]).sum() > 0
