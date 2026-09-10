import numpy as np

from src.observation_anchored_gaussian_reallocation import observation_anchored_gaussian_reallocation


class _GridProjector:
    image_shape = (96, 96)

    @staticmethod
    def project(points):
        points = np.asarray(points, dtype=np.float64)
        return points[:, :2] * 48. + 8., points[:, 2]


def test_observation_anchored_reallocation_stretches_to_partial_without_moving_stable_support():
    x, y = np.meshgrid(np.arange(34), np.arange(34), indexing="xy")
    prior = np.c_[x.reshape(-1), y.reshape(-1), np.full(x.size, 34.)] / 34.
    partial = prior.copy()
    # A connected visible appendage is too short in the prior.  Its observed
    # surface moves in both the camera plane and depth, while the left body is
    # already supported and must remain fixed.
    appendage = prior[:, 0] > .74
    partial[appendage, 0] += .11
    partial[appendage, 2] -= .09
    edited, info, masks = observation_anchored_gaussian_reallocation(
        prior, partial, _GridProjector(), maximum_screen_distance=12., residual_ratio=.045,
        minimum_component_pixels=18, maximum_anchor_residual_ratio=.30,
        influence_radius_ratios=(.16, .24, .32), maximum_influence_fraction=.8,
        maximum_edge_stretch=5.5,
    )
    assert info["active"]
    assert info["anchor_residual_after_median"] < 1e-10
    assert masks["anchors"].sum() >= 18
    stable_left = masks["stable"] & (prior[:, 0] < .55)
    assert stable_left.any()
    assert np.allclose(edited[stable_left], prior[stable_left])
    # Interior screen-plane motion is correspondence-ambiguous on a planar
    # patch; the observable depth change is not.  The test therefore checks
    # exact partial anchors plus a nontrivial continuous continuation rather
    # than pretending that arbitrary same-surface point identities are known.
    assert np.median(edited[appendage, 2] - prior[appendage, 2]) < -.02
    assert (masks["moved"] & ~masks["anchors"]).sum() > 0
