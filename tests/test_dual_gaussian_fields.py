import numpy as np

from src.dual_gaussian_fields import apply_prior_sim3, make_dual_gaussian_field, observed_anchor_loss


def test_dual_field_locks_partial_and_only_moves_prior_under_sim3():
    prior = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    partial = prior + np.array((2., 0., 0.))
    field = make_dual_gaussian_field(prior, partial)
    transform = np.eye(4); transform[:3, 3] = (3., 0., 0.)
    moved = apply_prior_sim3(field, transform)
    assert np.allclose(moved.means[field.prior_mask], prior + np.array((3., 0., 0.)))
    assert np.allclose(moved.means[field.observed_anchor], partial)
    assert np.all(moved.confidence[field.observed_anchor] == 1.0)


def test_observed_anchor_depth_loss_is_robust_and_masked():
    observed = np.array(((1., 2.), (3., 4.)))
    rendered = np.array(((1.1, 100.), (3.2, 4.3)))
    mask = np.array(((True, False), (True, True)))
    assert np.isclose(observed_anchor_loss(rendered, observed, mask, truncation=.15), (.1 + .15 + .15) / 3.)
