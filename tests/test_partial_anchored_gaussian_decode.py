import numpy as np

from src.partial_anchored_gaussian_decode import decode_partial_anchored_gaussians


def test_collision_free_decode_replaces_only_matched_complete_slots():
    prior = np.array(((0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (3., 0., 0.)))
    partial = np.array(((.1, 0., 0.), (1.1, 0., 0.), (2.8, 0., 0.)))
    # The first two anchors compete for slot 0; the closer one wins.  The
    # third replaces slot 3 and the unmatched prior slots remain untouched.
    pairs = np.array(((0, 0, 0, 0.), (1, 0, 0, .1), (2, 3, 3, 0.)))
    decoded, selected, info = decode_partial_anchored_gaussians(
        prior, partial, pairs, max_residual=.3,
    )
    assert len(decoded) == len(prior)
    assert np.array_equal(selected, np.array(((0, 0), (2, 3))))
    assert np.allclose(decoded[0], partial[0])
    assert np.allclose(decoded[3], partial[2])
    assert np.allclose(decoded[1:3], prior[1:3])
    assert info["all_prior_slots_preserved"]
