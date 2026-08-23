import unittest

import numpy as np

from src.bidirectional_consensus_registration import (
    _pair_consensus_rms,
    true_bidirectional_pairs,
)


class BidirectionalConsensusRegistrationTest(unittest.TestCase):
    def test_pair_consensus_is_zero_for_identity_pairs(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        pairs = np.c_[np.arange(3), np.arange(3)]
        value = _pair_consensus_rms(
            np.eye(4), points, points, pairs, pairs, diagonal=2.)
        self.assertAlmostEqual(value, 0.)

    def test_true_pair_builder_is_public(self):
        self.assertTrue(callable(true_bidirectional_pairs))


if __name__ == "__main__":
    unittest.main()
