import unittest

import numpy as np

from scripts.compose_complete_to_partial import compose_complete_to_partial


class ComposeCompleteToPartialTest(unittest.TestCase):
    def test_compose_complete_to_partial_multiplies_moge_to_partial_after_complete_to_moge(self):
        complete_to_moge = np.eye(4, dtype=np.float64)
        complete_to_moge[:3, 3] = [1.0, 2.0, 3.0]
        moge_to_partial = np.eye(4, dtype=np.float64)
        moge_to_partial[:3, :3] *= 2.0
        moge_to_partial[:3, 3] = [-1.0, 0.0, 1.0]

        complete_to_partial = compose_complete_to_partial(
            complete_to_moge=complete_to_moge,
            moge_to_partial=moge_to_partial,
        )

        expected = moge_to_partial @ complete_to_moge
        np.testing.assert_allclose(complete_to_partial, expected)
        np.testing.assert_allclose(complete_to_partial[:3, 3], [1.0, 4.0, 7.0])


if __name__ == "__main__":
    unittest.main()
