import unittest

import numpy as np

from src.observation_conditioned_surface_projection import project_visible_surface_mass


class ObservationConditionedSurfaceProjectionTest(unittest.TestCase):
    def test_mass_and_point_count_are_preserved(self):
        rng = np.random.default_rng(4)
        body = rng.normal(size=(1000, 3))
        partial = rng.normal(size=(500, 3))
        result, info = project_visible_surface_mass(
            body, partial, budget_ratio=.1, seed=7)
        self.assertTrue(info["valid"])
        self.assertEqual(len(result), len(body))
        self.assertEqual(info["complete_prior_points_preserved"], 900)
        self.assertEqual(info["exact_partial_points_inserted"], 100)


if __name__ == "__main__":
    unittest.main()
