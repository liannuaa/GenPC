import unittest

import numpy as np

from src.voxel_surface_measure_resampling import support_aware_voxel_resample


class VoxelSurfaceMeasureResamplingTest(unittest.TestCase):
    def test_returns_only_input_points_near_target(self):
        rng = np.random.default_rng(4)
        prior = np.c_[rng.random(8000), rng.random(8000), np.zeros(8000)]
        observation = np.c_[rng.random(1000), rng.random(1000), np.zeros(1000)]
        points = np.r_[prior, observation]
        result, info = support_aware_voxel_resample(
            points, observation_count=1000, target_points=3000)
        self.assertLess(abs(len(result) - 3000), 100)
        self.assertTrue(info["subset_of_input"])
        source = {tuple(row) for row in points}
        self.assertTrue(all(tuple(row) in source for row in result))

    def test_removes_only_unsupported_isolated_observation(self):
        rng = np.random.default_rng(8)
        prior = rng.normal(scale=.02, size=(4000, 3))
        observation = rng.normal(scale=.02, size=(500, 3))
        observation[-1] = [4., 4., 4.]
        _, info = support_aware_voxel_resample(
            np.r_[prior, observation], observation_count=500,
            target_points=2000, support_ratio=.02)
        self.assertGreaterEqual(info["observation_outliers_removed"], 1)


if __name__ == "__main__":
    unittest.main()
