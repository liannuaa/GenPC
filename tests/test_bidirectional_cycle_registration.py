import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.bidirectional_cycle_registration import (
    apply_transform,
    cycle_errors,
    interpolate_sim3,
    invert_proper_sim3,
    robust_fit_similarity,
    sim3_parts,
)


def make_sim3(scale=1.13, angle=0.23, translation=(0.2, -0.1, 0.05)):
    transform = np.eye(4)
    transform[:3, :3] = scale * Rotation.from_rotvec(
        np.array([0.2, -0.4, 0.7]) * angle).as_matrix()
    transform[:3, 3] = translation
    return transform


class BidirectionalCycleRegistrationTest(unittest.TestCase):
    def test_strict_inverse_closes_cycle(self):
        rng = np.random.default_rng(7)
        points = rng.normal(size=(256, 3))
        forward = make_sim3()
        inverse = invert_proper_sim3(forward)
        recovered = apply_transform(apply_transform(points, forward), inverse)
        self.assertLess(np.max(np.abs(recovered - points)), 1e-10)
        error = cycle_errors(forward, inverse, inverse, points, diagonal=3.0)
        self.assertLess(error["exact_inverse_cycle_rms"], 1e-10)

    def test_robust_forward_fit_and_inverse(self):
        rng = np.random.default_rng(11)
        source = rng.normal(size=(500, 3))
        truth = make_sim3(scale=0.91, angle=0.12)
        target = apply_transform(source, truth)
        target += rng.normal(scale=2e-4, size=target.shape)
        target[:80] += rng.normal(scale=0.2, size=(80, 3))
        fitted, kept = robust_fit_similarity(source, target)
        inverse = invert_proper_sim3(fitted)
        recovered = apply_transform(apply_transform(source[kept], fitted), inverse)
        self.assertGreater(kept.sum(), 250)
        self.assertLess(np.mean(np.linalg.norm(recovered - source[kept], axis=1)), 1e-10)

    def test_interpolation_remains_proper_isotropic(self):
        transform = make_sim3()
        for fraction in (0.0, 0.25, 0.5, 1.0):
            step = interpolate_sim3(transform, fraction)
            scale, rotation, _ = sim3_parts(step)
            self.assertGreater(scale, 0.0)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=7)
            singular = np.linalg.svd(step[:3, :3], compute_uv=False)
            self.assertLess(singular.max() - singular.min(), 1e-10)

    def test_rejects_anisotropic_transform(self):
        transform = np.eye(4)
        transform[0, 0] = 1.1
        with self.assertRaises(ValueError):
            sim3_parts(transform)


if __name__ == "__main__":
    unittest.main()
