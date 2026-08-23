import unittest

from src.cross_prior_scale_guard import PriorEvidence, choose_prior


class CrossPriorScaleGuardTest(unittest.TestCase):
    def test_selects_supported_fallback_on_opposite_scale_conflict(self):
        current = PriorEvidence(objective=.10, scale=.96, accepted=False)
        fallback = PriorEvidence(objective=.07, scale=1.04, accepted=True)
        self.assertEqual(choose_prior(current, fallback)[0], "fallback")

    def test_keeps_current_without_opposite_scale_directions(self):
        current = PriorEvidence(objective=.10, scale=1.03, accepted=False)
        fallback = PriorEvidence(objective=.07, scale=1.05, accepted=True)
        self.assertEqual(choose_prior(current, fallback)[0], "current")

    def test_keeps_current_when_visible_gain_is_small(self):
        current = PriorEvidence(objective=.10, scale=.96, accepted=False)
        fallback = PriorEvidence(objective=.09, scale=1.04, accepted=True)
        self.assertEqual(choose_prior(current, fallback)[0], "current")


if __name__ == "__main__":
    unittest.main()
