import unittest

import numpy as np

from scripts.register_complete_to_moge_icp_fallback import build_registration_info


class RegisterCompleteToMogeIcpFallbackTest(unittest.TestCase):
    def test_build_registration_info_exposes_hunyuan_to_moge_for_composition(self):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [1.0, 2.0, 3.0]

        info = build_registration_info(
            complete_path="complete.ply",
            moge_path="moge.ply",
            transform=transform,
            icp_info={"fitness": 0.5},
            registered_path="registered.ply",
            merged_path="merged.ply",
            complete_to_moge_stats={"mean": 0.1},
            moge_to_complete_stats={"mean": 0.2},
        )

        self.assertEqual(info["method"], "fallback_similarity_icp_complete_to_moge")
        self.assertEqual(info["fitness"], 0.5)
        np.testing.assert_allclose(info["hunyuan_to_moge"], transform)
        self.assertEqual(info["outputs"]["registered_complete"], "registered.ply")


if __name__ == "__main__":
    unittest.main()
