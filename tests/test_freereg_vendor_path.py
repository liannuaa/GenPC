import sys
import unittest
from pathlib import Path

from scripts.run_freereg_original_depthpro import (
    DEFAULT_FREEREG_ROOT,
    add_freereg_to_path,
    build_ir_3d_candidates,
    parse_float_list,
)


class FreeRegVendorPathTest(unittest.TestCase):
    def test_default_freereg_root_is_vendored_source(self):
        root = add_freereg_to_path(DEFAULT_FREEREG_ROOT)

        self.assertEqual(root, Path("third_party/FreeReg").resolve())
        self.assertTrue((root / "demo.py").exists())
        self.assertIn(str(root), sys.path)
        self.assertIn(str(root / "tools" / "DepthPro" / "src"), sys.path)

    def test_ir_3d_candidates_keep_explicit_value_single_candidate(self):
        candidates = build_ir_3d_candidates(
            auto_ir_3d=0.05,
            explicit_ir_3d=0.2,
            fallback_ir_3d=[0.2, 0.3],
        )

        self.assertEqual(candidates, [{"label": "explicit", "ir_3d": 0.2}])

    def test_ir_3d_candidates_try_auto_then_unique_fallbacks(self):
        candidates = build_ir_3d_candidates(
            auto_ir_3d=0.05,
            explicit_ir_3d=None,
            fallback_ir_3d=parse_float_list("0.05,0.2"),
        )

        self.assertEqual(
            candidates,
            [
                {"label": "auto", "ir_3d": 0.05},
                {"label": "fallback_0.2", "ir_3d": 0.2},
            ],
        )


if __name__ == "__main__":
    unittest.main()
