import sys
import unittest
from pathlib import Path

from scripts.run_freereg_original_depthpro import (
    DEFAULT_FREEREG_ROOT,
    add_freereg_to_path,
)


class FreeRegVendorPathTest(unittest.TestCase):
    def test_default_freereg_root_is_vendored_source(self):
        root = add_freereg_to_path(DEFAULT_FREEREG_ROOT)

        self.assertEqual(root, Path("third_party/FreeReg").resolve())
        self.assertTrue((root / "demo.py").exists())
        self.assertIn(str(root), sys.path)
        self.assertIn(str(root / "tools" / "DepthPro" / "src"), sys.path)


if __name__ == "__main__":
    unittest.main()
