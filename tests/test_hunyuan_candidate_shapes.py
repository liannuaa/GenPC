import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

from ScaleAdapter import ScaleAdapter


class HunyuanCandidateShapesTest(unittest.TestCase):
    def test_img2shape_generates_configured_seed_candidates(self):
        with TemporaryDirectory() as tmpdir:
            sample_dir = Path(tmpdir) / "05117"
            sample_dir.mkdir(parents=True)
            Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(sample_dir / "img_sam.png")

            cfg = SimpleNamespace(
                paths=SimpleNamespace(output_dir=tmpdir),
                generative_model="hunyuan2.1",
                skip_existing=False,
                hunyuan_candidate_seeds="101,102",
            )
            adapter = object.__new__(ScaleAdapter)
            adapter.cfg = cfg
            calls = []

            def fake_generative(call_cfg, flag, _img):
                calls.append((flag, getattr(call_cfg, "hunyuan_seed", None)))
                output_name = getattr(call_cfg, "hunyuan_output_ply_name", f"{flag}_hunyuan2.1.ply")
                (Path(tmpdir) / flag / output_name).write_text(str(getattr(call_cfg, "hunyuan_seed", None)))

            adapter.generative = fake_generative

            adapter.img2shape("05117")

            self.assertEqual(calls, [("05117", None), ("05117", 101), ("05117", 102)])
            self.assertTrue((sample_dir / "05117_hunyuan2.1.ply").exists())
            self.assertEqual((sample_dir / "05117_hunyuan2.1_seed101.ply").read_text(), "101")
            self.assertEqual((sample_dir / "05117_hunyuan2.1_seed102.ply").read_text(), "102")


if __name__ == "__main__":
    unittest.main()
