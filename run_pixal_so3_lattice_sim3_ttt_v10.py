#!/usr/bin/env python3
"""Project-root entry point for the v10 SO(3)-lattice experiment."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent
        / "scripts/run_pixal_so3_lattice_sim3_ttt_v10.py"),
    run_name="__main__",
)
