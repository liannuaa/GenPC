#!/usr/bin/env python3
"""Run the fixed observation-anchored fusion over independent samples."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_one(root: Path, output_root: Path, final_root: Path, sample: str) -> dict:
    posterior = root / "posterior" / sample / "posterior_prior_100k.ply"
    partial = root / "inputs" / "partial" / f"{sample}.ply"
    manifest = root / "residuals" / sample / "render" / "render_manifest.json"
    required = (posterior, partial, manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{sample}: missing inputs: {missing}")
    output = output_root / sample
    output.mkdir(parents=True, exist_ok=True)
    log = output / "stage.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_observation_anchored_fusion.py"),
        "--posterior", str(posterior),
        "--partial", str(partial),
        "--multiview-manifest", str(manifest),
        "--output-dir", str(output),
    ]
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    if result.returncode:
        raise RuntimeError(f"{sample}: fusion failed; see {log}")
    prediction = output / "observation_anchored_fused_100k.ply"
    destination = final_root / sample / "complete_100k.ply"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prediction, destination)
    info = json.loads((output / "fusion_info.json").read_text(encoding="utf-8"))
    return {
        "sample_id": sample,
        "status": "complete",
        "prediction": str(prediction.resolve()),
        "canonical_final": str(destination.resolve()),
        "fusion": info["fusion"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--final-root", type=Path)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--sample-workers", type=int, default=2)
    args = parser.parse_args()
    if args.sample_workers < 1:
        parser.error("--sample-workers must be positive")
    root = args.input_root.resolve()
    output_root = (args.output_root or root / "fusion").resolve()
    final_root = (args.final_root or root / "final").resolve()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(args.sample_workers)) as executor:
        futures = {
            executor.submit(_run_one, root, output_root, final_root, str(sample)): str(sample)
            for sample in args.samples
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                record = future.result()
                print(f"[complete] {sample}", flush=True)
            except Exception as error:
                record = {"sample_id": sample, "status": "failed", "error": str(error)}
                print(f"[failed] {sample}: {error}", flush=True)
            records.append(record)
    records.sort(key=lambda item: item["sample_id"])
    manifest = {
        "method": "four_view_observation_anchored_carrier_fusion",
        "ground_truth_used": False,
        "point_concatenation_used": False,
        "input_root": str(root),
        "output_root": str(output_root),
        "final_root": str(final_root),
        "shared_parameters": {
            "num_views": 4,
            "pixel_radius": 1.0,
            "cross_view_pixel_radius": 2.0,
            "cross_view_depth_ratio": 0.075,
            "anchor_residual_ratio": 0.075,
        },
        "samples": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    if any(record["status"] == "failed" for record in records):
        raise SystemExit("one or more fusion samples failed")


if __name__ == "__main__":
    main()
