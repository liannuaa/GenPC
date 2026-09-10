#!/usr/bin/env python3
"""Run one frozen PosteriorAdapter configuration over independent samples."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _sample_inputs(input_root: Path, sample: str, prior_template: str) -> dict[str, Path]:
    """Resolve either a flat collection or per-sample agent trace layout."""
    base = input_root if (input_root / "inputs").is_dir() else input_root / sample
    return {
        "prior": base / prior_template.format(sample=sample),
        "partial": base / "inputs" / "partial" / f"{sample}.ply",
        "camera": base / "inputs" / "camera" / sample / "camera.pth",
        "semantic": base / "inputs" / "camera" / sample / "img.png",
    }


def _run_one(input_root: Path, output_root: Path, sample: str, prior_template: str,
             device: str, overwrite: bool) -> dict:
    paths = _sample_inputs(input_root, sample, prior_template)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{sample}: missing inputs: {missing}")
    output = output_root / sample
    info_path = output / "posterior_info.json"
    if info_path.is_file() and not overwrite:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        return {"sample_id": sample, "status": "reused", "output": str(output.resolve()),
                "estimate": payload.get("estimate", {})}
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts" / "run_posterior_adapter.py"),
        "--prior", str(paths["prior"]), "--partial", str(paths["partial"]),
        "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
        "--output-dir", str(output), "--device", str(device),
    ]
    started = time.time()
    log_path = output / "stage.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{sample}: PosteriorAdapter failed; see {log_path}")
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    return {
        "sample_id": sample, "status": "completed", "elapsed_seconds": time.time() - started,
        "output": str(output.resolve()), "inputs": {key: str(value.resolve()) for key, value in paths.items()},
        "estimate": payload.get("estimate", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument(
        "--prior-template",
        default="registration/{sample}/final/camera1_amplified_registered_100k.ply",
        help="Prior path relative to each resolved sample base; {sample} is expanded.",
    )
    parser.add_argument("--sample-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.sample_workers) < 1:
        parser.error("--sample-workers must be positive")
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(args.sample_workers)) as executor:
        futures = {
            executor.submit(
                _run_one, input_root, output_root, str(sample), str(args.prior_template),
                str(args.device), bool(args.overwrite),
            ): str(sample)
            for sample in args.samples
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                record = future.result()
                print(f"[{record['status']}] {sample}", flush=True)
            except Exception as error:
                record = {"sample_id": sample, "status": "failed", "error": str(error)}
                print(f"[failed] {sample}: {error}", flush=True)
            records.append(record)
    records.sort(key=lambda row: row["sample_id"])
    failures = [row for row in records if row["status"] == "failed"]
    manifest = {
        "method": "structure_aware_partial_ot_complete_gaussian_posterior",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "category_or_part_rules_used": False,
        "input_root": str(input_root), "output_root": str(output_root),
        "prior_template": str(args.prior_template),
        "sample_workers": int(args.sample_workers), "samples": records,
    }
    (output_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} sample(s) failed")


if __name__ == "__main__":
    main()
