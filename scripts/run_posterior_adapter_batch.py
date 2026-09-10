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
        "multiview_manifest": base / "residuals" / sample / "render" / "render_manifest.json",
    }


def _estimate_summary(estimate: dict) -> dict:
    diagnostics = estimate.get("posterior_diagnostics", {})
    integrity = estimate.get("integrity", {})
    return {
        "active": bool(estimate.get("active", False)),
        "visible_objective_delta": diagnostics.get("objective_delta"),
        "carrier_slots_preserved": bool(estimate.get("carrier_slots_preserved", False)),
        "new_connected_components": integrity.get("new_connected_components"),
        "hidden_coverage_ratio": estimate.get("hidden_coverage_ratio"),
    }


def _run_one(paths: dict[str, Path], output_root: Path, sample: str,
             device: str, overwrite: bool,
             integrated_observation_fusion: bool,
             config_json: Path | None) -> dict:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{sample}: missing inputs: {missing}")
    output = output_root / sample
    info_path = output / "posterior_info.json"
    if info_path.is_file() and not overwrite:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        return {"sample_id": sample, "status": "reused", "output": str(output.resolve()),
                "estimate": _estimate_summary(payload.get("estimate", {}))}
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts" / "run_posterior_adapter.py"),
        "--prior", str(paths["prior"]), "--partial", str(paths["partial"]),
        "--camera", str(paths["camera"]), "--semantic", str(paths["semantic"]),
        "--multiview-manifest", str(paths["multiview_manifest"]),
        "--output-dir", str(output), "--device", str(device),
    ]
    if integrated_observation_fusion:
        command.append("--integrated-observation-fusion")
    if config_json is not None:
        command.extend(["--config-json", str(config_json)])
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
        "estimate": _estimate_summary(payload.get("estimate", {})),
    }


def _load_case_manifest(path: Path) -> list[tuple[str, dict[str, Path]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("case manifest must be a list or contain a 'samples' list")
    cases = []
    for record in records:
        sample = str(record["sample_id"])
        paths = {name: Path(record[name]).expanduser().resolve()
                 for name in ("prior", "partial", "camera", "semantic", "multiview_manifest")}
        cases.append((sample, paths))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--case-manifest", type=Path,
        help="JSON list with sample_id/prior/partial/camera/semantic/multiview_manifest paths.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+")
    parser.add_argument(
        "--prior-template",
        default="registration/{sample}/final/camera1_amplified_registered_100k.ply",
        help="Prior path relative to each resolved sample base; {sample} is expanded.",
    )
    parser.add_argument("--sample-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--integrated-observation-fusion",
        action="store_true",
        help="Assimilate four-view partial evidence inside the smooth posterior deformation.",
    )
    parser.add_argument(
        "--config-json", type=Path,
        help="Shared PosteriorAdapterConfig override JSON used unchanged for every sample.",
    )
    args = parser.parse_args()
    if int(args.sample_workers) < 1:
        parser.error("--sample-workers must be positive")
    if bool(args.case_manifest) == bool(args.input_root):
        parser.error("provide exactly one of --case-manifest or --input-root")
    if args.input_root and not args.samples:
        parser.error("--samples is required with --input-root")
    output_root = args.output_root.resolve()
    config_json = args.config_json.resolve() if args.config_json is not None else None
    if config_json is not None and not config_json.is_file():
        parser.error(f"--config-json does not exist: {config_json}")
    if args.case_manifest:
        cases = _load_case_manifest(args.case_manifest.resolve())
        input_root = None
    else:
        input_root = args.input_root.resolve()
        cases = [
            (str(sample), _sample_inputs(input_root, str(sample), str(args.prior_template)))
            for sample in args.samples
        ]
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(args.sample_workers)) as executor:
        futures = {
            executor.submit(
                _run_one, paths, output_root, sample, str(args.device), bool(args.overwrite),
                bool(args.integrated_observation_fusion), config_json,
            ): sample
            for sample, paths in cases
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
        "input_root": str(input_root) if input_root else None,
        "case_manifest": str(args.case_manifest.resolve()) if args.case_manifest else None,
        "output_root": str(output_root),
        "prior_template": str(args.prior_template),
        "sample_workers": int(args.sample_workers), "samples": records,
        "integrated_observation_fusion": bool(args.integrated_observation_fusion),
        "posterior_config": str(config_json) if config_json is not None else None,
    }
    (output_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} sample(s) failed")


if __name__ == "__main__":
    main()
