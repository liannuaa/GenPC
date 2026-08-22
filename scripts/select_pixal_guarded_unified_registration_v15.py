#!/usr/bin/env python3
"""Do-no-harm unified router with a full-resolution fallback guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.select_pixal_unified_registration_v14 as v14
from scripts.run_pixal_batched_adaptive_ttt_v12 import confidence_gate


ROOT = v14.ROOT
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822"


def render_score(metrics):
    return float(metrics["iou"] + .15 * metrics["coverage"]
                 - .45 * metrics["leakage"])


def process(args, sample):
    fast_info = json.loads((args.fast_root / sample /
        f"{sample}_batched_so3_sim3_ttt_v11_info.json").read_text())
    fallback_info = json.loads((args.fallback_root / sample /
        f"{sample}_scale_ttt_v8_info.json").read_text())
    fast_metrics = fast_info["selected"]["metrics"]
    fast_full = fast_info["full_resolution_projection"]
    fallback_full = fallback_info["full_resolution_projection"]
    observable_confident = confidence_gate(fast_metrics)
    fast_full_score = render_score(fast_full)
    fallback_full_score = render_score(fallback_full)
    full_guard = bool(
        fast_full_score >= fallback_full_score - args.full_score_tolerance)
    allow_fast = bool(observable_confident and full_guard)

    original_gate = v14.confidence_gate
    v14.confidence_gate = lambda _metrics: allow_fast
    try:
        result = v14.process(args, sample)
    finally:
        v14.confidence_gate = original_gate
    result.update({
        "method": "guarded_fast_so3_residual_over_genpc_pca_v15",
        "observable_confidence_passed": observable_confident,
        "full_resolution_do_no_harm_passed": full_guard,
        "fast_full_resolution_projection": fast_full,
        "fallback_full_resolution_projection": fallback_full,
        "fast_full_render_score": fast_full_score,
        "fallback_full_render_score": fallback_full_score,
        "full_score_tolerance": args.full_score_tolerance,
    })
    info_path = args.output_root / sample / f"{sample}_unified_registration_v14_info.json"
    info_path.write_text(json.dumps(v14.base.jsonable(result), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=v14.base.CAMERA_ROOT)
    parser.add_argument("--fast-root", type=Path, default=v14.FAST_ROOT)
    parser.add_argument("--fallback-root", type=Path, default=v14.FALLBACK_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=[
        "01184", "05117", "05452", "06127", "06145", "06188", "06830",
        "07136", "07306", "09639"])
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--full-score-tolerance", type=float, default=.005)
    args = parser.parse_args()
    results = [process(args, str(sample)) for sample in args.samples]
    summary = {
        "method": "guarded_fast_so3_residual_over_genpc_pca_v15",
        "ground_truth_used_for_inference_or_selection": False,
        "full_score_tolerance": args.full_score_tolerance,
        "samples": [{
            "sample_id": item["sample_id"],
            "route": item["selected_route"],
            "observable_confidence_passed": item["observable_confidence_passed"],
            "full_resolution_do_no_harm_passed":
                item["full_resolution_do_no_harm_passed"],
            "fast_full_render_score": item["fast_full_render_score"],
            "fallback_full_render_score": item["fallback_full_render_score"],
            "selected_full_projection": item["full_resolution_projection"],
        } for item in results],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "redwood_10_guarded_registration_summary.json").write_text(
        json.dumps(v14.base.jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
