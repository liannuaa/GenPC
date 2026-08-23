#!/usr/bin/env python3
"""Select final surface predictions using a shared cross-prior scale guard."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_prior_scale_guard import PriorEvidence, choose_prior


DEFAULT_SAMPLES = (
    "01184", "05117", "05452", "06127", "06145",
    "06188", "06830", "07136", "07306", "09639",
)


def _registration_evidence(root: Path, sample: str) -> PriorEvidence:
    path = root / sample / f"{sample}_bidirectional_cycle_info.json"
    info = json.loads(path.read_text())
    return PriorEvidence(
        objective=float(info["final_visible_score"]["objective"]),
        scale=float(info["cumulative_scale"]),
        accepted=bool(info["accepted"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-registration-root", type=Path, required=True)
    parser.add_argument("--current-surface-root", type=Path, required=True)
    parser.add_argument("--fallback-registration-root", type=Path, required=True)
    parser.add_argument("--fallback-surface-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    args = parser.parse_args()

    for sample in args.samples:
        current = _registration_evidence(args.current_registration_root, sample)
        fallback = _registration_evidence(args.fallback_registration_root, sample)
        selected, reason = choose_prior(current, fallback)
        surface_root = (
            args.fallback_surface_root if selected == "fallback"
            else args.current_surface_root
        )
        source_dir = surface_root / sample
        output_dir = args.output_root / sample
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{sample}_bidirectional_cycle"
        for suffix in (
            "_registered_100k.ply",
            "_partial_gray_pixal_red.ply",
            "_projection.png",
            "_surface_projection_info.json",
        ):
            shutil.copy2(source_dir / f"{stem}{suffix}", output_dir / f"{stem}{suffix}")

        audit = {
            "sample_id": sample,
            "method": "cross_prior_scale_consistency_guard",
            "strict_zero_shot": True,
            "ground_truth_used_for_selection": False,
            "sample_or_category_specific_parameters": False,
            "selected": selected,
            "reason": reason,
            "shared_parameters": {
                "min_scale_disagreement": 0.06,
                "objective_ratio": 0.80,
                "require_opposite_scale_directions": True,
                "require_fallback_accepted_and_current_rejected": True,
            },
            "current": current.__dict__,
            "fallback": fallback.__dict__,
            "source_surface_root": str(surface_root),
        }
        (output_dir / f"{stem}_cross_prior_guard_info.json").write_text(
            json.dumps(audit, indent=2)
        )
        print(sample, selected, reason)


if __name__ == "__main__":
    main()
