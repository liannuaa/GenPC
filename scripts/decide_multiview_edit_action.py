#!/usr/bin/env python3
"""Choose the next discrete multi-view refinement action without GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.multiview_edit_agent import decide_from_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-reliable-support-ratio", type=float, default=0.50)
    parser.add_argument("--high-coverage-threshold", type=float, default=0.97)
    parser.add_argument("--residual-coverage-floor", type=float, default=0.93)
    parser.add_argument("--residual-p90-tolerance-px", type=float, default=3.0)
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()
    decision = decide_from_files(
        args.incumbent_report,
        args.candidate_report,
        args.output,
        min_reliable_support_ratio=args.min_reliable_support_ratio,
        high_coverage_threshold=args.high_coverage_threshold,
        residual_coverage_floor=args.residual_coverage_floor,
        residual_p90_tolerance_px=args.residual_p90_tolerance_px,
        round_index=args.round_index,
        max_rounds=args.max_rounds,
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
