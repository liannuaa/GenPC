#!/usr/bin/env python3
"""Write one validated, state-bound decision for the isolated probe.

This helper only serializes an already chosen discrete planner action.  It
does not choose numerical registration or editing parameters and refuses any
decision that is not valid for the current persisted state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_probe_state import AgentDecision, ProbeState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--strategy")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--planner", default="Codex GPT-5.6 Terra")
    args = parser.parse_args()

    state = ProbeState.from_file(args.state)
    action_args: dict[str, str] = {}
    if args.candidate is not None:
        action_args["candidate"] = args.candidate
    if args.strategy is not None:
        action_args["strategy"] = args.strategy
    decision = AgentDecision(
        state_sha256=state.state_hash(),
        action=args.action,
        arguments=action_args,
        rationale=args.rationale,
        planner=args.planner,
        ground_truth_used=False,
    )
    decision.validate(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision.as_dict(), indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
