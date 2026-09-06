"""Unit tests for the bounded MLLM-to-geometry action contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.agentic_probe_state import AgentDecision, ProbeState


class AgenticProbeStateTest(unittest.TestCase):
    def _decision(self, state: ProbeState, action: str) -> AgentDecision:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps({
                "state_sha256": state.state_hash(),
                "action": {"name": action, "arguments": {}},
                "rationale": "Bounded test action.",
                "planner": "test-MLLM",
                "ground_truth_used": False,
            }), encoding="utf-8")
            return AgentDecision.from_file(path)

    def test_current_state_hash_accepts_permitted_action(self):
        state = ProbeState(sample_id="01184", phase="view_candidates", budget_remaining=8)
        decision = self._decision(state, "SELECT_VIEW")
        decision.validate(state)

    def test_stale_state_hash_is_rejected(self):
        state = ProbeState(sample_id="01184", phase="view_candidates", budget_remaining=8)
        decision = self._decision(state, "SELECT_VIEW")
        state.history.append({"step": 1})
        with self.assertRaisesRegex(ValueError, "state_sha256"):
            decision.validate(state)

    def test_phase_forbids_skipping_geometry(self):
        state = ProbeState(sample_id="07136", phase="semantic_pending", budget_remaining=8)
        decision = self._decision(state, "ACCEPT")
        with self.assertRaisesRegex(ValueError, "invalid for phase"):
            decision.validate(state)

    def test_ground_truth_flag_is_rejected(self):
        state = ProbeState(sample_id="01184", phase="view_candidates", budget_remaining=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps({
                "state_sha256": state.state_hash(),
                "action": {"name": "SELECT_VIEW", "arguments": {}},
                "rationale": "Invalid test action.",
                "planner": "test-MLLM",
                "ground_truth_used": True,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ground truth"):
                AgentDecision.from_file(path).validate(state)


if __name__ == "__main__":
    unittest.main()
