"""Unit tests for the bounded MLLM-to-geometry action contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.agentic_probe_state import AgentDecision, ProbeState


class AgenticProbeStateTest(unittest.TestCase):
    def _decision(self, state: ProbeState, action: str, arguments: dict | None = None) -> AgentDecision:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps({
                "state_sha256": state.state_hash(),
                "action": {"name": action, "arguments": arguments or {}},
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

    def test_alignment_diagnosis_allows_bounded_view_replan(self):
        state = ProbeState(sample_id="mvp_test_26167", phase="alignment_diagnosis", budget_remaining=6)
        decision = self._decision(state, "REPLAN_VIEW")
        decision.validate(state)

    def test_alignment_diagnosis_allows_restoring_an_audited_attempt(self):
        state = ProbeState(sample_id="mvp_test_35832", phase="alignment_diagnosis", budget_remaining=6)
        decision = self._decision(state, "RESTORE_ATTEMPT")
        decision.validate(state)

    def test_alignment_diagnosis_allows_zero_pair_global_rescue(self):
        state = ProbeState(sample_id="mvp_test_21341", phase="alignment_diagnosis", budget_remaining=6)
        decision = self._decision(state, "RESCUE_GLOBAL")
        decision.validate(state)

    def test_prior_pending_allows_explicit_pose_locked_clarity_tool(self):
        state = ProbeState(sample_id="domestic_pig", phase="prior_pending", budget_remaining=6)
        decision = self._decision(
            state, "REFINE_SEMANTIC", {"source": "/tmp/pose_locked_gpt_clarity.png"},
        )
        decision.validate(state)

    def test_clarity_tool_rejects_missing_external_source(self):
        state = ProbeState(sample_id="domestic_pig", phase="prior_pending", budget_remaining=6)
        decision = self._decision(state, "REFINE_SEMANTIC")
        with self.assertRaisesRegex(ValueError, "requires an explicit external source"):
            decision.validate(state)

    def test_conditioning_diagnosis_allows_only_a_named_image_candidate(self):
        state = ProbeState(sample_id="domestic_pig", phase="conditioning_diagnosis", budget_remaining=5)
        self._decision(state, "SELECT_CONDITIONING_IMAGE", {"source": "clarity"}).validate(state)
        invalid = self._decision(state, "SELECT_CONDITIONING_IMAGE", {"source": "other"})
        with self.assertRaisesRegex(ValueError, "must be qwen or clarity"):
            invalid.validate(state)

    def test_adaptation_diagnosis_allows_axis_scale_before_local_edit(self):
        state = ProbeState(sample_id="domestic_pig", phase="adaptation_diagnosis", budget_remaining=4)
        self._decision(state, "ADAPT_AXIS_SCALE").validate(state)
        state.phase = "axis_scale_diagnosis"
        invalid = self._decision(state, "ADAPT_AXIS_SCALE")
        with self.assertRaisesRegex(ValueError, "invalid for phase"):
            invalid.validate(state)

    def test_compact_agent_can_request_only_parameter_free_posterior_update(self):
        state = ProbeState(sample_id="generic_case", phase="posterior_diagnosis", budget_remaining=2)
        self._decision(state, "UPDATE_POSTERIOR").validate(state)
        invalid = self._decision(state, "UPDATE_POSTERIOR", {"axis": 0, "scale": 1.2})
        with self.assertRaisesRegex(ValueError, "does not accept"):
            invalid.validate(state)


if __name__ == "__main__":
    unittest.main()
