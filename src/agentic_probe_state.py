"""Auditable state and decision contracts for the isolated agentic probe.

This module intentionally contains no geometry optimizer and no learned policy.
It makes the boundary explicit: an MLLM supplies one validated *discrete*
decision for the current state, while the runner dispatches a fixed geometric
tool and records a fresh verifier state before another decision is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


SELECT_VIEW = "SELECT_VIEW"
COMPLETE_SEMANTIC = "COMPLETE_SEMANTIC"
GENERATE_PRIOR = "GENERATE_PRIOR"
ALIGN_GLOBAL = "ALIGN_GLOBAL"
REFINE_ALIGNMENT = "REFINE_ALIGNMENT"
ADAPT_LOCAL = "ADAPT_LOCAL"
ACCEPT = "ACCEPT"

ACTION_NAMES = (
    SELECT_VIEW,
    COMPLETE_SEMANTIC,
    GENERATE_PRIOR,
    ALIGN_GLOBAL,
    REFINE_ALIGNMENT,
    ADAPT_LOCAL,
    ACCEPT,
)

# A deliberately compact tool graph.  There is no free-form numerical action:
# all continuous pose, scale, and local geometry updates remain inside the
# fixed mainline solvers called by the probe runner.
ALLOWED_ACTIONS = {
    "view_candidates": (SELECT_VIEW,),
    "semantic_pending": (COMPLETE_SEMANTIC,),
    "prior_pending": (GENERATE_PRIOR,),
    "global_alignment_pending": (ALIGN_GLOBAL,),
    "alignment_diagnosis": (REFINE_ALIGNMENT, ADAPT_LOCAL, ACCEPT),
    "adaptation_diagnosis": (ADAPT_LOCAL, ACCEPT),
    "final_diagnosis": (ADAPT_LOCAL, ACCEPT),
    "accepted": (),
}


def canonical_json(value: Any) -> str:
    """Serialize state deterministically so decisions are bound to evidence."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass
class ProbeState:
    """Minimal persisted state for one sample and one bounded tool budget."""

    sample_id: str
    phase: str
    budget_remaining: int
    history: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    strict_zero_shot: bool = True
    ground_truth_cd_emd_used: bool = False

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sample_id": self.sample_id,
            "phase": self.phase,
            "budget_remaining": int(self.budget_remaining),
            "history": self.history,
            "artifacts": self.artifacts,
            "diagnostics": self.diagnostics,
            "strict_zero_shot": bool(self.strict_zero_shot),
            "ground_truth_cd_emd_used": bool(self.ground_truth_cd_emd_used),
        }

    def state_hash(self) -> str:
        return content_hash(self.body())

    def as_dict(self) -> dict[str, Any]:
        payload = self.body()
        payload["state_sha256"] = self.state_hash()
        payload["allowed_actions"] = list(ALLOWED_ACTIONS.get(self.phase, ()))
        return payload

    @classmethod
    def from_file(cls, path: Path) -> "ProbeState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            sample_id=str(payload["sample_id"]),
            phase=str(payload["phase"]),
            budget_remaining=int(payload["budget_remaining"]),
            history=list(payload.get("history", [])),
            artifacts={str(key): str(value) for key, value in payload.get("artifacts", {}).items()},
            diagnostics=dict(payload.get("diagnostics", {})),
            strict_zero_shot=bool(payload.get("strict_zero_shot", True)),
            ground_truth_cd_emd_used=bool(payload.get("ground_truth_cd_emd_used", False)),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class AgentDecision:
    """One MLLM-produced, state-bound action with a human-auditable rationale."""

    state_sha256: str
    action: str
    arguments: dict[str, Any]
    rationale: str
    planner: str
    ground_truth_used: bool

    @classmethod
    def from_file(cls, path: Path) -> "AgentDecision":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        action_payload = payload.get("action", payload)
        if not isinstance(action_payload, dict):
            raise ValueError("agent decision must contain an action object")
        return cls(
            state_sha256=str(payload.get("state_sha256", "")),
            action=str(action_payload.get("name", action_payload.get("action", ""))),
            arguments=dict(action_payload.get("arguments", {})),
            rationale=str(payload.get("rationale", "")).strip(),
            planner=str(payload.get("planner", "")).strip(),
            ground_truth_used=bool(payload.get("ground_truth_used", False)),
        )

    def validate(self, state: ProbeState) -> None:
        if self.ground_truth_used:
            raise ValueError("agent decisions must not use ground truth, CD, or EMD")
        if self.state_sha256 != state.state_hash():
            raise ValueError("decision state_sha256 does not match the latest verifier state")
        if self.action not in ACTION_NAMES:
            raise ValueError(f"unsupported agent action: {self.action}")
        if self.action not in ALLOWED_ACTIONS.get(state.phase, ()):
            allowed = ", ".join(ALLOWED_ACTIONS.get(state.phase, ())) or "none"
            raise ValueError(f"{self.action} is invalid for phase {state.phase}; allowed: {allowed}")
        if not self.rationale:
            raise ValueError("agent decision requires a non-empty rationale")
        if not self.planner:
            raise ValueError("agent decision must identify the planner")

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_sha256": self.state_sha256,
            "action": {"name": self.action, "arguments": self.arguments},
            "rationale": self.rationale,
            "planner": self.planner,
            "ground_truth_used": self.ground_truth_used,
        }
