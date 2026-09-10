#!/usr/bin/env python3
"""Run the bounded no-GT recovery policy over replayed agentic outputs.

The script is intentionally a *post-alignment* controller.  It reuses an
existing zero-shot attempt as a baseline, leaves samples with adequate verifier
evidence untouched, and invokes the extra Sim(3) recovery either for a true
Camera-1 evidence failure or as a verifier-compared rival for clearly weak
global evidence.  Thus it is suitable for sparse datasets such as MVP without
changing the frozen Redwood mainline.

All decisions are serialized as normal state-hash-bound agent decisions.  The
controller never opens a complete cloud, CD, or EMD file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_probe_state import ACCEPT, ADAPT_LOCAL, AgentDecision, ProbeState, REFINE_ALIGNMENT, RESCUE_GLOBAL
from src.agentic_recovery_policy import (
    evidence_quality,
    has_usable_visible_evidence,
    needs_global_recovery,
    needs_global_rescue_probe,
    prefer_rescue_over_refinement,
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "agentic_probe_runner", ROOT / "scripts" / "run_agentic_01184_probe.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the probe runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _link(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"conflicting target artifact: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _link_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            _link(item, destination)


def _source_prediction(source_root: Path, sample: str) -> tuple[Path, dict[str, Any]]:
    sample_root = source_root / "samples" / sample
    state_path = sample_root / "agent_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    source = Path(str(payload.get("artifacts", {}).get("selected_source", "")))
    if not source.is_file():
        source = sample_root / "final" / "agent_selected_100k.ply"
    if not source.is_file():
        raise FileNotFoundError(f"cannot locate selected source for {sample}")
    return source, payload


def _materialize_baseline(runner, source_root: Path, output_root: Path, sample: str, budget: int) -> Path:
    """Create a compact, hard-linked replay root with a fresh agent state."""
    source_sample = source_root / "samples" / sample
    target = output_root / "samples" / sample
    if target.exists():
        raise FileExistsError(f"target sample already exists: {target}")
    source_prediction, old_state = _source_prediction(source_root, sample)
    source_partial = source_root / "inputs" / "partial" / f"{sample}.ply"
    _link(source_partial, target / "inputs" / "partial" / f"{sample}.ply")
    _link_tree(source_sample / "inputs" / "camera" / sample, target / "inputs" / "camera" / sample)
    _link_tree(source_sample / "view_candidates", target / "view_candidates")

    old_context = json.loads((source_sample / "agent_input.json").read_text(encoding="utf-8"))
    context = {
        "partial": str((target / "inputs" / "partial" / f"{sample}.ply").resolve()),
        "object_label": str(old_context["object_label"]),
        "dataset_adapter": "replayed_agentic_baseline_with_no_gt_recovery",
        "ground_truth_cd_emd_used": False,
    }
    (target / "agent_input.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    _link(source_prediction, target / "baseline" / "agent_selected_100k.ply")

    runner.SAMPLE = sample
    paths = runner._paths(target)
    # A replayed carrier is the starting proposal.  Keep it in the joint slot
    # even if it was previously Gaussian-adapted: a newly emitted rescue must
    # supersede the baseline, while a residual refinement supersedes rescue
    # through final_prior.  The original selected PLY remains hard-linked in
    # ``baseline/`` for an exact verifier-selected rollback.
    destination = paths["joint_prior"]
    _link(source_prediction, destination)

    state = ProbeState(
        sample_id=sample,
        phase="alignment_diagnosis",
        budget_remaining=int(budget),
        artifacts={
            "partial": str(paths["partial"].resolve()),
            "candidate": "base",
            "tried_view_candidates": "base",
            "baseline_prediction": str((target / "baseline" / "agent_selected_100k.ply").resolve()),
            "replayed_source": str(source_prediction.resolve()),
            "replayed_history_length": str(len(old_state.get("history", []))),
        },
        diagnostics={},
    )
    runner._write_diagnostics(target, state)
    state.write(target / runner.STATE_FILENAME)
    runner._write_decision_packet(target, state)
    return target


def _apply(runner, root: Path, sample: str, action: str, rationale: str, *, candidate: str | None = None) -> ProbeState:
    runner.SAMPLE = sample
    state_path = root / runner.STATE_FILENAME
    state = ProbeState.from_file(state_path)
    arguments: dict[str, str] = {} if candidate is None else {"candidate": candidate}
    decision = AgentDecision(
        state_sha256=state.state_hash(), action=action, arguments=arguments,
        rationale=rationale, planner="Verifier-driven bounded recovery policy", ground_truth_used=False,
    )
    decision.validate(state)
    decision_path = root / "decisions" / f"{len(state.history) + 1:02d}_{action.lower()}.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision.as_dict(), indent=2) + "\n", encoding="utf-8")
    runner.apply(SimpleNamespace(root=root, decision=decision_path))
    return ProbeState.from_file(state_path)


def _select_final_candidate(state: ProbeState, *, rescue_diagnostics: dict[str, Any] | None = None,
                            registered_diagnostics: dict[str, Any] | None = None,
                            baseline_diagnostics: dict[str, Any] | None = None) -> str:
    """Choose a discrete already-produced carrier using verifier evidence."""
    current = state.diagnostics
    if rescue_diagnostics is not None and prefer_rescue_over_refinement(rescue_diagnostics, current):
        return "rescue"
    if registered_diagnostics is not None:
        if not has_usable_visible_evidence(current):
            return "registered"
        if evidence_quality(registered_diagnostics) < evidence_quality(current):
            return "registered"
    if baseline_diagnostics is not None:
        if not has_usable_visible_evidence(current):
            return "baseline"
        if evidence_quality(baseline_diagnostics) < evidence_quality(current):
            return "baseline"
    return "current"


def _run_one(runner, root: Path, sample: str) -> dict[str, Any]:
    state = ProbeState.from_file(root / runner.STATE_FILENAME)
    baseline = dict(state.diagnostics)
    if not needs_global_rescue_probe(baseline):
        final = _apply(
            runner, root, sample, ACCEPT,
            "Finite Camera-1 visible evidence is already available; preserve the baseline rather than spend recovery budget.",
        )
        return {"sample": sample, "route": ["ACCEPT"], "accepted_candidate": "current", "final": final.diagnostics}

    zero_evidence = needs_global_recovery(baseline)
    rescued = _apply(
        runner, root, sample, RESCUE_GLOBAL,
        "The verifier has fewer than six positive visible pairs or non-finite energy, indicating a global camera-gauge failure rather than a local shape residual.",
    )
    rescue_diagnostics = dict(rescued.diagnostics)
    if not has_usable_visible_evidence(rescue_diagnostics):
        # The controller intentionally stops here.  A semantic/prior replan is
        # a separate expensive tool branch and should be approved with a new
        # state decision instead of silently regenerating a model.
        candidate = "current" if zero_evidence else "baseline"
        final = _apply(
            runner, root, sample, ACCEPT,
            "The bounded rescue did not establish enough visible evidence for a safe residual solve; retain the complete rescue carrier without unsupported local deformation.",
            candidate=None if candidate == "current" else candidate,
        )
        return {"sample": sample, "route": ["RESCUE_GLOBAL", "ACCEPT"], "accepted_candidate": candidate, "final": final.diagnostics}

    # A weak-but-finite baseline is a valid candidate.  Treat recovery as a
    # proposal only; retain it without another solver call if it does not
    # improve the no-GT verifier.
    if not zero_evidence and evidence_quality(baseline) <= evidence_quality(rescue_diagnostics):
        final = _apply(
            runner, root, sample, ACCEPT,
            "The bounded recovery proposal did not beat the existing finite verifier evidence, so preserve the replayed complete prior.",
            candidate="baseline",
        )
        return {"sample": sample, "route": ["RESCUE_GLOBAL", "ACCEPT"], "accepted_candidate": "baseline", "final": final.diagnostics}

    refined = _apply(
        runner, root, sample, REFINE_ALIGNMENT,
        "The bounded rescue restored finite visible support, so run the fixed residual Sim(3) executor to resolve the remaining camera-chain error.",
    )
    registered_diagnostics = dict(refined.diagnostics)
    route = ["RESCUE_GLOBAL", "REFINE_ALIGNMENT"]
    candidate = _select_final_candidate(
        refined, rescue_diagnostics=rescue_diagnostics, baseline_diagnostics=baseline,
    )

    support = refined.diagnostics.get("observed_surface_support", {})
    if candidate == "current" and support.get("local_action_recommendation") == "ADAPT_LOCAL":
        adapted = _apply(
            runner, root, sample, ADAPT_LOCAL,
            "The residual is local and lies within the fixed partial-anchored edit trust region; apply the bounded adaptation executor.",
        )
        route.append("ADAPT_LOCAL")
        candidate = _select_final_candidate(
            adapted, rescue_diagnostics=rescue_diagnostics, registered_diagnostics=registered_diagnostics,
            baseline_diagnostics=baseline,
        )

    final = _apply(
        runner, root, sample, ACCEPT,
        "Select the already-produced candidate with the strongest finite visible verifier evidence; no offline metric is available to this decision.",
        candidate=None if candidate == "current" else candidate,
    )
    route.append("ACCEPT")
    return {"sample": sample, "route": route, "accepted_candidate": candidate, "final": final.diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Existing collection with inputs/ and samples/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh isolated recovery collection.")
    parser.add_argument("--samples", nargs="+", help="Optional subset; default is every source-root/samples entry.")
    parser.add_argument("--tool-budget", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true", help="Materialize and report verifier triggers without applying tools.")
    args = parser.parse_args()
    if args.tool_budget < 3:
        parser.error("--tool-budget must be at least 3")
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    samples = args.samples or sorted(
        path.name for path in (source_root / "samples").iterdir()
        if path.is_dir() and (path / "agent_state.json").is_file()
    )
    if not samples:
        raise ValueError("no source samples found")
    runner = _load_runner()
    summary: list[dict[str, Any]] = []
    for sample in samples:
        target = _materialize_baseline(runner, source_root, output_root, sample, int(args.tool_budget))
        state = ProbeState.from_file(target / runner.STATE_FILENAME)
        trigger = needs_global_rescue_probe(state.diagnostics)
        if args.plan_only:
            summary.append({"sample": sample, "recovery_triggered": trigger, "diagnostics": state.diagnostics})
            continue
        record = _run_one(runner, target, sample)
        record["recovery_triggered"] = trigger
        summary.append(record)
        print(json.dumps({"sample": sample, "route": record["route"], "candidate": record["accepted_candidate"]}), flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "verifier_triggered_bounded_global_recovery",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "source_root": str(source_root),
        "tool_budget": int(args.tool_budget),
        "plan_only": bool(args.plan_only),
        "results": summary,
    }
    (output_root / "policy_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "samples": len(summary),
                      "recovery_triggers": sum(bool(item.get("recovery_triggered")) for item in summary)}, indent=2))


if __name__ == "__main__":
    main()
