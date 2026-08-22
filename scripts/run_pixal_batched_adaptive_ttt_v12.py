#!/usr/bin/env python3
"""Adaptive wrapper for v11: unique rotations and evidence-based early stop.

The policy is shared across samples.  It never inspects sample IDs or GT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scripts.run_pixal_batched_so3_sim3_ttt_v11 as v11


ORIGINAL_BATCHED_COARSE_RANK = v11.batched_coarse_rank
ORIGINAL_LOCAL_REFINE = v11.local.refine
REFINE_STATE = {"source_id": None, "confident": False}


def unique_rotation_coarse_rank(*args, **kwargs):
    """Keep only the best coarse scale for each shared SO(3) orientation."""
    ranking = ORIGINAL_BATCHED_COARSE_RANK(*args, **kwargs)
    unique = []
    seen = set()
    for item in ranking:
        key = tuple(np.round(np.asarray(item["rotation"]), 7).ravel())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def confidence_gate(metrics):
    """Common observable-evidence gate for stopping local hypothesis search."""
    return bool(
        metrics["iou"] >= .85
        and metrics["coverage"] >= .90
        and metrics["leakage"] <= .08
        and metrics["normalized_visible_depth"] <= .10
        and metrics["normalized_surface_trim70"] <= .012
    )


def adaptive_local_refine(source, partial, rotation, scale, translation,
                          projector, target_mask, diagonal, render_size, levels):
    """Run full local TTT until one candidate passes the shared confidence gate."""
    source_id = id(source)
    if REFINE_STATE["source_id"] != source_id:
        REFINE_STATE.update({"source_id": source_id, "confident": False})
    if REFINE_STATE["confident"]:
        metrics = v11.depth.depth_aware_evaluate(
            source, partial, rotation, scale, translation, projector,
            target_mask, diagonal, render_size)
        trace = [{"adaptive_skip": True,
                  "reason": "earlier_candidate_passed_shared_confidence_gate"}]
        return rotation, scale, translation, metrics, trace
    result = ORIGINAL_LOCAL_REFINE(
        source, partial, rotation, scale, translation, projector, target_mask,
        diagonal, render_size, levels)
    REFINE_STATE["confident"] = confidence_gate(result[3])
    result[4].append({"adaptive_confidence_gate": REFINE_STATE["confident"]})
    return result


if __name__ == "__main__":
    v11.batched_coarse_rank = unique_rotation_coarse_rank
    v11.local.refine = adaptive_local_refine
    v11.OUTPUT_ROOT = (
        Path(v11.ROOT) / "gpt_version/_pixal_batched_adaptive_ttt_v12_20260822"
    )
    v11.main()
