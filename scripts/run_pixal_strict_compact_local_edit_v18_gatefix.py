#!/usr/bin/env python3
"""Run strict compact v18 with the visible-correspondence gate enforced."""

from __future__ import annotations

import scripts.run_pixal_partial_core_local_edit_v16 as v16
import scripts.run_pixal_residual_local_edit_v17 as v17
import scripts.run_pixal_strict_compact_local_edit_v18 as v18


def optimize_with_visible_gate(generated, partial, partial_normals,
                               projector, args):
    deformed, info, state = v18.optimize_strict_compact(
        generated, partial, partial_normals, projector, args)
    if not info.get("visible_gate", {}).get("accepted", False):
        info["pre_rejection_handle_improvement_ratio"] = info.get(
            "handle_improvement_ratio")
        info["handle_improvement_ratio"] = 0.0
        info["gatefix_rejection"] = "visible_correspondence_gate_not_met"
        return generated.copy(), info, None
    return deformed, info, state


def main():
    v17.OUTPUT_ROOT = v18.OUTPUT_ROOT
    v16.deform_points = v18.compact_deform_points
    v16.optimize_local_edit = optimize_with_visible_gate
    v17.main()


if __name__ == "__main__":
    main()
