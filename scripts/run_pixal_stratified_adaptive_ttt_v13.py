#!/usr/bin/env python3
"""Stratified fast registration: global SO(3) plus protected PCA candidates.

Every sample receives the same 8+4 shortlist: eight best global lattice
orientations and four best unperturbed PCA orientations.  Local refinement
uses the same observable-evidence early-stop gate as v12.  No sample IDs,
categories, or GT participate in routing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scripts.run_pixal_batched_so3_sim3_ttt_v11 as v11
import scripts.run_pixal_batched_adaptive_ttt_v12 as v12


ORIGINAL_BATCHED_COARSE_RANK = v11.batched_coarse_rank


def rotation_key(item):
    return tuple(np.round(np.asarray(item["rotation"]), 7).ravel())


def is_unperturbed_pca(item):
    return bool(np.max(np.abs(np.asarray(
        item["offset_degrees_xyz"], dtype=np.float64))) < 1e-8)


def stratified_coarse_rank(*args, **kwargs):
    raw = ORIGINAL_BATCHED_COARSE_RANK(*args, **kwargs)
    unique = []
    seen = set()
    for item in raw:
        key = rotation_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    global_candidates = unique[:8]
    selected_keys = {rotation_key(item) for item in global_candidates}
    protected_pca = []
    for item in unique:
        key = rotation_key(item)
        if is_unperturbed_pca(item) and key not in selected_keys:
            protected_pca.append(item)
            selected_keys.add(key)
            if len(protected_pca) == 4:
                break
    selected = global_candidates + protected_pca
    remainder = [item for item in unique if rotation_key(item) not in selected_keys]
    for index, item in enumerate(selected):
        item["shortlist_stratum"] = (
            "global_so3" if index < len(global_candidates) else "protected_pca")
    return selected + remainder


if __name__ == "__main__":
    v12.REFINE_STATE.update({"source_id": None, "confident": False})
    v11.batched_coarse_rank = stratified_coarse_rank
    v11.local.refine = v12.adaptive_local_refine
    v11.OUTPUT_ROOT = (
        Path(v11.ROOT) / "gpt_version/_pixal_stratified_adaptive_ttt_v13_20260822"
    )
    v11.main()
