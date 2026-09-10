"""Shared, inference-safe metadata helpers for the MVP completion benchmark."""

from __future__ import annotations

from typing import Iterable

import numpy as np


MVP_CATEGORIES = (
    "airplane", "cabinet", "car", "chair", "lamp", "sofa", "table", "watercraft",
    "bed", "bench", "bookshelf", "bus", "guitar", "motorbike", "pistol", "skateboard",
)


def case_key(partial_index: int) -> str:
    """Return the stable file-system key for an MVP test partial."""
    if int(partial_index) < 0:
        raise ValueError("partial index must be non-negative")
    return f"mvp_test_{int(partial_index):05d}"


def build_test_records(
    *,
    partial_count: int,
    complete_count: int,
    labels: Iterable[int],
    start: int = 0,
    stop: int | None = None,
) -> list[dict[str, int | str]]:
    """Build H5-index records without exposing complete clouds to inference."""
    if partial_count <= 0 or complete_count <= 0 or partial_count % complete_count:
        raise ValueError("MVP partial count must be a positive integral number of complete-shape views")
    labels_array = np.asarray(list(labels), dtype=np.int64)
    if len(labels_array) != int(partial_count):
        raise ValueError("MVP labels must have one entry per partial cloud")
    begin = max(0, int(start))
    end = int(partial_count) if stop is None else min(int(partial_count), int(stop))
    if end < begin:
        raise ValueError("stop must be at least start")
    views_per_complete = int(partial_count) // int(complete_count)
    records: list[dict[str, int | str]] = []
    for partial_index in range(begin, end):
        label_id = int(labels_array[partial_index])
        if not 0 <= label_id < len(MVP_CATEGORIES):
            raise ValueError(f"unknown MVP class label {label_id} at partial index {partial_index}")
        complete_index, view_index = divmod(partial_index, views_per_complete)
        records.append({
            "case_key": case_key(partial_index),
            "partial_index": partial_index,
            "complete_index": int(complete_index),
            "view_index": int(view_index),
            "views_per_complete": views_per_complete,
            "label_id": label_id,
            "category": MVP_CATEGORIES[label_id],
        })
    return records
