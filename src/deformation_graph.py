"""Small, category-independent graph primitives for prior deformation."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def local_surface_graph(
    points: np.ndarray,
    *,
    neighbours: int,
    edge_ratio: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, int]:
    """Build affinity and metric kNN graphs without bridging surface gaps."""
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    query_k = min(int(neighbours) + 1, count)
    if query_k < 2:
        empty = sparse.csr_matrix((count, count), dtype=np.float64)
        return empty, empty, 0

    distances, indices = cKDTree(points).query(points, k=query_k, workers=-1)
    local = np.median(distances[:, 1:], axis=1)
    local = np.maximum(local, max(float(np.quantile(local, 0.01)) * 0.25, 1e-9))
    source = np.repeat(np.arange(count, dtype=np.int64), query_k - 1)
    target = indices[:, 1:].reshape(-1).astype(np.int64)
    length = distances[:, 1:].reshape(-1)
    keep = source < target
    source, target, length = source[keep], target[keep], length[keep]
    scale = np.sqrt(local[source] * local[target])
    keep = length <= float(edge_ratio) * scale
    source, target, length, scale = (
        source[keep], target[keep], length[keep], scale[keep]
    )
    affinity = np.exp(-np.square(length / np.maximum(scale, 1e-12)))
    graph = sparse.coo_matrix(
        (np.r_[affinity, affinity], (np.r_[source, target], np.r_[target, source])),
        shape=(count, count),
        dtype=np.float64,
    ).tocsr()
    metric = sparse.coo_matrix(
        (np.r_[length, length], (np.r_[source, target], np.r_[target, source])),
        shape=(count, count),
        dtype=np.float64,
    ).tocsr()
    graph.sum_duplicates()
    metric.sum_duplicates()
    return graph, metric, int(len(source))


def edge_strain(
    original: np.ndarray,
    deformed: np.ndarray,
    graph: sparse.csr_matrix,
) -> dict[str, float]:
    """Summarize deformation strain on the prior's fixed graph edges."""
    rows, cols = sparse.triu(graph, k=1).nonzero()
    if len(rows) == 0:
        return {
            "edge_stretch_p01": 1.0,
            "edge_stretch_p50": 1.0,
            "edge_stretch_p99": 1.0,
            "edge_stretch_p999": 1.0,
        }
    before = np.linalg.norm(original[rows] - original[cols], axis=1)
    after = np.linalg.norm(deformed[rows] - deformed[cols], axis=1)
    ratio = after / np.maximum(before, 1e-12)
    return {
        "edge_stretch_p01": float(np.quantile(ratio, 0.01)),
        "edge_stretch_p50": float(np.quantile(ratio, 0.50)),
        "edge_stretch_p99": float(np.quantile(ratio, 0.99)),
        "edge_stretch_p999": float(np.quantile(ratio, 0.999)),
    }
