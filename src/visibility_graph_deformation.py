"""Confidence-weighted visible structural correspondence deformation graph."""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def solve_visibility_deformation_graph(
    prior: np.ndarray,
    partial: np.ndarray,
    partial_ids: np.ndarray,
    prior_ids: np.ndarray,
    confidence: np.ndarray,
    *,
    nodes: int = 4096,
    graph_neighbours: int = 8,
    interpolation_neighbours: int = 4,
    smoothness: float = 2.,
    identity: float = .35,
    max_displacement_ratio: float = .05,
    seed: int = 6145,
) -> tuple[np.ndarray, dict]:
    """Fit a small graph warp from mutual visible structural correspondences.

    Node displacements are solved from a confidence-weighted correspondence
    term plus graph Laplacian and identity priors.  Thus unmatched/hidden
    regions prefer the complete prior, while connected visible structures can
    move together.
    """
    prior, partial = np.asarray(prior, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    partial_ids, prior_ids = np.asarray(partial_ids, dtype=np.int64), np.asarray(prior_ids, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if not (len(partial_ids) == len(prior_ids) == len(confidence)) or len(prior_ids) < 32:
        raise ValueError("at least 32 aligned structural correspondences are required")
    if np.any(partial_ids < 0) or np.any(partial_ids >= len(partial)) or np.any(prior_ids < 0) or np.any(prior_ids >= len(prior)):
        raise ValueError("correspondence indices are out of bounds")
    node_count = min(int(nodes), len(prior))
    rng = np.random.default_rng(int(seed))
    node_rows = np.sort(rng.choice(len(prior), node_count, replace=False))
    node_points = prior[node_rows]
    node_tree = cKDTree(node_points)
    _, match_node = node_tree.query(prior[prior_ids], k=1, workers=-1)
    target_disp = partial[partial_ids] - prior[prior_ids]
    data_weight = np.zeros(node_count, dtype=np.float64)
    data_rhs = np.zeros((node_count, 3), dtype=np.float64)
    np.add.at(data_weight, match_node, confidence)
    np.add.at(data_rhs, match_node, confidence[:, None] * target_disp)
    edge_count = min(max(int(graph_neighbours) + 1, 2), node_count)
    distance, neighbours = node_tree.query(node_points, k=edge_count, workers=-1)
    laplacian = lil_matrix((node_count, node_count), dtype=np.float64)
    for row in range(node_count):
        for dist, col in zip(distance[row, 1:], neighbours[row, 1:]):
            weight = float(np.exp(-dist * dist / max(np.median(distance[:, -1]) ** 2, 1e-12)))
            laplacian[row, row] += weight; laplacian[row, col] -= weight
    system = diags(data_weight + float(identity)) + float(smoothness) * laplacian.tocsr()
    node_disp = np.column_stack([spsolve(system, data_rhs[:, axis]) for axis in range(3)])
    max_displacement = float(max_displacement_ratio * np.linalg.norm(np.ptp(partial, axis=0)))
    length = np.linalg.norm(node_disp, axis=1)
    over = length > max_displacement
    node_disp[over] *= max_displacement / np.maximum(length[over, None], 1e-12)
    count = min(max(int(interpolation_neighbours), 1), node_count)
    distance, neighbours = node_tree.query(prior, k=count, workers=-1)
    if count == 1:
        distance, neighbours = distance[:, None], neighbours[:, None]
    weight = 1. / np.maximum(distance, 1e-6)
    weight /= weight.sum(axis=1, keepdims=True)
    displacement = (node_disp[neighbours] * weight[..., None]).sum(axis=1)
    return prior + displacement, {
        "node_count": int(node_count), "graph_neighbours": int(edge_count - 1),
        "matched_nodes": int(np.count_nonzero(data_weight)), "correspondences": int(len(prior_ids)),
        "max_node_displacement": float(np.linalg.norm(node_disp, axis=1).max(initial=0.)),
        "mean_prior_displacement": float(np.linalg.norm(displacement, axis=1).mean()),
        "max_displacement": max_displacement,
    }
