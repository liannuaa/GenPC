"""Observation-conditioned posterior adaptation of a complete Gaussian carrier.

The adapter treats a generated complete shape as a *prior*, not as a target
that every partial point must accept.  A camera-aware unbalanced transport
finds positive surface evidence and may leave incompatible mass unmatched.
The accepted evidence drives a full-surface ARAP field: already explained
surface is fixed, residual support is attracted to the partial scan, and
unobserved carrier slots move only through the prior's neighbourhood graph.

All thresholds are expressed through sampling scale, robust residual
quantiles, or the partial bounding-box diagonal.  The implementation contains
no class, part, sample, or world-axis rules and never reads ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree
import torch

from fpsample import fps_sampling

from src.bidirectional_cycle_registration import visible_score
from src.camera_conditioned_gaussian_adaptation import (
    _edge_strain,
    _local_gaussian_graph,
)
from src.zbuffer import zbuffer_depth_with_indices


@dataclass(frozen=True)
class PosteriorAdapterConfig:
    """Dataset-independent controls for the posterior solver."""

    partial_samples: int = 3_072
    prior_samples: int = 6_144
    normal_neighbours: int = 24
    transport_iterations: int = 60
    transport_relaxation: float = .72
    transport_temperature_floor: float = .012
    maximum_3d_ratio: float = .34
    maximum_screen_ratio: float = .28
    maximum_depth_ratio: float = .34
    structural_rescue_candidates: int = 48
    structural_rescue_screen_fraction: float = .55
    structural_rescue_depth_fraction: float = .70
    intrinsic_landmarks: int = 8
    intrinsic_graph_neighbours: int = 12
    minimum_transport_pairs: int = 96
    stable_spacing_multiplier: float = 3.0
    stable_residual_quantile: float = .22
    residual_quantile: float = .42
    coarse_node_fraction: float = .020
    fine_node_fraction: float = .055
    minimum_graph_nodes: int = 192
    maximum_graph_nodes: int = 6_144
    skinning_neighbours: int = 4
    low_frequency_modes: int = 24
    low_frequency_regularization: float = .035
    low_frequency_stable_weight: float = 6.0
    coarse_neighbours: int = 18
    coarse_edge_ratio: float = 2.45
    coarse_data_weight: float = 2.5
    coarse_screening: float = .0008
    coarse_iterations: int = 5
    coarse_displacement_ratio: float = .28
    fine_neighbours: int = 10
    fine_edge_ratio: float = 1.85
    fine_data_weight: float = 4.0
    fine_screening: float = .003
    fine_iterations: int = 5
    fine_displacement_ratio: float = .12
    attachment_edge_ratio: float = 3.2
    minimum_observed_improvement: float = .025
    minimum_edge_compression: float = .55
    # This is a dimensionless rest-edge ratio.  The connectivity gate below
    # remains the hard topology constraint; this robust tail bound prevents a
    # vanishing number of sampling edges from stopping a coherent update.
    maximum_edge_stretch: float = 2.80
    minimum_hidden_coverage: float = .95
    coverage_resolution: int = 160

    def validate(self) -> None:
        if min(self.partial_samples, self.prior_samples) < 32:
            raise ValueError("transport sample counts must be at least 32")
        if self.normal_neighbours < 6 or self.transport_iterations < 1:
            raise ValueError("normal and transport iteration counts are invalid")
        if self.structural_rescue_candidates < 1 or self.intrinsic_landmarks < 2:
            raise ValueError("structural transport controls are invalid")
        if self.intrinsic_graph_neighbours < 3 or self.low_frequency_modes < 0:
            raise ValueError("intrinsic graph and modal controls are invalid")
        ratios = (
            self.maximum_3d_ratio, self.maximum_screen_ratio,
            self.maximum_depth_ratio, self.stable_spacing_multiplier,
            self.coarse_edge_ratio, self.fine_edge_ratio,
            self.attachment_edge_ratio, self.coarse_displacement_ratio,
            self.fine_displacement_ratio, self.structural_rescue_screen_fraction,
            self.structural_rescue_depth_fraction, self.low_frequency_regularization,
            self.low_frequency_stable_weight,
        )
        if min(ratios) <= 0.:
            raise ValueError("metric and graph ratios must be positive")
        unit = (
            self.transport_relaxation, self.stable_residual_quantile, self.residual_quantile,
            self.minimum_observed_improvement, self.minimum_edge_compression,
            self.minimum_hidden_coverage,
        )
        if any(not 0. < value < 1. for value in unit):
            raise ValueError("quantiles, relaxation, and acceptance ratios must lie in (0, 1)")
        if self.stable_residual_quantile >= self.residual_quantile:
            raise ValueError("stable_residual_quantile must precede residual_quantile")
        if self.maximum_edge_stretch <= 1. or self.coverage_resolution < 32:
            raise ValueError("invalid topology or coverage controls")
        if not 0. < self.coarse_node_fraction < self.fine_node_fraction < 1.:
            raise ValueError("deformation graph fractions must be ordered in (0, 1)")
        if self.minimum_graph_nodes < 32 or self.maximum_graph_nodes < self.minimum_graph_nodes:
            raise ValueError("invalid deformation graph node bounds")
        if self.skinning_neighbours < 1:
            raise ValueError("skinning_neighbours must be positive")


@dataclass(frozen=True)
class TransportResult:
    partial_ids: np.ndarray
    prior_ids: np.ndarray
    weights: np.ndarray
    costs: np.ndarray
    residuals: np.ndarray
    stable_prior_mask: np.ndarray
    residual_prior_mask: np.ndarray
    unsupported_prior_mask: np.ndarray
    diagnostics: dict


def _sample_ids(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= int(count):
        return np.arange(len(points), dtype=np.int64)
    return np.asarray(fps_sampling(points, int(count), start_idx=0), dtype=np.int64)


def _local_surface_features(points: np.ndarray, sample_ids: np.ndarray, neighbours: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unoriented normals and normalized covariance eigenvalues."""
    points = np.asarray(points, dtype=np.float64)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    k = min(max(6, int(neighbours)), len(points))
    _, local_ids = cKDTree(points).query(points[sample_ids], k=k, workers=-1)
    local = points[np.asarray(local_ids)]
    centred = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred) / max(k - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.)
    descriptor = eigenvalues / np.maximum(eigenvalues.sum(axis=1, keepdims=True), 1e-12)
    normals = eigenvectors[:, :, 0]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals, descriptor


def _intrinsic_graph_signature(
    points: np.ndarray,
    sample_ids: np.ndarray,
    *,
    neighbours: int,
    landmarks: int,
) -> np.ndarray:
    """Return a scale- and landmark-order-invariant geodesic signature.

    The signature is evaluated only on the transport samples.  Sorted
    distances to farthest-point landmarks encode whether a sample lies on a
    compact sheet, a thin appendage, or near an attachment, without assuming
    a semantic part or a world axis.  Local radial quantiles complement the
    global graph coordinates when a partial scan truncates a geodesic path.
    """
    sampled = np.asarray(points, dtype=np.float64)[np.asarray(sample_ids, dtype=np.int64)]
    if len(sampled) < 3:
        return np.zeros((len(sampled), 4), dtype=np.float64)
    k = min(max(3, int(neighbours)), len(sampled))
    distance, neighbour_ids = cKDTree(sampled).query(sampled, k=k, workers=-1)
    if k == 1:
        distance = np.asarray(distance)[:, None]
        neighbour_ids = np.asarray(neighbour_ids)[:, None]
    spacing = max(float(np.median(distance[:, 1])), 1e-9)
    source = np.repeat(np.arange(len(sampled), dtype=np.int64), k - 1)
    target = np.asarray(neighbour_ids[:, 1:], dtype=np.int64).reshape(-1)
    weight = np.asarray(distance[:, 1:], dtype=np.float64).reshape(-1) / spacing
    graph = sparse.coo_matrix((weight, (source, target)), shape=(len(sampled), len(sampled))).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    landmark_ids = _sample_ids(sampled, min(int(landmarks), len(sampled)))
    geodesic = np.asarray(csgraph.dijkstra(graph, directed=False, indices=landmark_ids), dtype=np.float64)
    finite = np.isfinite(geodesic)
    for row in range(len(geodesic)):
        row_finite = finite[row]
        fill = float(np.max(geodesic[row, row_finite])) if np.any(row_finite) else 1.
        geodesic[row, ~row_finite] = 1.25 * max(fill, 1.)
    normalization = max(float(np.quantile(geodesic, .90)), 1e-9)
    ordered_geodesic = np.sort(geodesic.T / normalization, axis=1)
    radial_columns = np.unique(np.clip(
        np.asarray([1, max(1, k // 3), max(1, 2 * k // 3), k - 1]), 1, k - 1,
    ))
    radial = distance[:, radial_columns] / np.maximum(distance[:, -1:], spacing)
    return np.concatenate((ordered_geodesic, radial), axis=1)


def _median_spacing(points: np.ndarray, *, maximum_samples: int = 8_192) -> float:
    ids = _sample_ids(points, min(int(maximum_samples), len(points)))
    distance, _ = cKDTree(points).query(np.asarray(points)[ids], k=min(2, len(points)), workers=-1)
    if np.ndim(distance) == 1:
        return max(float(np.median(distance)), 1e-9)
    return max(float(np.median(distance[:, 1])), 1e-9)


def _visible_prior_ids(prior: np.ndarray, projector) -> np.ndarray:
    uv, depth = projector.project(prior)
    _, mask, indices = zbuffer_depth_with_indices(uv, depth, projector.image_shape, splat_radius=1)
    ids = np.unique(indices[mask])
    ids = ids[ids >= 0]
    return ids.astype(np.int64, copy=False)


def _unbalanced_log_transport(cost: torch.Tensor, valid: torch.Tensor, *, iterations: int,
                              relaxation: float, temperature_floor: float) -> tuple[torch.Tensor, float]:
    """Entropic unbalanced transport on a dense, subsampled cost matrix."""
    finite_min = torch.where(valid, cost, torch.inf).amin(dim=1)
    usable = torch.isfinite(finite_min)
    if not bool(usable.any()):
        return torch.zeros_like(cost), float(temperature_floor)
    temperature = max(float(torch.quantile(finite_min[usable], .50).item()) * .55, float(temperature_floor))
    log_kernel = -cost / temperature
    log_kernel = torch.where(valid, log_kernel, torch.full_like(log_kernel, -1.0e4))
    n, m = cost.shape
    log_a = torch.full((n,), -math.log(max(n, 1)), dtype=cost.dtype, device=cost.device)
    log_b = torch.full((m,), -math.log(max(m, 1)), dtype=cost.dtype, device=cost.device)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    rho = float(relaxation)
    for _ in range(int(iterations)):
        log_u = rho * (log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1))
        log_v = rho * (log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0))
    log_plan = log_kernel + log_u[:, None] + log_v[None, :]
    plan = torch.exp(torch.clamp(log_plan, min=-80., max=20.))
    plan = torch.where(valid, plan, torch.zeros_like(plan))
    return plan, float(temperature)


def partial_optimal_transport(
    prior: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    config: PosteriorAdapterConfig,
    device: str = "cuda",
) -> TransportResult:
    """Estimate capacity-aware, positive-only partial-to-prior evidence."""
    config.validate()
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1:] != (3,) or partial.shape[1:] != (3,):
        raise ValueError("prior and partial must be shaped (N, 3)")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
    spacing = _median_spacing(partial)
    visible_ids = _visible_prior_ids(prior, projector)
    if len(visible_ids) < int(config.minimum_transport_pairs):
        raise ValueError("too few camera-visible prior surfels for partial transport")
    partial_sample_ids = _sample_ids(partial, int(config.partial_samples))
    visible_sample_local = _sample_ids(prior[visible_ids], int(config.prior_samples))
    prior_sample_ids = visible_ids[visible_sample_local]

    partial_normal, partial_shape = _local_surface_features(
        partial, partial_sample_ids, int(config.normal_neighbours),
    )
    prior_normal, prior_shape = _local_surface_features(
        prior, prior_sample_ids, int(config.normal_neighbours),
    )
    partial_uv_all, partial_depth_all = projector.project(partial)
    prior_uv_all, prior_depth_all = projector.project(prior)
    partial_uv, prior_uv = partial_uv_all[partial_sample_ids], prior_uv_all[prior_sample_ids]
    partial_depth, prior_depth = partial_depth_all[partial_sample_ids], prior_depth_all[prior_sample_ids]
    uv_low, uv_high = np.quantile(partial_uv, [.01, .99], axis=0)
    screen_extent = max(float(np.max(uv_high - uv_low)), 1.)
    depth_low, depth_high = np.quantile(partial_depth, [.01, .99])
    depth_extent = max(float(depth_high - depth_low), spacing)

    actual_device = str(device)
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        actual_device = "cpu"
    dtype = torch.float32
    pp = torch.as_tensor(partial[partial_sample_ids] / diagonal, dtype=dtype, device=actual_device)
    qp = torch.as_tensor(prior[prior_sample_ids] / diagonal, dtype=dtype, device=actual_device)
    pu = torch.as_tensor(partial_uv / screen_extent, dtype=dtype, device=actual_device)
    qu = torch.as_tensor(prior_uv / screen_extent, dtype=dtype, device=actual_device)
    pz = torch.as_tensor(partial_depth / depth_extent, dtype=dtype, device=actual_device)
    qz = torch.as_tensor(prior_depth / depth_extent, dtype=dtype, device=actual_device)
    pn = torch.as_tensor(partial_normal, dtype=dtype, device=actual_device)
    qn = torch.as_tensor(prior_normal, dtype=dtype, device=actual_device)
    ps = torch.as_tensor(partial_shape, dtype=dtype, device=actual_device)
    qs = torch.as_tensor(prior_shape, dtype=dtype, device=actual_device)
    with torch.no_grad():
        distance_3d = torch.cdist(pp, qp)
        distance_uv = torch.cdist(pu, qu)
        distance_depth = torch.abs(pz[:, None] - qz[None, :])
        normal_cost = 1. - torch.abs(pn @ qn.T)
        shape_cost = torch.cdist(ps, qs)
        cost = (
            .46 * distance_3d.square()
            + .23 * distance_uv.square()
            + .18 * distance_depth.square()
            + .08 * normal_cost
            + .05 * shape_cost
        )
        valid = (
            (distance_3d <= float(config.maximum_3d_ratio))
            & (distance_uv <= float(config.maximum_screen_ratio))
            & (distance_depth <= float(config.maximum_depth_ratio))
            & (normal_cost <= .92)
        )
        plan, temperature = _unbalanced_log_transport(
            cost, valid, iterations=int(config.transport_iterations),
            relaxation=float(config.transport_relaxation),
            temperature_floor=float(config.transport_temperature_floor),
        )
        row_mass = plan.sum(dim=1)
        top_k = min(8, plan.shape[1])
        top_weight, top_prior_local = torch.topk(plan, k=top_k, dim=1, largest=True, sorted=True)
        top_cost = cost.gather(1, top_prior_local)
        top_valid = valid.gather(1, top_prior_local)
        top_concentration = top_weight / torch.clamp(row_mass[:, None], min=1e-12)
        arrays = [value.detach().cpu().numpy() for value in (
            row_mass, top_weight, top_prior_local, top_cost, top_valid, top_concentration,
        )]
    row_mass_np, top_weight_np, top_local_np, top_cost_np, top_valid_np, top_concentration_np = arrays
    valid_rows = top_valid_np[:, 0].astype(bool) & np.isfinite(top_cost_np[:, 0]) & (row_mass_np > 0.)
    if int(valid_rows.sum()) < int(config.minimum_transport_pairs):
        empty = np.zeros(len(prior), dtype=bool)
        return TransportResult(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0), np.empty(0), np.empty(0), empty.copy(), empty.copy(), ~empty,
            {"active": False, "reason": "insufficient_valid_transport_rows", "valid_rows": int(valid_rows.sum())},
        )
    best_local = top_local_np[:, 0].astype(np.int64)
    best_prior = prior_sample_ids[best_local]
    row_residual = np.linalg.norm(partial[partial_sample_ids] - prior[best_prior], axis=1)
    stable_limit = max(
        float(config.stable_spacing_multiplier) * spacing,
        float(np.quantile(row_residual[valid_rows], float(config.stable_residual_quantile))),
    )
    stable_limit = min(stable_limit, .035 * diagonal)
    residual_limit = max(
        stable_limit * 1.35,
        float(np.quantile(row_residual[valid_rows], float(config.residual_quantile))),
    )

    # High residual does not mean invalid: it is precisely the positive
    # evidence needed when a complete prior is locally too short.  Reject
    # only isolated/incoherent residuals, then allocate their top-k transport
    # choices with one carrier slot per observed sample.  Stable body points
    # cannot consume all target capacity before an extremity is considered.
    residual_rows = np.flatnonzero(valid_rows & (row_residual >= residual_limit))
    coherent_rows: list[int] = []
    if len(residual_rows):
        residual_points = partial[partial_sample_ids[residual_rows]]
        residual_spacing = _median_spacing(residual_points)
        component_radius = max(3.0 * residual_spacing, .012 * diagonal)
        pairs = cKDTree(residual_points).query_pairs(component_radius, output_type="ndarray")
        if len(pairs):
            component_graph = sparse.coo_matrix(
                (np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                shape=(len(residual_rows), len(residual_rows)),
            ).tocsr()
            component_count, labels = csgraph.connected_components(component_graph, directed=False)
        else:
            component_count, labels = len(residual_rows), np.arange(len(residual_rows), dtype=np.int64)
        minimum_component = max(8, int(math.ceil(.003 * len(partial_sample_ids))))
        for component in range(component_count):
            member = np.flatnonzero(labels == component)
            if len(member) < minimum_component:
                continue
            rows_local = residual_rows[member]
            vectors = partial[partial_sample_ids[rows_local]] - prior[best_prior[rows_local]]
            median_vector = np.median(vectors, axis=0)
            magnitude = float(np.linalg.norm(median_vector))
            if magnitude <= 1e-12:
                continue
            coherence = float(np.mean((vectors @ median_vector) / np.maximum(
                np.linalg.norm(vectors, axis=1) * magnitude, 1e-12,
            )))
            if coherence >= .20:
                coherent_rows.extend(rows_local.tolist())
    coherent_rows_np = np.asarray(coherent_rows, dtype=np.int64)
    coherent_fraction = float(len(coherent_rows_np) / max(int(valid_rows.sum()), 1))
    coherent_prior_fraction = float(
        len(np.unique(best_prior[coherent_rows_np])) / max(len(prior_sample_ids), 1)
    )
    # A posterior update is identifiable only when reliable residual support
    # is local relative to the already explained and uncertain quantile bands.
    # Broad coherent residuals indicate a global observation/prior mismatch;
    # allowing a local deformation graph to absorb them would overfit one view.
    localization_bound = float(config.residual_quantile - config.stable_residual_quantile)
    localized_residual_evidence = bool(
        len(coherent_rows_np) >= int(config.minimum_transport_pairs)
        and coherent_fraction <= localization_bound
    )
    if not localized_residual_evidence:
        coherent_rows_np = np.empty(0, dtype=np.int64)
    edges: list[tuple[float, float, float, int, int, int]] = []
    for row in coherent_rows_np:
        for rank in range(top_local_np.shape[1]):
            if not bool(top_valid_np[row, rank]):
                continue
            local_prior = int(top_local_np[row, rank])
            edges.append((
                -float(row_residual[row]),
                float(top_cost_np[row, rank]),
                -float(top_concentration_np[row, rank]),
                int(row), local_prior, int(rank),
            ))
    edges.sort()
    used_rows: set[int] = set()
    used_prior: set[int] = set()
    assignments: list[tuple[int, int, int]] = []
    for _, _, _, row, local_prior, rank in edges:
        prior_id = int(prior_sample_ids[local_prior])
        if row in used_rows or prior_id in used_prior:
            continue
        used_rows.add(row); used_prior.add(prior_id)
        assignments.append((row, local_prior, rank))
    if assignments:
        selected_rows = np.asarray([row[0] for row in assignments], dtype=np.int64)
        selected_local = np.asarray([row[1] for row in assignments], dtype=np.int64)
        selected_rank = np.asarray([row[2] for row in assignments], dtype=np.int64)
        selected_partial = partial_sample_ids[selected_rows]
        selected_prior = prior_sample_ids[selected_local]
        selected_residual = np.linalg.norm(partial[selected_partial] - prior[selected_prior], axis=1)
        selected_weight = top_concentration_np[selected_rows, selected_rank]
        selected_cost = top_cost_np[selected_rows, selected_rank]
    else:
        selected_rows = np.empty(0, dtype=np.int64)
        selected_partial = np.empty(0, dtype=np.int64)
        selected_prior = np.empty(0, dtype=np.int64)
        selected_residual = np.empty(0, dtype=np.float64)
        selected_weight = np.empty(0, dtype=np.float64)
        selected_cost = np.empty(0, dtype=np.float64)

    stable_rows = valid_rows & (row_residual <= stable_limit)
    stable_mask = np.zeros(len(prior), dtype=bool)
    residual_mask = np.zeros(len(prior), dtype=bool)
    stable_mask[best_prior[stable_rows]] = True
    residual_mask[selected_prior] = True
    # Expand stable evidence only within local carrier sampling scale.  This
    # locks a measured surface patch, not a hand-authored semantic part.
    if stable_mask.any():
        nearest_stable, _ = cKDTree(prior[stable_mask]).query(prior, k=1, workers=-1)
        stable_mask |= nearest_stable <= max(2.0 * _median_spacing(prior), spacing)
    stable_mask &= ~residual_mask
    unsupported = ~(stable_mask | residual_mask)
    return TransportResult(
        partial_ids=selected_partial,
        prior_ids=selected_prior,
        weights=selected_weight,
        costs=selected_cost,
        residuals=selected_residual,
        stable_prior_mask=stable_mask,
        residual_prior_mask=residual_mask,
        unsupported_prior_mask=unsupported,
        diagnostics={
            "active": bool(residual_mask.any()),
            "method": "camera_structure_unbalanced_partial_transport",
            "device": actual_device,
            "partial_diagonal": diagonal,
            "partial_spacing": spacing,
            "visible_prior": int(len(visible_ids)),
            "partial_samples": int(len(partial_sample_ids)),
            "prior_samples": int(len(prior_sample_ids)),
            "valid_transport_rows": int(valid_rows.sum()),
            "retained_capacity_pairs": int(len(selected_rows)),
            "stable_pairs": int(stable_rows.sum()),
            "coherent_residual_rows": int(len(coherent_rows_np)),
            "raw_coherent_residual_rows": int(len(coherent_rows)),
            "coherent_residual_fraction": coherent_fraction,
            "coherent_prior_support_fraction": coherent_prior_fraction,
            "localization_bound": localization_bound,
            "localized_residual_evidence": localized_residual_evidence,
            "residual_pairs": int(len(selected_rows)),
            "stable_prior": int(stable_mask.sum()),
            "residual_prior": int(residual_mask.sum()),
            "unsupported_prior": int(unsupported.sum()),
            "unmatched_partial_fraction": float(1. - (stable_rows.sum() + len(selected_rows)) / max(len(partial_sample_ids), 1)),
            "temperature": temperature,
            "stable_limit": stable_limit,
            "residual_limit": residual_limit,
            "residual_median": float(np.median(selected_residual)) if len(selected_residual) else None,
            "residual_p90": float(np.quantile(selected_residual, .90)) if len(selected_residual) else None,
        },
    )


def _augment_attachment_edges(graph: sparse.csr_matrix, metric: sparse.csr_matrix,
                              points: np.ndarray, *, edge_ratio: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix, int]:
    """Join only near-contact graph components in the original complete prior."""
    component_count, labels = csgraph.connected_components(graph, directed=False)
    if component_count <= 1:
        return graph, metric, 0
    points = np.asarray(points, dtype=np.float64)
    query_k = min(32, len(points))
    distances, neighbours = cKDTree(points).query(points, k=query_k, workers=-1)
    local_scale = np.median(distances[:, 1:min(query_k, 9)], axis=1)
    candidates: list[tuple[float, int, int]] = []
    for column in range(1, query_k):
        other = neighbours[:, column]
        cross = labels != labels[other]
        allowed = distances[:, column] <= float(edge_ratio) * np.sqrt(
            np.maximum(local_scale * local_scale[other], 1e-18)
        )
        for source in np.flatnonzero(cross & allowed):
            target = int(other[source])
            if source < target:
                candidates.append((float(distances[source, column]), int(source), target))
    candidates.sort()
    parent = np.arange(component_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    accepted: list[tuple[float, int, int]] = []
    for distance, source, target in candidates:
        left, right = find(int(labels[source])), find(int(labels[target]))
        if left == right:
            continue
        parent[right] = left
        accepted.append((distance, source, target))
    if not accepted:
        return graph, metric, 0
    source = np.asarray([row[1] for row in accepted], dtype=np.int64)
    target = np.asarray([row[2] for row in accepted], dtype=np.int64)
    distance = np.asarray([row[0] for row in accepted], dtype=np.float64)
    scale = np.sqrt(np.maximum(local_scale[source] * local_scale[target], 1e-18))
    weight = .25 * np.exp(-np.square(distance / np.maximum(scale, 1e-12)))
    attachment_graph = sparse.coo_matrix(
        (np.r_[weight, weight], (np.r_[source, target], np.r_[target, source])), shape=graph.shape,
    ).tocsr()
    attachment_metric = sparse.coo_matrix(
        (np.r_[distance, distance], (np.r_[source, target], np.r_[target, source])), shape=metric.shape,
    ).tocsr()
    return (graph + attachment_graph).tocsr(), (metric + attachment_metric).tocsr(), int(len(accepted))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, .5 * cumulative[-1], side="left")]])


def _coherent_targets(source: np.ndarray, target: np.ndarray, *, diagonal: float) -> tuple[np.ndarray, dict]:
    """Replace pointwise residuals by robust local component motion.

    Partial transport establishes *where* positive support exists.  A coarse
    deformation should not turn each transported sample into an independent
    pin, because that creates spikes and tears.  Connected observations with
    compatible displacement instead vote for one robust motion, irrespective
    of class or semantic part.
    """
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 8:
        return target.copy(), {"components": 0, "regularized_targets": 0}
    spacing = _median_spacing(target)
    radius = max(4.0 * spacing, .015 * float(diagonal))
    edges = cKDTree(target).query_pairs(radius, output_type="ndarray")
    if len(edges):
        graph = sparse.coo_matrix(
            (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
            shape=(len(target), len(target)),
        ).tocsr()
        count, labels = csgraph.connected_components(graph, directed=False)
    else:
        count, labels = len(target), np.arange(len(target), dtype=np.int64)
    residual = target - source
    result = target.copy()
    records: list[dict] = []
    minimum = max(8, int(math.ceil(.01 * len(target))))
    regularized = 0
    for component in range(count):
        member = np.flatnonzero(labels == component)
        if len(member) < minimum:
            continue
        median = np.median(residual[member], axis=0)
        magnitude = float(np.linalg.norm(median))
        if magnitude <= 1e-12:
            continue
        coherence = float(np.mean((residual[member] @ median) / np.maximum(
            np.linalg.norm(residual[member], axis=1) * magnitude, 1e-12,
        )))
        if coherence < .20:
            continue
        result[member] = source[member] + median
        regularized += len(member)
        records.append({
            "points": int(len(member)), "median_residual": median.tolist(),
            "median_residual_norm": magnitude, "directional_coherence": coherence,
        })
    return result, {
        "component_radius": radius, "minimum_component_points": minimum,
        "components": int(len(records)), "regularized_targets": int(regularized),
        "component_records": records,
    }


def _solve_preconditioned(matrix: sparse.csr_matrix, rhs: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Jacobi-preconditioned CG for the full 100k-carrier system."""
    inverse_diagonal = 1. / np.maximum(np.asarray(matrix.diagonal()).reshape(-1), 1e-12)
    preconditioner = sparse.diags(inverse_diagonal, format="csr")
    result = np.zeros_like(rhs, dtype=np.float64)
    status: list[int] = []
    for axis in range(rhs.shape[1]):
        try:
            value, code = sparse_linalg.cg(
                matrix, rhs[:, axis], M=preconditioner, rtol=2e-6, atol=0., maxiter=900,
            )
        except TypeError:
            value, code = sparse_linalg.cg(
                matrix, rhs[:, axis], M=preconditioner, tol=2e-6, maxiter=900,
            )
        result[:, axis] = value
        status.append(int(code))
    return result, status


def _orthographic_coverage(reference: np.ndarray, candidate: np.ndarray, *, resolution: int) -> tuple[float, list[float]]:
    """Measure fixed-frame support retention in six signed PCA views."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    centre = reference.mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(reference - centre, full_matrices=False)
    ref_coordinates = (reference - centre) @ basis.T
    candidate_coordinates = (candidate - centre) @ basis.T
    combined = np.concatenate((ref_coordinates, candidate_coordinates), axis=0)
    ratios: list[float] = []
    for depth_axis in range(3):
        planar = [axis for axis in range(3) if axis != depth_axis]
        lower, upper = np.quantile(combined[:, planar], [.005, .995], axis=0)
        extent = np.maximum(upper - lower, 1e-9)
        occupied: list[int] = []
        for coordinates in (ref_coordinates, candidate_coordinates):
            uv = np.floor((coordinates[:, planar] - lower) / extent * (int(resolution) - 1)).astype(np.int64)
            valid = ((uv >= 0) & (uv < int(resolution))).all(axis=1)
            key = uv[valid, 0] * int(resolution) + uv[valid, 1]
            occupied.append(int(len(np.unique(key))))
        ratios.append(float(occupied[1] / max(occupied[0], 1)))
    return float(min(ratios, default=1.)), ratios


def _soft_full_surface_arap(
    prior: np.ndarray,
    target_ids: np.ndarray,
    targets: np.ndarray,
    target_weights: np.ndarray,
    stable_mask: np.ndarray,
    *,
    neighbours: int,
    edge_ratio: float,
    attachment_edge_ratio: float,
    data_weight: float,
    screening: float,
    iterations: int,
    maximum_displacement: float,
    minimum_improvement: float,
    minimum_edge_compression: float,
    maximum_edge_stretch: float,
    minimum_hidden_coverage: float,
    coverage_resolution: int,
    coherent_component_targets: bool,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Fit soft observations while ARAP propagates motion over complete support."""
    prior = np.asarray(prior, dtype=np.float64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    stable_mask = np.asarray(stable_mask, dtype=bool)
    graph, metric, original_edges = _local_gaussian_graph(prior, neighbours=int(neighbours), edge_ratio=float(edge_ratio))
    graph, metric, attachment_edges = _augment_attachment_edges(
        graph, metric, prior, edge_ratio=float(attachment_edge_ratio),
    )
    if graph.nnz == 0 or len(target_ids) == 0:
        return prior.copy(), {"active": False, "reason": "empty_graph_or_targets"}, np.zeros(len(prior), dtype=bool)
    target_ids_unique, inverse = np.unique(target_ids, return_inverse=True)
    target_sum = np.zeros((len(target_ids_unique), 3), dtype=np.float64)
    weight_sum = np.zeros(len(target_ids_unique), dtype=np.float64)
    np.add.at(target_sum, inverse, targets * target_weights[:, None])
    np.add.at(weight_sum, inverse, target_weights)
    targets_unique = target_sum / np.maximum(weight_sum[:, None], 1e-12)
    weights_unique = weight_sum / max(float(np.median(weight_sum[weight_sum > 0.])), 1e-12)
    keep = ~stable_mask[target_ids_unique]
    target_ids_unique, targets_unique, weights_unique = (
        target_ids_unique[keep], targets_unique[keep], weights_unique[keep]
    )
    if len(target_ids_unique) == 0:
        return prior.copy(), {"active": False, "reason": "all_transport_targets_are_stable"}, np.zeros(len(prior), dtype=bool)
    coherence_info = {"components": 0, "regularized_targets": 0}
    if bool(coherent_component_targets):
        diagonal = max(float(np.linalg.norm(np.ptp(np.concatenate((prior, targets_unique), axis=0), axis=0))), 1e-9)
        targets_unique, coherence_info = _coherent_targets(
            prior[target_ids_unique], targets_unique, diagonal=diagonal,
        )
    initial_residual = np.linalg.norm(prior[target_ids_unique] - targets_unique, axis=1)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - graph
    fixed_ids = np.flatnonzero(stable_mask)
    unknown = np.flatnonzero(~stable_mask)
    lookup = np.full(len(prior), -1, dtype=np.int64)
    lookup[unknown] = np.arange(len(unknown), dtype=np.int64)
    local_targets = lookup[target_ids_unique]
    data_diagonal = np.zeros(len(unknown), dtype=np.float64)
    effective_data_weight = float(data_weight) * max(float(np.median(degree[degree > 0.])), 1e-6)
    np.add.at(data_diagonal, local_targets, effective_data_weight * weights_unique)
    system = laplacian[unknown][:, unknown].tocsr() + sparse.diags(
        np.full(len(unknown), float(screening)) + data_diagonal, format="csr",
    )
    rows, cols = sparse.triu(graph, k=1).nonzero()
    edge_weight = np.asarray(graph[rows, cols]).reshape(-1)
    rest_edge = prior[rows] - prior[cols]
    deformed = prior.copy()
    rotations = np.broadcast_to(np.eye(3), (len(prior), 3, 3)).copy()
    cg_status: list[int] = [0, 0, 0]
    for _ in range(int(iterations)):
        deformed_edge = deformed[rows] - deformed[cols]
        covariance = np.zeros((len(prior), 3, 3), dtype=np.float64)
        contribution = edge_weight[:, None, None] * np.einsum("ni,nj->nij", deformed_edge, rest_edge)
        np.add.at(covariance, rows, contribution)
        np.add.at(covariance, cols, contribution)
        active = np.flatnonzero(~stable_mask)
        left, _, right = np.linalg.svd(covariance[active], full_matrices=False)
        local_rotation = left @ right
        reflected = np.linalg.det(local_rotation) < 0.
        if np.any(reflected):
            left[reflected, :, -1] *= -1.
            local_rotation = left @ right
        rotations[active] = local_rotation
        desired = np.einsum("nij,nj->ni", .5 * (rotations[rows] + rotations[cols]), rest_edge)
        rhs = np.zeros((len(prior), 3), dtype=np.float64)
        np.add.at(rhs, rows, edge_weight[:, None] * desired)
        np.add.at(rhs, cols, -edge_weight[:, None] * desired)
        rhs_unknown = rhs[unknown] + float(screening) * prior[unknown]
        np.add.at(
            rhs_unknown, local_targets,
            (effective_data_weight * weights_unique)[:, None] * targets_unique,
        )
        if len(fixed_ids):
            rhs_unknown -= laplacian[unknown][:, fixed_ids] @ prior[fixed_ids]
        solution, cg_status = _solve_preconditioned(system, rhs_unknown)
        displacement = solution - prior[unknown]
        length = np.linalg.norm(displacement, axis=1, keepdims=True)
        displacement *= np.minimum(1., float(maximum_displacement) / np.maximum(length, 1e-12))
        deformed[unknown] = prior[unknown] + displacement
        deformed[fixed_ids] = prior[fixed_ids]

    accepted = None
    best_improvement = -np.inf
    last: dict | None = None
    trace: list[dict] = []
    for alpha in (1., .85, .70, .55, .40, .25, .15):
        candidate = prior + float(alpha) * (deformed - prior)
        candidate[stable_mask] = prior[stable_mask]
        after = np.linalg.norm(candidate[target_ids_unique] - targets_unique, axis=1)
        before_median = _weighted_median(initial_residual, weights_unique)
        after_median = _weighted_median(after, weights_unique)
        improvement = 1. - after_median / max(before_median, 1e-12)
        strain = _edge_strain(prior, candidate, graph)
        coverage, per_view = _orthographic_coverage(prior, candidate, resolution=int(coverage_resolution))
        effective_graph = graph.copy().tolil()
        edge_ratio_now = np.linalg.norm(candidate[rows] - candidate[cols], axis=1) / np.maximum(
            np.linalg.norm(prior[rows] - prior[cols], axis=1), 1e-12,
        )
        broken = edge_ratio_now > float(maximum_edge_stretch)
        if np.any(broken):
            effective_graph[rows[broken], cols[broken]] = 0.; effective_graph[cols[broken], rows[broken]] = 0.
        candidate_components, candidate_labels = csgraph.connected_components(effective_graph.tocsr(), directed=False)
        original_components, original_labels = csgraph.connected_components(graph, directed=False)
        significant_size = max(32, int(math.ceil(.001 * len(prior))))
        candidate_sizes = np.bincount(candidate_labels, minlength=candidate_components)
        original_sizes = np.bincount(original_labels, minlength=original_components)
        candidate_significant = int(np.sum(candidate_sizes >= significant_size))
        original_significant = int(np.sum(original_sizes >= significant_size))
        candidate_small_mass = int(candidate_sizes[candidate_sizes < significant_size].sum())
        last = {
            "alpha": float(alpha), "observed_residual_before": before_median,
            "observed_residual_after": after_median, "observed_improvement": float(improvement),
            "hidden_coverage_ratio": coverage, "hidden_coverage_per_view": per_view,
            "original_graph_components": int(original_components),
            "candidate_graph_components": int(candidate_components), **strain,
            "significant_component_size": int(significant_size),
            "original_significant_components": original_significant,
            "candidate_significant_components": candidate_significant,
            "candidate_small_component_mass": candidate_small_mass,
        }
        trace.append(last)
        best_improvement = max(best_improvement, float(improvement))
        if (
            improvement >= float(minimum_improvement)
            and strain["edge_stretch_p01"] >= float(minimum_edge_compression)
            and strain["edge_stretch_p999"] <= float(maximum_edge_stretch)
            and coverage >= float(minimum_hidden_coverage)
            and candidate_significant <= original_significant
        ):
            accepted = candidate, last
            break
    if accepted is None:
        return prior.copy(), {
            "active": False, "reason": "no_pareto_valid_structural_update",
            "best_observed_improvement": float(best_improvement), "last_trial": last,
            "surface_graph_edges": int(original_edges), "attachment_edges": int(attachment_edges),
            "coherent_targets": coherence_info,
            "line_search": trace,
        }, np.zeros(len(prior), dtype=bool)
    candidate, accepted_info = accepted
    moved = np.linalg.norm(candidate - prior, axis=1) > 1e-8
    return candidate, {
        "active": True, "field": "soft_full_surface_arap_posterior",
        "surface_graph_edges": int(original_edges), "attachment_edges": int(attachment_edges),
        "coherent_targets": coherence_info,
        "line_search": trace,
        "target_gaussians": int(len(target_ids_unique)), "stable_gaussians": int(stable_mask.sum()),
        "moved_gaussians": int(moved.sum()), "moved_fraction": float(moved.mean()),
        "maximum_displacement": float(np.linalg.norm(candidate - prior, axis=1).max(initial=0.)),
        "mean_displacement": float(np.linalg.norm(candidate - prior, axis=1).mean()),
        "cg_status": cg_status, **accepted_info,
    }, moved


def _embedded_arap_posterior(
    prior: np.ndarray,
    target_ids: np.ndarray,
    targets: np.ndarray,
    target_weights: np.ndarray,
    stable_mask: np.ndarray,
    *,
    node_fraction: float,
    minimum_nodes: int,
    maximum_nodes: int,
    skinning_neighbours: int,
    neighbours: int,
    edge_ratio: float,
    attachment_edge_ratio: float,
    data_weight: float,
    screening: float,
    iterations: int,
    maximum_displacement: float,
    minimum_improvement: float,
    minimum_edge_compression: float,
    maximum_edge_stretch: float,
    minimum_hidden_coverage: float,
    coverage_resolution: int,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Deform the full carrier with a sparse, rotation-aware ARAP graph.

    Every transported carrier is retained as a graph node.  Additional nodes
    are sampled uniformly from the complete prior and drive all 100k surfels
    through smooth embedded-deformation skinning.  Consequently an observed
    residual moves its attached structure continuously, while stable evidence
    provides fixed boundary conditions and unsupported regions retain their
    relative geometry.
    """
    prior = np.asarray(prior, dtype=np.float64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    stable_mask = np.asarray(stable_mask, dtype=bool)
    if len(target_ids) == 0:
        return prior.copy(), {"active": False, "reason": "empty_transport_targets"}, np.zeros(len(prior), dtype=bool)

    unique_target_ids, inverse = np.unique(target_ids, return_inverse=True)
    target_sum = np.zeros((len(unique_target_ids), 3), dtype=np.float64)
    weight_sum = np.zeros(len(unique_target_ids), dtype=np.float64)
    np.add.at(target_sum, inverse, targets * target_weights[:, None])
    np.add.at(weight_sum, inverse, target_weights)
    unique_targets = target_sum / np.maximum(weight_sum[:, None], 1e-12)
    unique_weights = weight_sum / max(float(np.median(weight_sum[weight_sum > 0.])), 1e-12)
    keep = ~stable_mask[unique_target_ids]
    unique_target_ids = unique_target_ids[keep]
    unique_targets = unique_targets[keep]
    unique_weights = unique_weights[keep]
    if len(unique_target_ids) == 0:
        return prior.copy(), {"active": False, "reason": "all_transport_targets_are_stable"}, np.zeros(len(prior), dtype=bool)

    diagonal = max(float(np.linalg.norm(np.ptp(np.concatenate((prior, unique_targets), axis=0), axis=0))), 1e-9)
    unique_targets, coherence_info = _coherent_targets(
        prior[unique_target_ids], unique_targets, diagonal=diagonal,
    )

    requested_nodes = int(np.clip(
        math.ceil(float(node_fraction) * len(prior)), int(minimum_nodes), int(maximum_nodes),
    ))
    sampled_node_ids = _sample_ids(prior, requested_nodes)
    node_ids = np.unique(np.r_[sampled_node_ids, unique_target_ids]).astype(np.int64)
    nodes = prior[node_ids]
    node_lookup = np.full(len(prior), -1, dtype=np.int64)
    node_lookup[node_ids] = np.arange(len(node_ids), dtype=np.int64)
    target_nodes = node_lookup[unique_target_ids]
    stable_nodes = stable_mask[node_ids].copy()
    if np.any(stable_mask):
        # Stable observations constrain their nearest graph controls.  They
        # are high-rigidity boundary conditions rather than pointwise clamps,
        # which avoids cracks between individually frozen samples and a
        # smoothly moving unsupported surface.
        nearest_stable_nodes = cKDTree(nodes).query(prior[stable_mask], k=1, workers=-1)[1]
        stable_nodes[np.asarray(nearest_stable_nodes, dtype=np.int64)] = True
    stable_nodes[target_nodes] = False

    node_targets = nodes[target_nodes] + (unique_targets - prior[unique_target_ids])
    deformed_nodes, node_info, _ = _soft_full_surface_arap(
        nodes, target_nodes, node_targets, unique_weights, stable_nodes,
        neighbours=int(neighbours), edge_ratio=float(edge_ratio),
        attachment_edge_ratio=float(attachment_edge_ratio),
        data_weight=float(data_weight), screening=float(screening), iterations=int(iterations),
        maximum_displacement=float(maximum_displacement),
        minimum_improvement=max(.5 * float(minimum_improvement), 1e-4),
        minimum_edge_compression=max(.5 * float(minimum_edge_compression), .20),
        maximum_edge_stretch=1.5 * float(maximum_edge_stretch),
        minimum_hidden_coverage=min(float(minimum_hidden_coverage), .85),
        coverage_resolution=max(48, int(coverage_resolution) // 2),
        coherent_component_targets=False,
    )
    if not node_info.get("active"):
        return prior.copy(), {
            "active": False, "reason": "deformation_graph_rejected",
            "graph_nodes": int(len(nodes)), "node_solver": node_info,
            "coherent_targets": coherence_info,
        }, np.zeros(len(prior), dtype=bool)

    node_graph, _, node_edges = _local_gaussian_graph(
        nodes, neighbours=int(neighbours), edge_ratio=float(edge_ratio),
    )
    node_graph, _, attachment_edges = _augment_attachment_edges(
        node_graph, node_graph.copy(), nodes, edge_ratio=float(attachment_edge_ratio),
    )
    rows, cols = sparse.triu(node_graph, k=1).nonzero()
    rotations = np.broadcast_to(np.eye(3), (len(nodes), 3, 3)).copy()
    if len(rows):
        weights = np.asarray(node_graph[rows, cols]).reshape(-1)
        rest = nodes[rows] - nodes[cols]
        changed = deformed_nodes[rows] - deformed_nodes[cols]
        covariance = np.zeros((len(nodes), 3, 3), dtype=np.float64)
        contribution = weights[:, None, None] * np.einsum("ni,nj->nij", changed, rest)
        np.add.at(covariance, rows, contribution)
        np.add.at(covariance, cols, contribution)
        active_nodes = np.flatnonzero(np.asarray(node_graph.sum(axis=1)).reshape(-1) > 0.)
        left, _, right = np.linalg.svd(covariance[active_nodes], full_matrices=False)
        local_rotation = left @ right
        reflected = np.linalg.det(local_rotation) < 0.
        if np.any(reflected):
            left[reflected, :, -1] *= -1.
            local_rotation = left @ right
        rotations[active_nodes] = local_rotation

    skin_k = min(max(1, int(skinning_neighbours)), len(nodes))
    skin_distance, skin_ids = cKDTree(nodes).query(prior, k=skin_k, workers=-1)
    if skin_k == 1:
        skin_distance = np.asarray(skin_distance)[:, None]
        skin_ids = np.asarray(skin_ids)[:, None]
    node_spacing = _median_spacing(nodes)
    radius = np.maximum(np.asarray(skin_distance)[:, -1], .5 * node_spacing)
    skin_weight = np.exp(-.5 * np.square(skin_distance / np.maximum(radius[:, None], 1e-12)))
    skin_weight /= np.maximum(skin_weight.sum(axis=1, keepdims=True), 1e-12)
    warped = np.zeros_like(prior)
    for column in range(skin_k):
        local = skin_ids[:, column]
        offset = prior - nodes[local]
        transformed = np.einsum("nij,nj->ni", rotations[local], offset) + deformed_nodes[local]
        warped += skin_weight[:, column, None] * transformed
    if np.any(stable_mask):
        stable_distance = cKDTree(prior[stable_mask]).query(prior, k=1, workers=-1)[0]
        transition = max(2.0 * _median_spacing(prior), _median_spacing(prior[stable_mask]))
        release = 1. - np.exp(-.5 * np.square(stable_distance / max(transition, 1e-12)))
        warped = prior + release[:, None] * (warped - prior)
    carrier_graph, _, carrier_edges = _local_gaussian_graph(
        prior, neighbours=int(neighbours), edge_ratio=float(edge_ratio),
    )
    carrier_graph, _, carrier_attachment_edges = _augment_attachment_edges(
        carrier_graph, carrier_graph.copy(), prior, edge_ratio=float(attachment_edge_ratio),
    )
    carrier_rows, carrier_cols = sparse.triu(carrier_graph, k=1).nonzero()
    initial_residual = np.linalg.norm(prior[unique_target_ids] - unique_targets, axis=1)
    original_components, original_labels = csgraph.connected_components(carrier_graph, directed=False)
    significant_size = max(32, int(math.ceil(.001 * len(prior))))
    original_sizes = np.bincount(original_labels, minlength=original_components)
    original_significant = int(np.sum(original_sizes >= significant_size))
    trace: list[dict] = []
    accepted: tuple[np.ndarray, dict] | None = None
    for alpha in (1., .85, .70, .55, .40, .25, .15):
        candidate = prior + float(alpha) * (warped - prior)
        after = np.linalg.norm(candidate[unique_target_ids] - unique_targets, axis=1)
        before_median = _weighted_median(initial_residual, unique_weights)
        after_median = _weighted_median(after, unique_weights)
        improvement = 1. - after_median / max(before_median, 1e-12)
        strain = _edge_strain(prior, candidate, carrier_graph)
        coverage, per_view = _orthographic_coverage(prior, candidate, resolution=int(coverage_resolution))
        edge_ratio_now = np.linalg.norm(candidate[carrier_rows] - candidate[carrier_cols], axis=1) / np.maximum(
            np.linalg.norm(prior[carrier_rows] - prior[carrier_cols], axis=1), 1e-12,
        )
        effective_graph = carrier_graph.copy().tolil()
        broken = edge_ratio_now > float(maximum_edge_stretch)
        if np.any(broken):
            effective_graph[carrier_rows[broken], carrier_cols[broken]] = 0.
            effective_graph[carrier_cols[broken], carrier_rows[broken]] = 0.
        candidate_components, candidate_labels = csgraph.connected_components(
            effective_graph.tocsr(), directed=False,
        )
        candidate_sizes = np.bincount(candidate_labels, minlength=candidate_components)
        candidate_significant = int(np.sum(candidate_sizes >= significant_size))
        candidate_small_mass = int(candidate_sizes[candidate_sizes < significant_size].sum())
        record = {
            "alpha": float(alpha), "observed_residual_before": before_median,
            "observed_residual_after": after_median, "observed_improvement": float(improvement),
            "hidden_coverage_ratio": coverage, "hidden_coverage_per_view": per_view,
            "original_graph_components": int(original_components),
            "candidate_graph_components": int(candidate_components),
            "significant_component_size": int(significant_size),
            "original_significant_components": original_significant,
            "candidate_significant_components": candidate_significant,
            "candidate_small_component_mass": candidate_small_mass,
            "robust_edge_quantile": .99,
            **strain,
        }
        trace.append(record)
        if (
            improvement >= float(minimum_improvement)
            and strain["edge_stretch_p01"] >= float(minimum_edge_compression)
            and strain["edge_stretch_p99"] <= float(maximum_edge_stretch)
            and coverage >= float(minimum_hidden_coverage)
            and candidate_significant <= original_significant
        ):
            accepted = candidate, record
            break
    if accepted is None:
        return prior.copy(), {
            "active": False, "reason": "no_pareto_valid_embedded_update",
            "graph_nodes": int(len(nodes)), "node_graph_edges": int(node_edges),
            "attachment_edges": int(attachment_edges), "carrier_graph_edges": int(carrier_edges),
            "carrier_attachment_edges": int(carrier_attachment_edges),
            "coherent_targets": coherence_info, "node_solver": node_info,
            "line_search": trace,
        }, np.zeros(len(prior), dtype=bool)
    candidate, accepted_info = accepted
    moved = np.linalg.norm(candidate - prior, axis=1) > 1e-8
    return candidate, {
        "active": True, "field": "hierarchical_embedded_arap_posterior",
        "graph_nodes": int(len(nodes)), "node_graph_edges": int(node_edges),
        "attachment_edges": int(attachment_edges), "carrier_graph_edges": int(carrier_edges),
        "carrier_attachment_edges": int(carrier_attachment_edges),
        "target_gaussians": int(len(unique_target_ids)), "stable_gaussians": int(stable_mask.sum()),
        "moved_gaussians": int(moved.sum()), "moved_fraction": float(moved.mean()),
        "maximum_displacement": float(np.linalg.norm(candidate - prior, axis=1).max(initial=0.)),
        "mean_displacement": float(np.linalg.norm(candidate - prior, axis=1).mean()),
        "coherent_targets": coherence_info, "node_solver": node_info,
        "line_search": trace, **accepted_info,
    }, moved


def _project_small_topology_breaks(
    initial: np.ndarray,
    candidate: np.ndarray,
    stable_mask: np.ndarray,
    *,
    neighbours: int,
    edge_ratio: float,
    component_threshold: int,
    maximum_passes: int = 3,
) -> tuple[np.ndarray, dict]:
    """Project accidental tiny fragments back onto their structural motion.

    A sparse Gaussian carrier can contain pre-existing isolated samples.  The
    relevant failure is therefore not the absolute component count, but a
    *new* small fragment split from a previously significant component.  Such
    points inherit the robust displacement of their neighbours in the rest
    graph.  This projection is category- and axis-free and leaves stable
    observations fixed.
    """
    initial = np.asarray(initial, dtype=np.float64)
    projected = np.asarray(candidate, dtype=np.float64).copy()
    stable_mask = np.asarray(stable_mask, dtype=bool)
    rest_graph, _, _ = _local_gaussian_graph(
        initial, neighbours=int(neighbours), edge_ratio=float(edge_ratio),
    )
    input_components, input_labels = csgraph.connected_components(rest_graph, directed=False)
    input_sizes = np.bincount(input_labels, minlength=input_components)
    corrected: set[int] = set()
    trace: list[dict] = []
    for pass_id in range(max(1, int(maximum_passes))):
        output_graph, _, _ = _local_gaussian_graph(
            projected, neighbours=int(neighbours), edge_ratio=float(edge_ratio),
        )
        output_components, output_labels = csgraph.connected_components(output_graph, directed=False)
        output_sizes = np.bincount(output_labels, minlength=output_components)
        bad_components: list[int] = []
        for label, size in enumerate(output_sizes):
            if int(size) >= int(component_threshold):
                continue
            members = np.flatnonzero(output_labels == label)
            if len(members) and np.any(input_sizes[input_labels[members]] >= int(component_threshold)):
                bad_components.append(int(label))
        bad_ids = np.flatnonzero(np.isin(output_labels, bad_components))
        bad_ids = bad_ids[~stable_mask[bad_ids]]
        trace.append({
            "pass": int(pass_id),
            "components_before": int(output_components),
            "new_small_fragment_points": int(len(bad_ids)),
        })
        if not len(bad_ids):
            break
        displacement = projected - initial
        replacement = projected[bad_ids].copy()
        for local, point_id in enumerate(bad_ids):
            start, stop = rest_graph.indptr[point_id], rest_graph.indptr[point_id + 1]
            adjacent = rest_graph.indices[start:stop]
            adjacent = adjacent[~np.isin(adjacent, bad_ids)]
            if not len(adjacent):
                same_component = np.flatnonzero(input_labels == input_labels[point_id])
                same_component = same_component[same_component != point_id]
                if len(same_component):
                    k = min(8, len(same_component))
                    nearest = cKDTree(initial[same_component]).query(initial[point_id], k=k)[1]
                    adjacent = same_component[np.atleast_1d(nearest)]
            if len(adjacent):
                replacement[local] = initial[point_id] + np.median(displacement[adjacent], axis=0)
        projected[bad_ids] = replacement
        corrected.update(int(value) for value in bad_ids)
    projected[stable_mask] = initial[stable_mask]
    return projected, {
        "active": bool(corrected),
        "corrected_points": int(len(corrected)),
        "passes": trace,
    }


def _compact_visible_score(score: dict) -> dict:
    projection = score["projection"]
    geometric = score["geometric"]
    return {
        "objective": float(score["objective"]),
        "visible_3d": float(geometric["objective"]),
        "visible_pair_count": int(len(geometric["partial_ids"])),
        "silhouette_iou": float(projection["iou"]),
        "coverage": float(projection["coverage"]),
        "leakage": float(projection["leakage"]),
        "depth_error_normalized": float(projection["visible_depth_normalized"]),
    }


class PosteriorAdapter:
    """One category-agnostic coarse-to-fine complete-prior update."""

    def __init__(self, config: PosteriorAdapterConfig | None = None, *, device: str = "cuda"):
        self.config = config or PosteriorAdapterConfig()
        self.config.validate()
        self.device = str(device)

    def run(self, prior: np.ndarray, partial: np.ndarray, projector) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
        prior = np.asarray(prior, dtype=np.float64)
        partial = np.asarray(partial, dtype=np.float64)
        initial = prior.copy()
        diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-9)
        stage_records: list[dict] = []
        transports: list[TransportResult] = []
        displacements: list[np.ndarray] = []
        inherited_stable = np.zeros(len(prior), dtype=bool)
        stage_specs = (
            ("coarse", self.config.coarse_neighbours, self.config.coarse_edge_ratio,
             self.config.coarse_data_weight, self.config.coarse_screening,
             self.config.coarse_iterations, self.config.coarse_displacement_ratio,
             self.config.coarse_node_fraction),
            ("fine", self.config.fine_neighbours, self.config.fine_edge_ratio,
             self.config.fine_data_weight, self.config.fine_screening,
             self.config.fine_iterations, self.config.fine_displacement_ratio,
             self.config.fine_node_fraction),
        )
        current = prior.copy()
        for name, neighbours, edge_ratio, data_weight, screening, iterations, displacement_ratio, node_fraction in stage_specs:
            transport = partial_optimal_transport(
                current, partial, projector, config=self.config, device=self.device,
            )
            transports.append(transport)
            stable = inherited_stable | transport.stable_prior_mask
            residual_pair = np.isin(transport.prior_ids, np.flatnonzero(transport.residual_prior_mask))
            target_ids = transport.prior_ids[residual_pair]
            targets = partial[transport.partial_ids[residual_pair]]
            weights = transport.weights[residual_pair]
            if len(target_ids) < int(self.config.minimum_transport_pairs):
                stage_records.append({
                    "stage": name, "active": False, "reason": "insufficient_residual_transport_pairs",
                    "transport": transport.diagnostics,
                })
                displacements.append(np.zeros_like(current))
                if not bool(transport.diagnostics.get("localized_residual_evidence", False)):
                    break
                continue
            updated, solve, moved = _embedded_arap_posterior(
                current, target_ids, targets, weights, stable,
                node_fraction=float(node_fraction),
                minimum_nodes=int(self.config.minimum_graph_nodes),
                maximum_nodes=int(self.config.maximum_graph_nodes),
                skinning_neighbours=int(self.config.skinning_neighbours),
                neighbours=int(neighbours), edge_ratio=float(edge_ratio),
                attachment_edge_ratio=float(self.config.attachment_edge_ratio),
                data_weight=float(data_weight), screening=float(screening), iterations=int(iterations),
                maximum_displacement=float(displacement_ratio) * diagonal,
                minimum_improvement=float(self.config.minimum_observed_improvement),
                minimum_edge_compression=float(self.config.minimum_edge_compression),
                maximum_edge_stretch=float(self.config.maximum_edge_stretch),
                minimum_hidden_coverage=float(self.config.minimum_hidden_coverage),
                coverage_resolution=int(self.config.coverage_resolution),
            )
            stage_records.append({"stage": name, "transport": transport.diagnostics, "solver": solve})
            displacements.append(updated - current)
            if solve.get("active"):
                current = updated
                inherited_stable = stable
        while len(displacements) < 2:
            displacements.append(np.zeros_like(current))
        component_threshold = max(32, int(math.ceil(.001 * len(initial))))
        current, topology_projection = _project_small_topology_breaks(
            initial, current, transports[0].stable_prior_mask,
            neighbours=int(self.config.fine_neighbours),
            edge_ratio=float(self.config.fine_edge_ratio),
            component_threshold=int(component_threshold),
        )
        initial_visible = visible_score(partial, initial, projector, diagonal)
        posterior_visible = visible_score(partial, current, projector, diagonal)
        # The local OT objective is intentionally one-sided.  A candidate is
        # therefore accepted only on the Pareto frontier: it must improve its
        # transported residual (enforced inside the solver) without worsening
        # the full visible observation.  This strict comparison introduces no
        # dataset, category, scale, or hand-tuned acceptance threshold.
        visible_pareto_accepted = bool(
            not np.any(np.linalg.norm(current - initial, axis=1) > 1e-8)
            or float(posterior_visible["objective"]) <= float(initial_visible["objective"])
        )
        posterior_verifier = {
            "criterion": "local_transport_improves_and_global_visible_score_does_not_worsen",
            "accepted": visible_pareto_accepted,
            "initial": _compact_visible_score(initial_visible),
            "candidate": _compact_visible_score(posterior_visible),
            "objective_delta": float(posterior_visible["objective"] - initial_visible["objective"]),
            "ground_truth_cd_emd_used": False,
        }
        if not visible_pareto_accepted:
            current = initial.copy()
            displacements = [np.zeros_like(initial), np.zeros_like(initial)]
            topology_projection = {**topology_projection, "candidate_reverted_by_visible_pareto": True}
        final_coverage, final_coverage_views = _orthographic_coverage(
            initial, current, resolution=int(self.config.coverage_resolution),
        )
        final_graph, _, _ = _local_gaussian_graph(
            initial, neighbours=int(self.config.fine_neighbours), edge_ratio=float(self.config.fine_edge_ratio),
        )
        posterior_graph, _, _ = _local_gaussian_graph(
            current, neighbours=int(self.config.fine_neighbours), edge_ratio=float(self.config.fine_edge_ratio),
        )
        final_strain = _edge_strain(initial, current, final_graph)
        input_components, input_labels = csgraph.connected_components(final_graph, directed=False)
        output_components, output_labels = csgraph.connected_components(posterior_graph, directed=False)
        input_sizes = np.bincount(input_labels, minlength=input_components)
        output_sizes = np.bincount(output_labels, minlength=output_components)
        edge_rows, edge_cols = sparse.triu(final_graph, k=1).nonzero()
        tracked_edge_ratio = np.linalg.norm(current[edge_rows] - current[edge_cols], axis=1) / np.maximum(
            np.linalg.norm(initial[edge_rows] - initial[edge_cols], axis=1), 1e-12,
        )
        final_transport = transports[-1]
        # ``supported_stable`` is deliberately the support detected before
        # any deformation.  It is the auditable no-motion set.  Later stages
        # may discover additional now-explained points, recorded separately,
        # but must not retroactively describe a moved point as initially
        # stable.
        initial_stable = transports[0].stable_prior_mask
        stable_motion = np.linalg.norm(current[initial_stable] - initial[initial_stable], axis=1)
        integrity = {
            "input_connected_components": int(input_components),
            "output_connected_components": int(output_components),
            "new_connected_components": int(max(0, output_components - input_components)),
            "component_significance_size": int(component_threshold),
            "input_significant_components": int(np.sum(input_sizes >= component_threshold)),
            "output_significant_components": int(np.sum(output_sizes >= component_threshold)),
            "input_component_sizes": np.sort(input_sizes)[::-1][:16].tolist(),
            "output_component_sizes": np.sort(output_sizes)[::-1][:16].tolist(),
            "tracked_original_edges": int(len(edge_rows)),
            "tracked_edge_replacement_fraction": float(np.mean(
                tracked_edge_ratio > float(self.config.maximum_edge_stretch),
            )) if len(tracked_edge_ratio) else 0.,
            "stable_motion_p50": float(np.quantile(stable_motion, .50)) if len(stable_motion) else 0.,
            "stable_motion_p95": float(np.quantile(stable_motion, .95)) if len(stable_motion) else 0.,
            "stable_motion_max": float(stable_motion.max(initial=0.)),
        }
        integrity["passed"] = bool(
            output_components <= input_components
            and integrity["output_significant_components"] <= integrity["input_significant_components"]
            and final_coverage >= float(self.config.minimum_hidden_coverage)
            and len(current) == len(initial)
        )
        masks = {
            "supported_stable": initial_stable,
            "supported_stable_final": inherited_stable,
            "supported_residual": final_transport.residual_prior_mask,
            "unsupported": final_transport.unsupported_prior_mask,
            "moved": np.linalg.norm(current - initial, axis=1) > 1e-8,
            "coarse_displacement": displacements[0],
            "fine_displacement": displacements[1],
            "transport_partial_ids": final_transport.partial_ids,
            "transport_prior_ids": final_transport.prior_ids,
            "transport_weights": final_transport.weights,
            "transport_costs": final_transport.costs,
        }
        info = {
            "active": bool(np.any(masks["moved"])),
            "method": "structure_aware_partial_ot_complete_gaussian_posterior",
            "strict_zero_shot": True,
            "ground_truth_cd_emd_used": False,
            "category_or_part_rules_used": False,
            "carrier_slots_before": int(len(initial)),
            "carrier_slots_after": int(len(current)),
            "carrier_slots_preserved": bool(len(initial) == len(current)),
            "config": asdict(self.config),
            "stages": stage_records,
            "supported_stable": int(masks["supported_stable"].sum()),
            "supported_stable_final": int(masks["supported_stable_final"].sum()),
            "supported_residual": int(masks["supported_residual"].sum()),
            "unsupported": int(masks["unsupported"].sum()),
            "moved": int(masks["moved"].sum()),
            "hidden_coverage_ratio": final_coverage,
            "hidden_coverage_per_view": final_coverage_views,
            "integrity": integrity,
            "topology_projection": topology_projection,
            "posterior_verifier": posterior_verifier,
            **final_strain,
        }
        return current, info, masks
