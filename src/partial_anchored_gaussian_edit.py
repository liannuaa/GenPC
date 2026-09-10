"""Single-view boundary-conditioned edits for a partial-anchored Gaussian field.

The edit is deliberately a geometry-only 3DGS mean update: observed partial
points are immutable evidence, while the complete Pixal population remains the
only complete-body carrier.  In particular, this module never creates a raw
point-cloud union and never inspects GT metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

from src.partial_anchored_gaussian_decode import collision_free_anchor_pairs
from src.zbuffer import zbuffer_depth_with_indices


def _clip_vectors(vectors: np.ndarray, maximum: float) -> np.ndarray:
    """Bound vector length without changing its direction."""
    magnitude = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * np.minimum(1., float(maximum) / np.maximum(magnitude, 1e-12))


def _solve_cg(matrix: sparse.csr_matrix, rhs: np.ndarray, *, tolerance: float,
              max_iterations: int) -> tuple[np.ndarray, list[int]]:
    """Solve the three displacement coordinates with a compatible SciPy CG API."""
    solution = np.zeros((matrix.shape[0], rhs.shape[1]), dtype=np.float64)
    status: list[int] = []
    for axis in range(rhs.shape[1]):
        try:
            vector, code = sparse_linalg.cg(
                matrix, rhs[:, axis], rtol=float(tolerance), atol=0.,
                maxiter=int(max_iterations),
            )
        except TypeError:  # SciPy < 1.12
            vector, code = sparse_linalg.cg(
                matrix, rhs[:, axis], tol=float(tolerance), maxiter=int(max_iterations),
            )
        solution[:, axis] = vector
        status.append(int(code))
    return solution, status


def visible_axis_stretch(
    prior: np.ndarray,
    partial: np.ndarray,
    pixel_pairs: np.ndarray,
    *,
    max_anchor_residual: float,
    max_log_stretch: float = .16,
    minimum_pairs: int = 512,
    minimum_anisotropy: float = .025,
    axis_basis: np.ndarray | None = None,
    minimum_axis_spread_ratio: float = .08,
    minimum_residual_reduction: float = .03,
    trim_quantile: float = .75,
    iterations: int = 3,
    retain_isotropic_component: bool = False,
) -> tuple[np.ndarray, dict]:
    """Estimate a Camera-1 relative-axis scale displacement from partial anchors.

    Proper Sim(3) deliberately owns the global gauge.  This routine therefore
    removes the geometric-mean scale and translation from a robust local fit,
    retaining only relative stretch along observed axes by default.  When
    ``retain_isotropic_component`` is enabled, the same fully supported fit
    additionally keeps its uniform scale component as a coherent
    whole-carrier proposal.  When ``axis_basis`` is supplied, its rows are
    the world-space Camera-1 axes and the local scale is estimated in
    image-horizontal, image-vertical, and depth coordinates.  Otherwise a
    PCA basis is used for backwards-compatible generic geometry tests. The
    returned displacement is a coherent affine carrier update about the
    robust visible-prior centre. It applies only scale, never a residual
    translation, so it can precede the bounded local Gaussian edit without
    breaking the complete prior's continuity.

    Consequently an unconstrained monocular depth direction cannot by itself
    trigger an edit: the measured matched surface must contain enough
    well-conditioned three-dimensional support and exhibit a non-isotropic
    residual after the Sim(3) component is factored out.
    """
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    zero = np.zeros_like(prior)
    anchors, anchor_info = collision_free_anchor_pairs(
        pixel_pairs, partial, prior, max_residual=float(max_anchor_residual),
    )
    if len(anchors) < int(minimum_pairs):
        return zero, {
            **anchor_info, "active": False, "reason": "insufficient_visible_axis_anchors",
            "controls": int(len(anchors)),
        }
    if minimum_axis_spread_ratio <= 0. or minimum_axis_spread_ratio >= 1.:
        raise ValueError("minimum_axis_spread_ratio must lie in (0, 1)")
    if minimum_residual_reduction <= 0. or minimum_residual_reduction >= 1.:
        raise ValueError("minimum_residual_reduction must lie in (0, 1)")
    fixed_basis = None
    if axis_basis is not None:
        fixed_basis = np.asarray(axis_basis, dtype=np.float64)
        if fixed_basis.shape != (3, 3) or not np.isfinite(fixed_basis).all():
            raise ValueError("axis_basis must be a finite (3, 3) orthonormal row-axis matrix")
        if not np.allclose(fixed_basis @ fixed_basis.T, np.eye(3), rtol=1e-4, atol=1e-5):
            raise ValueError("axis_basis must be orthonormal")
    partial_ids, prior_ids = anchors[:, 0], anchors[:, 1]
    source, target = prior[prior_ids], partial[partial_ids]
    keep = np.ones(len(source), dtype=bool)
    trace: list[dict] = []
    basis = np.eye(3, dtype=np.float64) if fixed_basis is None else fixed_basis.copy()
    scale = np.ones(3, dtype=np.float64)
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    for _ in range(int(iterations)):
        if int(keep.sum()) < int(minimum_pairs):
            break
        source_kept, target_kept = source[keep], target[keep]
        source_centre, target_centre = source_kept.mean(axis=0), target_kept.mean(axis=0)
        _, singular, pca_basis = np.linalg.svd(source_kept - source_centre, full_matrices=False)
        if fixed_basis is None:
            basis = pca_basis
        # A visible surface can be nearly planar, so two independent observed
        # directions are enough for screen-plane scaling.  A single line is
        # not a stable support for any local scale decision.
        if not np.isfinite(singular).all() or singular[1] <= max(singular[0] * 1e-4, 1e-10):
            return zero, {
                **anchor_info, "active": False, "reason": "degenerate_visible_axis_support",
                "controls": int(len(anchors)), "singular_values": singular.tolist(),
            }
        source_coordinates = (source_kept - source_centre) @ basis.T
        target_coordinates = (target_kept - target_centre) @ basis.T
        source_energy = np.mean(np.square(source_coordinates), axis=0)
        target_energy = np.mean(np.square(target_coordinates), axis=0)
        raw_scale = np.sqrt(target_energy / np.maximum(source_energy, 1e-12))
        scale = np.clip(raw_scale, np.exp(-float(max_log_stretch)), np.exp(float(max_log_stretch)))
        predicted = target_centre + ((source - source_centre) @ basis.T * scale) @ basis
        residual = np.linalg.norm(predicted - target, axis=1)
        limit = max(float(np.quantile(residual, float(trim_quantile))), 1e-9)
        keep = residual <= limit
        trace.append({
            "kept": int(keep.sum()), "raw_scale": raw_scale.tolist(), "clipped_scale": scale.tolist(),
            "residual_median": float(np.median(residual)), "residual_trim_limit": limit,
        })
    if int(keep.sum()) < int(minimum_pairs):
        return zero, {
            **anchor_info, "active": False, "reason": "insufficient_trimmed_visible_axis_anchors",
            "controls": int(len(anchors)), "trace": trace,
        }
    source_kept, target_kept = source[keep], target[keep]
    source_centre, target_centre = source_kept.mean(axis=0), target_kept.mean(axis=0)
    _, singular, pca_basis = np.linalg.svd(source_kept - source_centre, full_matrices=False)
    if fixed_basis is None:
        basis = pca_basis
    source_coordinates = (source_kept - source_centre) @ basis.T
    target_coordinates = (target_kept - target_centre) @ basis.T
    source_spread = np.sqrt(np.mean(np.square(source_coordinates), axis=0))
    target_spread = np.sqrt(np.mean(np.square(target_coordinates), axis=0))
    spread_floor = float(minimum_axis_spread_ratio) * max(float(source_spread.max()), 1e-12)
    axis_supported = (source_spread >= spread_floor) & (target_spread >= spread_floor)
    raw_scale = np.sqrt(
        np.mean(np.square(target_coordinates), axis=0)
        / np.maximum(np.mean(np.square(source_coordinates), axis=0), 1e-12)
    )
    scale = np.ones(3, dtype=np.float64)
    scale[axis_supported] = np.clip(
        raw_scale[axis_supported], np.exp(-float(max_log_stretch)), np.exp(float(max_log_stretch)),
    )
    if not np.any(axis_supported):
        return zero, {
            **anchor_info, "active": False, "reason": "insufficient_directional_camera_support",
            "controls": int(len(anchors)), "source_axis_spread": source_spread.tolist(),
            "target_axis_spread": target_spread.tolist(), "axis_supported": axis_supported.tolist(),
        }
    # If all three Camera-1 directions are observed, factor out their common
    # scale because the preceding proper Sim(3) already owns that gauge.  If
    # one direction is unobserved, it must remain fixed rather than receive an
    # inverse scale merely to make a global geometric mean equal one.
    isotropic_scale = 1.
    relative_scale = scale.copy()
    if bool(np.all(axis_supported)):
        isotropic_scale = float(np.exp(np.mean(np.log(np.maximum(scale, 1e-12)))))
        relative_scale = scale / isotropic_scale
    anisotropy = float(np.max(np.abs(np.log(np.maximum(relative_scale, 1e-12)))))
    isotropic_prediction = target_centre + ((source_kept - source_centre) @ basis.T * isotropic_scale) @ basis
    anisotropic_prediction = target_centre + ((source_kept - source_centre) @ basis.T * scale) @ basis
    isotropic_error = float(np.mean(np.linalg.norm(isotropic_prediction - target_kept, axis=1)))
    anisotropic_error = float(np.mean(np.linalg.norm(anisotropic_prediction - target_kept, axis=1)))
    residual_reduction = float(1. - anisotropic_error / max(isotropic_error, 1e-12))
    if (
        anisotropy < float(minimum_anisotropy)
        or residual_reduction < float(minimum_residual_reduction)
    ):
        return zero, {
            **anchor_info, "active": False, "reason": "no_supported_anisotropic_residual",
            "controls": int(len(anchors)), "relative_scale": relative_scale.tolist(),
            "anisotropy_log": anisotropy, "isotropic_error": isotropic_error,
            "anisotropic_error": anisotropic_error, "axis_supported": axis_supported.tolist(),
            "source_axis_spread": source_spread.tolist(), "target_axis_spread": target_spread.tolist(),
            "residual_reduction": residual_reduction,
            "minimum_residual_reduction": float(minimum_residual_reduction),
            "trace": trace,
        }
    applied_scale = scale if bool(retain_isotropic_component) else relative_scale
    all_coordinates = (prior - source_centre) @ basis.T
    warped = source_centre + (all_coordinates * applied_scale) @ basis
    return warped - prior, {
        **anchor_info, "active": True, "controls": int(len(anchors)),
        "trimmed_controls": int(keep.sum()), "principal_axes": basis.tolist(),
        "visible_scale": scale.tolist(), "relative_scale": relative_scale.tolist(),
        "applied_scale": applied_scale.tolist(),
        "retain_isotropic_component": bool(retain_isotropic_component),
        "isotropic_scale_removed": isotropic_scale, "anisotropy_log": anisotropy,
        "isotropic_error": isotropic_error, "anisotropic_error": anisotropic_error,
        "axis_supported": axis_supported.tolist(),
        "source_axis_spread": source_spread.tolist(), "target_axis_spread": target_spread.tolist(),
        "minimum_axis_spread_ratio": float(minimum_axis_spread_ratio),
        "axis_frame": "saved_camera" if fixed_basis is not None else "visible_pca",
        "residual_reduction": residual_reduction,
        "max_log_stretch": float(max_log_stretch), "minimum_anisotropy": float(minimum_anisotropy),
        "minimum_residual_reduction": float(minimum_residual_reduction),
        "singular_values": singular.tolist(), "trace": trace,
        "field": "camera1_supported_relative_axis_scale_carrier",
    }


def visible_depth_extent_calibration(
    prior: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    axis_basis: np.ndarray,
    quantile: float = .01,
    minimum_visible_points: int = 512,
    minimum_log_expansion: float = .025,
    max_log_expansion: float = .25,
) -> tuple[np.ndarray, dict]:
    """Calibrate a globally short visible carrier along Camera-1 depth.

    The registered prior and partial can agree on their pixel-aligned central
    surface while disagreeing in their *visible depth extent*.  Such a
    discrepancy is a global camera-frame scale error, not a local Gaussian
    deformation.  This routine therefore compares robust two-sided depth
    quantiles of the complete prior's Camera-1 z-buffer with all visible
    partial samples, then applies one coherent whole-carrier depth scale and
    the associated depth translation.  It proposes expansion only: an
    incomplete scan must never shrink the complete prior's hidden support.
    """
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    zero = np.zeros_like(prior)
    if not 0. < float(quantile) < .5:
        raise ValueError("quantile must lie in (0, .5)")
    if int(minimum_visible_points) < 6 or min(float(minimum_log_expansion), float(max_log_expansion)) <= 0.:
        raise ValueError("visible support and scale bounds must be positive")
    if float(minimum_log_expansion) > float(max_log_expansion):
        raise ValueError("minimum_log_expansion cannot exceed max_log_expansion")
    basis = np.asarray(axis_basis, dtype=np.float64)
    if basis.shape != (3, 3) or not np.allclose(basis @ basis.T, np.eye(3), rtol=1e-4, atol=1e-5):
        raise ValueError("axis_basis must be an orthonormal (3, 3) camera-axis matrix")
    partial_uv, partial_depth = projector.project(partial)
    prior_uv, prior_depth = projector.project(prior)
    _, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=0,
    )
    _, prior_mask, prior_index = zbuffer_depth_with_indices(
        prior_uv, prior_depth, projector.image_shape, splat_radius=1,
    )
    partial_ids = np.unique(partial_index[partial_mask])
    prior_ids = np.unique(prior_index[prior_mask])
    partial_ids, prior_ids = partial_ids[partial_ids >= 0], prior_ids[prior_ids >= 0]
    if min(len(partial_ids), len(prior_ids)) < int(minimum_visible_points):
        return zero, {
            "active": False, "reason": "insufficient_two_sided_visible_extent_support",
            "partial_visible_points": int(len(partial_ids)), "prior_visible_points": int(len(prior_ids)),
        }
    partial_depth_axis = (partial[partial_ids] @ basis.T)[:, 2]
    prior_depth_axis = (prior[prior_ids] @ basis.T)[:, 2]
    partial_low, partial_high = np.quantile(partial_depth_axis, [float(quantile), 1. - float(quantile)])
    prior_low, prior_high = np.quantile(prior_depth_axis, [float(quantile), 1. - float(quantile)])
    partial_extent, prior_extent = float(partial_high - partial_low), float(prior_high - prior_low)
    if prior_extent <= 1e-9 or partial_extent <= 1e-9:
        return zero, {
            "active": False, "reason": "degenerate_visible_depth_extent",
            "partial_extent": partial_extent, "prior_extent": prior_extent,
        }
    requested_scale = float(partial_extent / prior_extent)
    requested_log = float(np.log(max(requested_scale, 1e-12)))
    if requested_log < float(minimum_log_expansion):
        return zero, {
            "active": False, "reason": "no_supported_visible_depth_expansion",
            "requested_depth_scale": requested_scale, "requested_log_expansion": requested_log,
            "partial_extent": partial_extent, "prior_extent": prior_extent,
        }
    applied_scale = float(min(requested_scale, np.exp(float(max_log_expansion))))
    prior_centre = .5 * float(prior_low + prior_high)
    partial_centre = .5 * float(partial_low + partial_high)
    depth_translation = float(partial_centre - applied_scale * prior_centre)
    coordinates = prior @ basis.T
    coordinates[:, 2] = applied_scale * coordinates[:, 2] + depth_translation
    warped = coordinates @ basis
    return warped - prior, {
        "active": True,
        "method": "camera1_visible_depth_extent_global_calibration",
        "partial_visible_points": int(len(partial_ids)), "prior_visible_points": int(len(prior_ids)),
        "quantile": float(quantile),
        "partial_depth_interval": [float(partial_low), float(partial_high)],
        "prior_depth_interval": [float(prior_low), float(prior_high)],
        "partial_extent": partial_extent, "prior_extent": prior_extent,
        "requested_depth_scale": requested_scale, "applied_depth_scale": applied_scale,
        "depth_translation": depth_translation,
        "minimum_log_expansion": float(minimum_log_expansion),
        "max_log_expansion": float(max_log_expansion),
        "carrier_slots_preserved": bool(len(warped) == len(prior)),
        "field": "camera1_visible_depth_extent_global_carrier",
    }


def visible_principal_extent_calibration(
    prior: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    quantile: float = .01,
    minimum_visible_points: int = 512,
    minimum_axis_anisotropy: float = 1.15,
    anchor_tolerance: float = .03,
    minimum_log_expansion: float = .025,
    max_log_expansion: float = .25,
) -> tuple[np.ndarray, dict]:
    """Apply a globally anchored length calibration along the partial's main axis.

    A complete carrier may agree with a partial scan at one end of an elongated
    object while being globally too short at the other.  The mismatch is not a
    local deformation: it is a one-dimensional global scale error.  This
    routine obtains the stable principal axis from the partial, compares robust
    two-sided extents against the z-buffer-visible prior, and expands the
    entire carrier about the already aligned endpoint.  It never contracts an
    unobserved prior and declines nearly isotropic observations.
    """
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    zero = np.zeros_like(prior)
    if not 0. < float(quantile) < .5:
        raise ValueError("quantile must lie in (0, .5)")
    if min(int(minimum_visible_points), float(minimum_axis_anisotropy), float(anchor_tolerance),
           float(minimum_log_expansion), float(max_log_expansion)) <= 0.:
        raise ValueError("principal-axis support and scale bounds must be positive")
    if float(minimum_log_expansion) > float(max_log_expansion):
        raise ValueError("minimum_log_expansion cannot exceed max_log_expansion")
    partial_centre = partial.mean(axis=0)
    _, singular, axes = np.linalg.svd(partial - partial_centre, full_matrices=False)
    if not np.isfinite(singular).all() or singular[1] <= 1e-12:
        return zero, {"active": False, "reason": "degenerate_partial_principal_axis"}
    axis_anisotropy = float(singular[0] / singular[1])
    if axis_anisotropy < float(minimum_axis_anisotropy):
        return zero, {
            "active": False, "reason": "insufficient_principal_axis_anisotropy",
            "singular_values": singular.tolist(), "axis_anisotropy": axis_anisotropy,
        }
    partial_uv, partial_depth = projector.project(partial)
    prior_uv, prior_depth = projector.project(prior)
    _, partial_mask, partial_index = zbuffer_depth_with_indices(
        partial_uv, partial_depth, projector.image_shape, splat_radius=0,
    )
    _, prior_mask, prior_index = zbuffer_depth_with_indices(
        prior_uv, prior_depth, projector.image_shape, splat_radius=1,
    )
    partial_ids = np.unique(partial_index[partial_mask])
    prior_ids = np.unique(prior_index[prior_mask])
    partial_ids, prior_ids = partial_ids[partial_ids >= 0], prior_ids[prior_ids >= 0]
    if min(len(partial_ids), len(prior_ids)) < int(minimum_visible_points):
        return zero, {
            "active": False, "reason": "insufficient_visible_principal_extent_support",
            "partial_visible_points": int(len(partial_ids)), "prior_visible_points": int(len(prior_ids)),
        }
    axis = axes[0]
    partial_coordinate = partial[partial_ids] @ axis
    prior_coordinate = prior[prior_ids] @ axis
    partial_low, partial_high = np.quantile(partial_coordinate, [float(quantile), 1. - float(quantile)])
    prior_low, prior_high = np.quantile(prior_coordinate, [float(quantile), 1. - float(quantile)])
    partial_extent, prior_extent = float(partial_high - partial_low), float(prior_high - prior_low)
    if prior_extent <= 1e-9 or partial_extent <= 1e-9:
        return zero, {
            "active": False, "reason": "degenerate_visible_principal_extent",
            "partial_extent": partial_extent, "prior_extent": prior_extent,
        }
    requested_scale = float(partial_extent / prior_extent)
    requested_log = float(np.log(max(requested_scale, 1e-12)))
    if requested_log < float(minimum_log_expansion):
        return zero, {
            "active": False, "reason": "no_supported_principal_axis_expansion",
            "requested_axis_scale": requested_scale, "requested_log_expansion": requested_log,
            "partial_extent": partial_extent, "prior_extent": prior_extent,
        }
    endpoint_errors = np.asarray((abs(float(partial_low - prior_low)), abs(float(partial_high - prior_high))))
    anchor_index = int(np.argmin(endpoint_errors))
    if float(endpoint_errors[anchor_index]) > float(anchor_tolerance):
        return zero, {
            "active": False, "reason": "no_aligned_principal_extent_anchor",
            "endpoint_errors": endpoint_errors.tolist(), "anchor_tolerance": float(anchor_tolerance),
        }
    applied_scale = float(min(requested_scale, np.exp(float(max_log_expansion))))
    prior_anchor = float((prior_low, prior_high)[anchor_index])
    partial_anchor = float((partial_low, partial_high)[anchor_index])
    axis_translation = float(partial_anchor - applied_scale * prior_anchor)
    coordinate = prior @ axis
    displacement = ((applied_scale - 1.) * coordinate + axis_translation)[:, None] * axis[None, :]
    warped = prior + displacement
    return displacement, {
        "active": True,
        "method": "partial_principal_axis_visible_extent_global_calibration",
        "partial_visible_points": int(len(partial_ids)), "prior_visible_points": int(len(prior_ids)),
        "quantile": float(quantile), "principal_axis": axis.tolist(),
        "singular_values": singular.tolist(), "axis_anisotropy": axis_anisotropy,
        "partial_axis_interval": [float(partial_low), float(partial_high)],
        "prior_axis_interval": [float(prior_low), float(prior_high)],
        "partial_extent": partial_extent, "prior_extent": prior_extent,
        "requested_axis_scale": requested_scale, "applied_axis_scale": applied_scale,
        "endpoint_errors": endpoint_errors.tolist(), "anchored_endpoint": ("low", "high")[anchor_index],
        "axis_translation": axis_translation, "anchor_tolerance": float(anchor_tolerance),
        "minimum_log_expansion": float(minimum_log_expansion),
        "max_log_expansion": float(max_log_expansion),
        "carrier_slots_preserved": bool(len(warped) == len(prior)),
        "field": "partial_principal_axis_visible_extent_global_carrier",
    }


def _orthographic_prior_protection(
    prior: np.ndarray,
    *,
    views: int,
    bins: int = 192,
) -> np.ndarray:
    """Return prior means exposed in fixed self-rendered protection views.

    These are not invented partial observations.  They are deterministic
    orthographic z-buffers of the initial complete prior along signed PCA
    directions, used only to regularize unsupported deformation.  A point is
    retained if it wins at least one front-most pixel in a protection view.
    """
    if views <= 0:
        return np.zeros(len(prior), dtype=bool)
    centre = prior.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(prior - centre, full_matrices=False)
    coordinates = (prior - centre) @ right.T
    selected = np.zeros(len(prior), dtype=bool)
    directions = [(axis, sign) for axis in range(3) for sign in (-1, 1)][:int(views)]
    for depth_axis, sign in directions:
        planar_axes = [axis for axis in range(3) if axis != depth_axis]
        plane = coordinates[:, planar_axes]
        lower, upper = plane.min(axis=0), plane.max(axis=0)
        extent = np.maximum(upper - lower, 1e-9)
        uv = np.floor((plane - lower) / extent * (int(bins) - 1)).astype(np.int64)
        key = uv[:, 0] * int(bins) + uv[:, 1]
        # Sign chooses either the front or rear side in this deterministic
        # self-render; stable sorting keeps ties reproducible.
        depth = sign * coordinates[:, depth_axis]
        order = np.lexsort((np.arange(len(prior)), depth, key))
        ordered_key = key[order]
        keep = np.concatenate(([True], ordered_key[1:] != ordered_key[:-1]))
        selected[order[keep]] = True
    return selected


def boundary_conditioned_graph_displacement(
    prior: np.ndarray,
    partial: np.ndarray,
    pixel_pairs: np.ndarray,
    *,
    max_anchor_residual: float,
    max_displacement: float,
    neighbours: int = 8,
    edge_ratio: float = 1.8,
    screening: float = .003,
    prior_protection_views: int = 6,
    prior_protection_weight: float = .02,
    protection_exclusion_ratio: float = .08,
    remote_gain: float = 1.,
    remote_gain_radius_ratio: float = .08,
    remote_displacement_cap_multiplier: float = 1.,
    cg_tolerance: float = 1e-5,
    cg_max_iterations: int = 240,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Propagate hard saved-view evidence across the prior's surface graph.

    Pixel-indexed partial/Pixal matches are *Dirichlet controls*: their prior
    Gaussian centres are fixed to the measured partial offsets.  All other
    Pixal Gaussians receive a displacement only through a local 3-D kNN graph.
    The graph is pruned by local sampling scale, which prevents propagation
    across large gaps (e.g. chair legs or the two sides of an armrest).  A
    small zero-displacement screen keeps components with no observation still
    and limits extrapolation into genuinely unobserved structure.

    This is the single-view counterpart of boundary-value Gaussian editing:
    observed image neighbourhoods are fixed, while remote geometry follows
    only if it belongs to the same local surface structure.
    """
    prior = np.asarray(prior, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if len(prior) < 3:
        raise ValueError("at least three prior points are required")
    if min(max_anchor_residual, max_displacement, edge_ratio, screening, cg_tolerance,
           protection_exclusion_ratio, remote_gain, remote_gain_radius_ratio,
           remote_displacement_cap_multiplier) <= 0. or prior_protection_weight < 0.:
        raise ValueError("metric bounds, graph bounds, screening, and tolerance must be positive")
    if neighbours < 2 or cg_max_iterations <= 0:
        raise ValueError("neighbours must be at least two and cg_max_iterations positive")
    anchors, anchor_info = collision_free_anchor_pairs(
        pixel_pairs, partial, prior, max_residual=float(max_anchor_residual),
    )
    if len(anchors) < 6:
        return prior.copy(), anchors, {
            **anchor_info, "active": False,
            "reason": "insufficient_collision_free_anchors",
            "field": "boundary_conditioned_surface_graph",
        }

    partial_ids, control_ids = anchors[:, 0], anchors[:, 1]
    control_offsets = _clip_vectors(
        partial[partial_ids] - prior[control_ids], float(max_displacement),
    )
    is_control = np.zeros(len(prior), dtype=bool)
    is_control[control_ids] = True

    # The graph is built in the registered complete-prior frame.  Its local
    # radius follows sampling density, rather than a global Euclidean ball,
    # so neighbouring but distinct surfaces do not get unintentionally tied.
    query_k = min(int(neighbours) + 1, len(prior))
    distances, indices = cKDTree(prior).query(prior, k=query_k, workers=-1)
    if query_k == 1:  # covered above, kept for static safety
        return prior.copy(), anchors, {**anchor_info, "active": False, "reason": "too_few_prior_points"}
    local_scale = np.median(distances[:, 1:], axis=1)
    local_scale = np.maximum(local_scale, max(float(np.quantile(local_scale, .01)) * .25, 1e-9))
    source = np.repeat(np.arange(len(prior), dtype=np.int64), query_k - 1)
    target = indices[:, 1:].reshape(-1).astype(np.int64)
    edge_distance = distances[:, 1:].reshape(-1)
    # Retaining an undirected edge once avoids directional density bias.
    keep = source < target
    source, target, edge_distance = source[keep], target[keep], edge_distance[keep]
    scale = np.sqrt(local_scale[source] * local_scale[target])
    keep = edge_distance <= float(edge_ratio) * scale
    source, target, edge_distance, scale = (
        source[keep], target[keep], edge_distance[keep], scale[keep]
    )
    if len(source) == 0:
        return prior.copy(), anchors, {**anchor_info, "active": False, "reason": "empty_surface_graph"}
    weights = np.exp(-np.square(edge_distance / np.maximum(scale, 1e-12)))
    adjacency = sparse.coo_matrix(
        (np.concatenate((weights, weights)),
         (np.concatenate((source, target)), np.concatenate((target, source)))),
        shape=(len(prior), len(prior)), dtype=np.float64,
    ).tocsr()
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - adjacency
    components, labels = sparse.csgraph.connected_components(adjacency, directed=False)
    component_has_control = np.zeros(components, dtype=bool)
    component_has_control[labels[control_ids]] = True
    active_nodes = component_has_control[labels]
    unknown_ids = np.flatnonzero(~is_control)
    displacement = np.zeros_like(prior)
    displacement[control_ids] = control_offsets
    # The only target-view evidence is the saved camera.  Fixed PCA self-views
    # therefore contribute *zero-displacement preservation* rather than fake
    # negative evidence from an incomplete rotated partial scan.  Excluding a
    # metric neighbourhood of actual controls permits coherent local follow.
    control_tree = cKDTree(prior[control_ids])
    nearest_control, _ = control_tree.query(prior, k=1, workers=-1)
    protection_radius = float(protection_exclusion_ratio) * max(
        float(np.linalg.norm(np.ptp(prior, axis=0))), 1e-9,
    )
    protection_visible = _orthographic_prior_protection(
        prior, views=int(prior_protection_views),
    )
    protection_mask = protection_visible & ~is_control & (nearest_control > protection_radius)
    if len(unknown_ids):
        # Dirichlet harmonic extension with a weak screened term:
        #  min sum_ij w_ij ||d_i-d_j||^2 + screening * sum_i ||d_i||^2,
        #  subject to d_control = measured partial offset.
        unknown_laplacian = laplacian[unknown_ids][:, unknown_ids].tocsr()
        diagonal = (
            float(screening)
            + float(prior_protection_weight) * protection_mask[unknown_ids]
        )
        system = unknown_laplacian + sparse.diags(diagonal, format="csr")
        coupling = laplacian[unknown_ids][:, control_ids]
        rhs = -(coupling @ control_offsets)
        solution, cg_status = _solve_cg(
            system, np.asarray(rhs), tolerance=float(cg_tolerance),
            max_iterations=int(cg_max_iterations),
        )
        # Controls remain exact.  Only non-controls may receive a continuous
        # distance-monotone gain, allowing remote structure to follow a local
        # edit more strongly without loosening any observed correspondence.
        gain_radius = float(remote_gain_radius_ratio) * max(
            float(np.linalg.norm(np.ptp(prior, axis=0))), 1e-9,
        )
        gain = 1. + (float(remote_gain) - 1.) * (1. - np.exp(-nearest_control[unknown_ids] / gain_radius))
        remote_solution = solution * gain[:, None]
        displacement[unknown_ids] = _clip_vectors(
            remote_solution,
            float(max_displacement) * float(remote_displacement_cap_multiplier),
        )
    else:
        cg_status = [0, 0, 0]

    magnitude = np.linalg.norm(displacement, axis=1)
    edited = prior + displacement
    return edited, anchors, {
        **anchor_info,
        "active": True,
        "controls": int(len(control_ids)),
        "max_displacement": float(max_displacement),
        "neighbours": int(query_k - 1),
        "edge_ratio": float(edge_ratio),
        "screening": float(screening),
        "prior_protection_views": int(prior_protection_views),
        "prior_protection_weight": float(prior_protection_weight),
        "protection_exclusion_radius": float(protection_radius),
        "prior_protection_gaussians": int(protection_mask.sum()),
        "remote_gain": float(remote_gain),
        "remote_gain_radius": float(gain_radius) if len(unknown_ids) else 0.,
        "remote_displacement_cap": float(max_displacement) * float(remote_displacement_cap_multiplier),
        "graph_edges": int(len(source)),
        "graph_components": int(components),
        "control_components": int(component_has_control.sum()),
        "graph_reachable_prior_gaussians": int(active_nodes.sum()),
        "edited_prior_gaussians": int((magnitude > 1e-12).sum()),
        "untouched_prior_gaussians": int((magnitude <= 1e-12).sum()),
        "mean_displacement": float(magnitude.mean()),
        "p95_displacement": float(np.quantile(magnitude, .95)),
        "max_observed_displacement": float(magnitude.max(initial=0.)),
        "mean_noncontrol_displacement": float(magnitude[~is_control].mean()) if np.any(~is_control) else 0.,
        "cg_status": cg_status,
        "field": "boundary_conditioned_surface_graph_with_multiview_prior_protection",
    }


def compact_anchor_displacement(
    prior: np.ndarray,
    partial: np.ndarray,
    pixel_pairs: np.ndarray,
    *,
    max_anchor_residual: float,
    support_radius: float,
    max_displacement: float,
    neighbours: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Condition editable Gaussian centres on hard anchors with compact support.

    The input pairs are mutually visible saved-view matches.  Their observed
    point-to-prior offsets are handles of a compact Wendland field over the
    *registered prior* coordinates.  Hence only the observed neighbourhood is
    edited, no raw partial union is formed, and all remote complete-prior
    Gaussians receive exactly zero displacement.
    """
    prior, partial = np.asarray(prior, dtype=np.float64), np.asarray(partial, dtype=np.float64)
    if prior.ndim != 2 or partial.ndim != 2 or prior.shape[1] != 3 or partial.shape[1] != 3:
        raise ValueError("prior and partial must be (N, 3)")
    if min(max_anchor_residual, support_radius, max_displacement) <= 0. or neighbours <= 0:
        raise ValueError("all metric bounds and neighbours must be positive")
    anchors, anchor_info = collision_free_anchor_pairs(
        pixel_pairs, partial, prior, max_residual=float(max_anchor_residual),
    )
    if len(anchors) < 6:
        return prior.copy(), anchors, {**anchor_info, "active": False,
                                       "reason": "insufficient_collision_free_anchors"}
    partial_ids, prior_ids = anchors[:, 0], anchors[:, 1]
    controls, offsets = prior[prior_ids], partial[partial_ids] - prior[prior_ids]
    tree = cKDTree(controls)
    distance, control_ids = tree.query(prior, k=min(int(neighbours), len(controls)), workers=-1)
    if distance.ndim == 1:
        distance, control_ids = distance[:, None], control_ids[:, None]
    normalized = distance / float(support_radius)
    # C2 Wendland kernel: smooth at the support boundary and exactly zero
    # outside, which protects unobserved complete-prior geometry.
    inside = normalized < 1.
    weights = np.where(inside, (1. - normalized) ** 4 * (4. * normalized + 1.), 0.)
    weight_sum = weights.sum(axis=1, keepdims=True)
    displacement = (weights[..., None] * offsets[control_ids]).sum(axis=1)
    displacement /= np.maximum(weight_sum, 1e-12)
    magnitude = np.linalg.norm(displacement, axis=1, keepdims=True)
    displacement *= np.minimum(1., float(max_displacement) / np.maximum(magnitude, 1e-12))
    edited = prior + displacement
    return edited, anchors, {
        **anchor_info, "active": True, "controls": int(len(controls)),
        "support_radius": float(support_radius), "max_displacement": float(max_displacement),
        "neighbours": int(min(int(neighbours), len(controls))),
        "edited_prior_gaussians": int((weight_sum[:, 0] > 0.).sum()),
        "untouched_prior_gaussians": int((weight_sum[:, 0] == 0.).sum()),
        "mean_displacement": float(np.linalg.norm(displacement, axis=1).mean()),
        "p95_displacement": float(np.quantile(np.linalg.norm(displacement, axis=1), .95)),
        "max_observed_displacement": float(np.linalg.norm(displacement, axis=1).max(initial=0.)),
        "field": "compact_wendland_source_space_gaussian_mean_displacement",
    }
