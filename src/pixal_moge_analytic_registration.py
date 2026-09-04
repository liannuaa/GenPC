"""Camera-anchored local Pixal-to-MoGe registration.

Pixal exposes the FOV and front-camera distance used to condition generation.
Its exported GLB has a fixed axis conversion.  Therefore Pixal and a MoGe map
inferred from the exact ``pixal3d_input.png`` are not treated as unrelated
point clouds: the camera conversion supplies the only global initialization,
and optimisation is restricted to a small residual Sim(3) in camera space.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.ray_consistent_registration import apply_transform
from src.zbuffer import zbuffer_depth_with_indices


def pixal_export_to_camera(distance: float) -> np.ndarray:
    """Map the fixed exported Pixal axes to its OpenCV front-camera frame.

    ``o_voxel.to_glb`` first applies ``B(x,y,z)=(x,z,-y)`` for GLB
    compatibility; the Pixal runner then applies its fixed export transform
    ``E``.  Their composition is ``E B = diag(-1,1,-1)``.  Before these two
    exports, Pixal's official front-camera code gives ``(x,-y,distance-z)``.
    Substitution with ``p=diag(-1,1,-1) p_export`` yields
    ``(-x_export,-y_export,distance+z_export)``.  Positive z is the same
    forward-depth convention as MoGe's ``points`` output.
    """
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("Pixal camera distance must be positive")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    transform[:3, 3] = np.array([0.0, 0.0, float(distance)])
    return transform


def analytic_pixal_to_moge_initial(prior: np.ndarray, moge: np.ndarray, distance: float) -> np.ndarray:
    """Apply the known camera transform and only resolve MoGe's depth gauge."""
    prior_camera = apply_transform(prior, pixal_export_to_camera(distance))
    source_depth = prior_camera[:, 2]
    target_depth = np.asarray(moge, dtype=np.float64)[:, 2]
    source_depth = source_depth[np.isfinite(source_depth) & (source_depth > 1e-6)]
    target_depth = target_depth[np.isfinite(target_depth) & (target_depth > 1e-6)]
    if len(source_depth) < 96 or len(target_depth) < 96:
        raise ValueError("insufficient positive camera depths for analytic scale")
    # Global monocular scale is free. Scaling *camera-frame* coordinates keeps
    # the known image projection fixed and maps Pixal's conditioned distance to
    # the MoGe depth gauge without inventing an arbitrary 3-D translation.
    scale = float(np.median(target_depth) / np.median(source_depth))
    scale = float(np.clip(scale, 0.25, 4.0))
    transform = pixal_export_to_camera(distance)
    transform[:3, :] *= scale
    return transform


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a binary image without circular wraparound."""
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        result[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = mask[src_y0:src_y1, src_x0:src_x1]
    return result


def visible_mask_translation_step(
    moge: np.ndarray,
    prior_camera: np.ndarray,
    projector,
    *,
    max_depth_ratio: float = .04,
) -> tuple[np.ndarray, dict]:
    """Convert a same-camera silhouette-correlation peak into an x/y shift.

    This is a deterministic camera-plane correction after the known Pixal
    transform.  It is intentionally disabled for large offsets, which belong
    to a wrong global coordinate convention rather than residual alignment.
    """
    target_depth, target_mask, _ = _mask_and_depth(moge, projector)
    _, prior_mask, _ = _mask_and_depth(prior_camera, projector)
    target_y, target_x = np.where(target_mask)
    prior_y, prior_x = np.where(prior_mask)
    step = np.eye(4, dtype=np.float64)
    if len(target_x) < 96 or len(prior_x) < 96:
        return step, {"accepted": False, "reason": "insufficient_visible_points"}
    # Coarse-to-fine discrete correlation is resilient to asymmetric objects:
    # it aligns the actual projected support, not only its centroid.
    shift = np.zeros(2, dtype=np.int64)
    target_count = max(int(target_mask.sum()), 1)
    for radius in (12, 4, 1):
        candidates = []
        for dx in (-radius, 0, radius):
            for dy in (-radius, 0, radius):
                candidate = _shift_mask(prior_mask, int(shift[0] + dx), int(shift[1] + dy))
                intersection = int((candidate & target_mask).sum())
                union = int((candidate | target_mask).sum())
                # IoU is primary; a small coverage tie-break avoids a trivial
                # shift of a too-small prediction into a target subregion.
                candidates.append((intersection / max(union, 1) + .02 * intersection / target_count,
                                   int(shift[0] + dx), int(shift[1] + dy)))
        _, best_x, best_y = max(candidates, key=lambda item: item[0])
        shift[:] = (best_x, best_y)
    pixel_delta = shift.astype(np.float64)
    diagonal = math.hypot(*projector.image_shape)
    if np.linalg.norm(pixel_delta) > .08 * diagonal:
        return step, {"accepted": False, "reason": "centroid_offset_too_large",
                      "pixel_delta": pixel_delta.tolist()}
    depth = float(np.median(target_depth[target_mask]))
    fx, fy = float(projector.intrinsic[0, 0]), float(projector.intrinsic[1, 1])
    translation = np.array([pixel_delta[0] * depth / fx, pixel_delta[1] * depth / fy, 0.0])
    cap = float(max_depth_ratio) * depth
    translation = np.clip(translation, -cap, cap)
    step[:3, 3] = translation
    return step, {"accepted": True, "pixel_delta": pixel_delta.tolist(),
                  "translation": translation.tolist(), "reference_depth": depth}


def visible_ray_depth_scale_step(
    moge: np.ndarray,
    prior_camera: np.ndarray,
    projector,
    *,
    max_depth_ratio: float = .12,
) -> tuple[np.ndarray, dict]:
    """Estimate a camera-ray scale from same-pixel visible 3-D surfaces.

    A monocular MoGe map has a global depth gauge.  Scaling all three camera
    coordinates corrects that gauge while preserving the already-aligned
    projection ``(x/z, y/z)``; a pure z translation would change silhouette
    scale under perspective projection.
    """
    target_depth, target_mask, _ = _mask_and_depth(moge, projector)
    prior_depth, prior_mask, _ = _mask_and_depth(prior_camera, projector)
    shared = target_mask & prior_mask
    step = np.eye(4, dtype=np.float64)
    if int(shared.sum()) < 96:
        return step, {"accepted": False, "reason": "insufficient_shared_visible_surface"}
    ratio = target_depth[shared] / np.maximum(prior_depth[shared], 1e-8)
    lo, hi = np.quantile(ratio, (.10, .90))
    stable = ratio[(ratio >= lo) & (ratio <= hi)]
    raw_scale = float(np.median(stable))
    reference_depth = float(np.median(target_depth[target_mask]))
    scale = float(np.clip(raw_scale, 1.0 - max_depth_ratio, 1.0 + max_depth_ratio))
    step[:3, :3] *= scale
    return step, {
        "accepted": True, "shared_visible_pixels": int(shared.sum()),
        "raw_median_target_over_prior_z": raw_scale,
        "ray_depth_scale": scale, "reference_depth": reference_depth,
        "trimmed_ratio_range": [float(lo), float(hi)],
    }


def _mask_and_depth(points: np.ndarray, projector, *, splat_radius: int = 1):
    uv, depth = projector.project(points)
    rendered, mask, indices = zbuffer_depth_with_indices(
        uv, depth, projector.image_shape, splat_radius=splat_radius
    )
    return rendered, mask, indices


def pixal_moge_render_score(
    moge: np.ndarray,
    prior_camera: np.ndarray,
    projector,
    *,
    moge_colors: np.ndarray | None = None,
    prior_colors: np.ndarray | None = None,
) -> dict:
    """Visible silhouette, depth, and depth-boundary residual in the same view."""
    target_depth, target_mask, target_indices = _mask_and_depth(moge, projector)
    prior_depth, prior_mask, prior_indices = _mask_and_depth(prior_camera, projector)
    intersection = target_mask & prior_mask
    union = target_mask | prior_mask
    target_count = max(int(target_mask.sum()), 1)
    prior_count = max(int(prior_mask.sum()), 1)
    iou = float(intersection.sum() / max(int(union.sum()), 1))
    coverage = float(intersection.sum() / target_count)
    leakage = float((prior_mask & ~target_mask).sum() / prior_count)
    if np.any(intersection):
        log_delta = np.abs(np.log(np.maximum(prior_depth[intersection], 1e-8)
                                  / np.maximum(target_depth[intersection], 1e-8)))
        depth = float(np.median(log_delta))
    else:
        depth = float("inf")
    kernel = np.ones((3, 3), dtype=np.uint8)
    target_edge = cv2.morphologyEx(target_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    prior_edge = cv2.morphologyEx(prior_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    diagonal = math.hypot(*projector.image_shape)
    if target_edge.any() and prior_edge.any():
        target_dt = cv2.distanceTransform((~target_edge).astype(np.uint8), cv2.DIST_L2, 3)
        prior_dt = cv2.distanceTransform((~prior_edge).astype(np.uint8), cv2.DIST_L2, 3)
        boundary = float((target_dt[prior_edge].mean() + prior_dt[target_edge].mean()) * .5 / max(diagonal, 1.0))
    else:
        boundary = 1.0
    color = None
    if moge_colors is not None and prior_colors is not None and np.any(intersection):
        moge_colors = np.asarray(moge_colors, dtype=np.float64)
        prior_colors = np.asarray(prior_colors, dtype=np.float64)
        if len(moge_colors) == len(moge) and len(prior_colors) == len(prior_camera):
            lhs, rhs = moge_colors[target_indices[intersection]], prior_colors[prior_indices[intersection]]
            color = float(np.median(np.linalg.norm(lhs - rhs, axis=1)))
    depth_term = min(depth / .08, 2.0) if np.isfinite(depth) else 2.0
    boundary_term = min(boundary / .04, 2.0)
    color_term = min(float(color) / .30, 2.0) if color is not None else 0.0
    objective = float(.48 * (1.0 - iou) + .12 * (1.0 - coverage) + .11 * leakage
                      + .17 * depth_term + .07 * boundary_term + .05 * color_term)
    return {
        "objective": objective, "iou": iou, "coverage": coverage, "leakage": leakage,
        "log_depth_median": depth, "depth_boundary": boundary, "rgb_median": color,
    }


def _step_transform(*, scale: float = 1.0, rotvec: np.ndarray | None = None,
                    translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    rotation = np.eye(3) if rotvec is None else Rotation.from_rotvec(rotvec).as_matrix()
    transform[:3, :3] = float(scale) * rotation
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def local_pixal_moge_refine(
    prior: np.ndarray,
    moge: np.ndarray,
    projector,
    initial: np.ndarray,
    *,
    moge_colors: np.ndarray | None = None,
    prior_colors: np.ndarray | None = None,
    levels: tuple[tuple[float, float, float], ...] = ((.025, 2.0, .025), (.010, .8, .010), (.004, .3, .004)),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Small camera-frame Sim(3) coordinate descent; no orientation search."""
    current = apply_transform(prior, initial)
    total = np.asarray(initial, dtype=np.float64).copy()
    initial_score = pixal_moge_render_score(
        moge, current, projector, moge_colors=moge_colors, prior_colors=prior_colors
    )
    trace = []
    reference_depth = float(np.median(moge[np.asarray(moge)[:, 2] > 1e-6, 2]))
    for scale_delta, degrees, translation_ratio in levels:
        translation = float(translation_ratio) * reference_depth
        proposals = [("identity", np.eye(4, dtype=np.float64))]
        for multiplier in (1.0 - scale_delta, 1.0 + scale_delta):
            proposals.append((f"scale_{multiplier:.4f}", _step_transform(scale=multiplier)))
        radians = math.radians(float(degrees))
        for axis, label in enumerate(("x", "y", "z")):
            for sign in (-1., 1.):
                vec = np.zeros(3); vec[axis] = sign * radians
                proposals.append((f"rot_{label}_{sign:+.0f}", _step_transform(rotvec=vec)))
                shift = np.zeros(3); shift[axis] = sign * translation
                proposals.append((f"trans_{label}_{sign:+.0f}", _step_transform(translation=shift)))
        scored = []
        for action, step in proposals:
            moved = apply_transform(current, step)
            scored.append((pixal_moge_render_score(
                moge, moved, projector, moge_colors=moge_colors, prior_colors=prior_colors
            ), action, step, moved))
        score, action, step, current = min(scored, key=lambda item: item[0]["objective"])
        total = step @ total
        trace.append({"level": {"scale_delta": scale_delta, "rotation_deg": degrees,
                                 "translation_depth_ratio": translation_ratio},
                      "action": action, "score": score})
    return current, total, {"before": initial_score,
                             "after": pixal_moge_render_score(
                                 moge, current, projector, moge_colors=moge_colors, prior_colors=prior_colors
                             ),
                             "trace": trace}


def _compact_partial_score(score: dict) -> dict:
    """Keep the scalar saved-view evidence of a partial registration score."""
    geometric, projection = score["geometric"], score["projection"]
    return {
        "objective": float(score["objective"]),
        "geometric_objective": float(geometric["objective"]),
        "pair_count": int(len(geometric["partial_ids"])),
        "projection": {key: float(value) for key, value in projection.items()},
    }


def local_pixal_partial_refine(
    prior_native_moge: np.ndarray,
    moge: np.ndarray,
    moge_projector,
    partial: np.ndarray,
    partial_projector,
    native_moge_to_partial: np.ndarray,
    *,
    partial_diagonal: float,
    moge_colors: np.ndarray | None = None,
    prior_colors: np.ndarray | None = None,
    levels: tuple[tuple[float, float, float], ...] = ((.010, .75, .010), (.004, .30, .004), (.002, .12, .002)),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tightly correct Camera-1/Camera-2 coupling error in Pixal's camera.

    This is deliberately the same local coordinate-descent family as
    :func:`local_pixal_moge_refine`: identity, small isotropic scale, small
    rotations, and camera-frame translations are considered at each level.
    The difference is only the score.  Camera-1 partial and Camera-2
    Pixal--MoGe evidence are fixed weighted terms of the same local objective;
    neither is used as an accept/reject gate.  It never redoes a global
    orientation search.
    """
    from src.bidirectional_cycle_registration import visible_score

    current = np.asarray(prior_native_moge, dtype=np.float64).copy()
    total = np.eye(4, dtype=np.float64)
    native_before = pixal_moge_render_score(
        moge, current, moge_projector, moge_colors=moge_colors, prior_colors=prior_colors
    )
    partial_before_raw = visible_score(
        partial, apply_transform(current, native_moge_to_partial), partial_projector,
        partial_diagonal, pixel_radius=5.,
    )
    partial_before = _compact_partial_score(partial_before_raw)
    native_reference = max(float(native_before["objective"]), 1e-8)
    partial_reference = max(float(partial_before["objective"]), 1e-8)
    reference_depth = float(np.median(np.asarray(moge)[np.asarray(moge)[:, 2] > 1e-6, 2]))
    trace = []
    for scale_delta, degrees, translation_ratio in levels:
        translation = float(translation_ratio) * reference_depth
        proposals = [("identity", np.eye(4, dtype=np.float64))]
        for multiplier in (1.0 - scale_delta, 1.0 + scale_delta):
            proposals.append((f"scale_{multiplier:.4f}", _step_transform(scale=multiplier)))
        radians = math.radians(float(degrees))
        for axis, label in enumerate(("x", "y", "z")):
            for sign in (-1., 1.):
                vector = np.zeros(3); vector[axis] = sign * radians
                proposals.append((f"rot_{label}_{sign:+.0f}", _step_transform(rotvec=vector)))
                shift = np.zeros(3); shift[axis] = sign * translation
                proposals.append((f"trans_{label}_{sign:+.0f}", _step_transform(translation=shift)))
        candidates = []
        for action, step in proposals:
            moved = apply_transform(current, step)
            native = pixal_moge_render_score(
                moge, moved, moge_projector, moge_colors=moge_colors, prior_colors=prior_colors
            )
            partial_raw = visible_score(
                partial, apply_transform(moved, native_moge_to_partial), partial_projector,
                partial_diagonal, pixel_radius=5.,
            )
            partial_score = _compact_partial_score(partial_raw)
            # Camera-1 evidence remains primary, while the native rendering
            # term keeps the residual in the Pixal/MoGe coordinate basin.
            joint = float(.75 * partial_score["objective"] / partial_reference
                          + .25 * native["objective"] / native_reference)
            candidates.append((joint, action, step, moved, native, partial_score))
        joint, action, step, current, native_score, partial_score = min(
            candidates, key=lambda item: item[0]
        )
        total = step @ total
        trace.append({
            "level": {"scale_delta": float(scale_delta), "rotation_deg": float(degrees),
                      "translation_depth_ratio": float(translation_ratio)},
            "action": action, "joint_objective": joint,
            "native": native_score, "partial": partial_score,
        })
    native_after = pixal_moge_render_score(
        moge, current, moge_projector, moge_colors=moge_colors, prior_colors=prior_colors
    )
    partial_after = _compact_partial_score(visible_score(
        partial, apply_transform(current, native_moge_to_partial), partial_projector,
        partial_diagonal, pixel_radius=5.,
    ))
    return current, total, {
        "method": "pixal_moge_style_local_camera_residual_for_two_camera_partial_bridge",
        "native_before": native_before, "native_after": native_after,
        "partial_before": partial_before, "partial_after": partial_after,
        "levels": [{"scale_delta": item[0], "rotation_deg": item[1],
                    "translation_depth_ratio": item[2]} for item in levels],
        "trace": trace,
    }
