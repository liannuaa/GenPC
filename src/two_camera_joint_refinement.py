"""Closed-loop residual Sim(3) for Pixal, native MoGe, and a partial scan.

Sequential composition ``T_MP @ T_QM`` compounds the small Camera-1/Camera-2
bridge error with the small Pixal/MoGe rendering error.  This module keeps
those two calibrated transforms as the initialization and optimizes only two
tiny residual proper Sim(3)s in a three-factor graph:

``Pixal -- Camera 2 render -- MoGe -- pixel bridge -- partial``
``Pixal ------------------------ Camera 1 2D+3D ------------------- partial``

There is intentionally no PCA/orientation hypothesis, proposal gate, or
GT-dependent model selection.  Native MoGe, bridge, and Camera-1 evidence are
joint terms of one fixed local objective, not accept/reject constraints.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from src.bidirectional_cycle_registration import visible_score
from src.indexed_pixel_sim3 import unique_pixel_matches
from src.pixal_moge_analytic_registration import pixal_moge_render_score
from src.ray_consistent_registration import apply_transform


def compose_two_camera_transform(
    pixal_to_native_moge: np.ndarray,
    native_moge_to_partial: np.ndarray,
    pixal_native_residual: np.ndarray | None = None,
    moge_partial_residual: np.ndarray | None = None,
) -> np.ndarray:
    """Compose the two calibrated edges and their left-multiplied residuals."""
    q_to_m = np.asarray(pixal_to_native_moge, dtype=np.float64)
    m_to_p = np.asarray(native_moge_to_partial, dtype=np.float64)
    delta_q = np.eye(4, dtype=np.float64) if pixal_native_residual is None else np.asarray(pixal_native_residual, dtype=np.float64)
    delta_m = np.eye(4, dtype=np.float64) if moge_partial_residual is None else np.asarray(moge_partial_residual, dtype=np.float64)
    if any(matrix.shape != (4, 4) for matrix in (q_to_m, m_to_p, delta_q, delta_m)):
        raise ValueError("all two-camera transforms must be 4x4")
    return delta_m @ m_to_p @ delta_q @ q_to_m


def bridge_match_score(
    moge: np.ndarray,
    partial: np.ndarray,
    matches: np.ndarray,
    native_moge_to_partial: np.ndarray,
    *,
    diagonal: float,
) -> dict:
    """Robust geometric evidence for the fixed Camera-1/Camera-2 pixel bridge."""
    pairs = unique_pixel_matches(matches)
    m_ids, p_ids = pairs[:, 1].astype(np.int64), pairs[:, 0].astype(np.int64)
    aligned = apply_transform(np.asarray(moge, dtype=np.float64)[m_ids], native_moge_to_partial)
    residual = np.linalg.norm(aligned - np.asarray(partial, dtype=np.float64)[p_ids], axis=1)
    trim = float(np.quantile(residual, .75))
    kept = residual <= trim
    return {
        "pairs": int(len(residual)), "trimmed_mean": float(residual[kept].mean()),
        "median": float(np.median(residual)), "p90": float(np.quantile(residual, .90)),
        "normalized_trimmed_mean": float(residual[kept].mean() / max(float(diagonal), 1e-8)),
    }


def _compact_visible(score: dict) -> dict:
    return {
        "objective": float(score["objective"]),
        "geometric_objective": float(score["geometric"]["objective"]),
        "pairs": int(len(score["geometric"]["partial_ids"])),
        "projection": {key: float(value) for key, value in score["projection"].items()},
    }


def _step(*, scale: float, rotation_deg: float, axis: int | None,
          translation: np.ndarray | None, centre: np.ndarray) -> np.ndarray:
    """Build a residual Sim(3), rotating/scaling about the specified frame origin."""
    transform = np.eye(4, dtype=np.float64)
    rotation = np.eye(3) if axis is None else Rotation.from_rotvec(
        np.eye(3)[axis] * math.radians(float(rotation_deg))
    ).as_matrix()
    linear = float(scale) * rotation
    transform[:3, :3] = linear
    transform[:3, 3] = np.asarray(centre, dtype=np.float64) - linear @ np.asarray(centre, dtype=np.float64)
    if translation is not None:
        transform[:3, 3] += np.asarray(translation, dtype=np.float64)
    return transform


def residual_proposals(*, scale_delta: float, rotation_deg: float,
                       translation: float, centre: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The small coordinate-descent action set shared by both graph edges."""
    proposals = [("identity", np.eye(4, dtype=np.float64))]
    for value in (1. - float(scale_delta), 1. + float(scale_delta)):
        proposals.append((f"scale_{value:.4f}", _step(scale=value, rotation_deg=0., axis=None,
                                                         translation=None, centre=centre)))
    for axis, name in enumerate(("x", "y", "z")):
        for sign in (-1., 1.):
            proposals.append((f"rot_{name}_{sign:+.0f}", _step(
                scale=1., rotation_deg=sign * float(rotation_deg), axis=axis,
                translation=None, centre=centre)))
            shift = np.zeros(3); shift[axis] = sign * float(translation)
            proposals.append((f"trans_{name}_{sign:+.0f}", _step(
                scale=1., rotation_deg=0., axis=None, translation=shift, centre=centre)))
    return proposals


def joint_two_camera_refine(
    prior_native_moge: np.ndarray,
    moge: np.ndarray,
    partial: np.ndarray,
    matches: np.ndarray,
    native_projector,
    partial_projector,
    native_moge_to_partial: np.ndarray,
    *,
    partial_diagonal: float,
    bridge_moge: np.ndarray | None = None,
    bridge_partial: np.ndarray | None = None,
    levels: tuple[tuple[float, float, float], ...] = ((.008, .60, .008), (.003, .22, .003)),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Alternating local optimization of the two residual Sim(3) graph edges.

    ``prior_native_moge`` is already aligned to native MoGe.  The returned
    residuals are respectively in native MoGe and partial frames.  Every
    level applies the lowest joint-objective proposal from a fixed bounded
    Sim(3) lattice; full-resolution evidence is diagnostic only.
    """
    prior_native_moge, moge, partial = (np.asarray(value, dtype=np.float64)
                                        for value in (prior_native_moge, moge, partial))
    bridge_moge = moge if bridge_moge is None else np.asarray(bridge_moge, dtype=np.float64)
    bridge_partial = partial if bridge_partial is None else np.asarray(bridge_partial, dtype=np.float64)
    bridge = np.asarray(native_moge_to_partial, dtype=np.float64)
    delta_q, delta_m = np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)
    q_centre, p_centre = np.zeros(3, dtype=np.float64), np.median(partial, axis=0)
    native_before = pixal_moge_render_score(moge, prior_native_moge, native_projector)
    bridge_before = bridge_match_score(bridge_moge, bridge_partial, matches, bridge, diagonal=partial_diagonal)
    partial_before = _compact_visible(visible_score(
        partial, apply_transform(prior_native_moge, bridge), partial_projector,
        partial_diagonal, pixel_radius=5.,
    ))
    native_reference = max(native_before["objective"], 1e-8)
    bridge_reference = max(bridge_before["normalized_trimmed_mean"], 1e-8)
    partial_reference = max(partial_before["objective"], 1e-8)

    def score(candidate_q: np.ndarray, candidate_m: np.ndarray) -> tuple[dict, dict, dict, float]:
        native_points = apply_transform(prior_native_moge, candidate_q)
        native = pixal_moge_render_score(moge, native_points, native_projector)
        bridge_score = bridge_match_score(bridge_moge, bridge_partial, matches, candidate_m @ bridge,
                                          diagonal=partial_diagonal)
        partial_score = _compact_visible(visible_score(
            partial, apply_transform(native_points, candidate_m @ bridge), partial_projector,
            partial_diagonal, pixel_radius=5.,
        ))
        value = float(.35 * native["objective"] / native_reference
                      + .20 * bridge_score["normalized_trimmed_mean"] / bridge_reference
                      + .45 * partial_score["objective"] / partial_reference)
        return native, bridge_score, partial_score, value

    trace = []
    native_depth = float(np.median(moge[moge[:, 2] > 1e-6, 2]))
    for scale_delta, degrees, translation_ratio in levels:
        for edge in ("pixal_native", "moge_partial"):
            magnitude = (float(translation_ratio) * native_depth if edge == "pixal_native"
                         else float(translation_ratio) * float(partial_diagonal))
            centre = q_centre if edge == "pixal_native" else p_centre
            candidates = []
            for action, step in residual_proposals(
                scale_delta=scale_delta, rotation_deg=degrees, translation=magnitude, centre=centre
            ):
                candidate_q = step @ delta_q if edge == "pixal_native" else delta_q
                candidate_m = step @ delta_m if edge == "moge_partial" else delta_m
                native, bridge_score, partial_score, value = score(candidate_q, candidate_m)
                candidates.append((value, action, candidate_q, candidate_m,
                                   native, bridge_score, partial_score))
            value, action, delta_q, delta_m, native, bridge_score, partial_score = min(
                candidates, key=lambda item: item[0]
            )
            trace.append({
                "edge": edge, "level": {"scale_delta": float(scale_delta), "rotation_deg": float(degrees),
                                             "translation_ratio": float(translation_ratio)},
                "action": action, "joint_objective": value, "native": native,
                "bridge": bridge_score, "partial": partial_score,
            })
    native_after, bridge_after, partial_after, _ = score(delta_q, delta_m)
    return delta_q, delta_m, {
        "method": "two_camera_closed_loop_residual_sim3",
        "weights": {"pixal_native_moge": .35, "moge_partial_bridge": .20, "pixal_partial": .45},
        "levels": [{"scale_delta": x[0], "rotation_deg": x[1], "translation_ratio": x[2]} for x in levels],
        "native_before": native_before, "native_after": native_after,
        "bridge_before": bridge_before, "bridge_after": bridge_after,
        "partial_before": partial_before, "partial_after": partial_after,
        "trace": trace,
    }
