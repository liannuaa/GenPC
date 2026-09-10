"""Analytic-view initialization from a selected TRELLIS canonical camera."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.bidirectional_cycle_registration import prepare_visible_target, visible_score


def _camera_rotation(projector) -> np.ndarray:
    rotation = projector.camera.R.detach().float().cpu().numpy().reshape(-1, 3, 3)[0]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("Camera-1 extrinsic rotation is not orthonormal")
    return rotation.astype(np.float64)


def initialise_from_selected_camera1_view(
    source: np.ndarray,
    partial: np.ndarray,
    projector,
    selection_json: Path,
    *,
    sample_points: int = 12_000,
    view_shortlist: int = 12,
    seed: int = 6145,
) -> tuple[np.ndarray, dict]:
    """Map a selected TRELLIS camera to Camera-1 and fit scale/translation.

    Image evidence proposes a small discrete rotation shortlist.  Isotropic
    scale and translation are fitted for every proposal, after which partial
    silhouette/depth/visible-3D evidence chooses the basin.  This avoids both
    unconstrained Sim(3) search and reliance on category-specific orientation.
    """
    source = np.asarray(source, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    selection = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    target_camera_basis = _camera_rotation(projector).T
    source_centre = np.median(source, axis=0)
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(len(source), min(sample_points, len(source)), replace=False)
    target_uv, target_depth = projector.project(partial)
    target_uv_q = np.quantile(target_uv, (0.02, 0.5, 0.98), axis=0)
    target_depth_q = np.quantile(target_depth, (0.02, 0.5, 0.98))
    partial_centre = np.median(partial, axis=0)
    centred_sample = source[sample_ids] - source_centre
    source_radius = np.quantile(
        np.linalg.norm(centred_sample - np.median(centred_sample, axis=0), axis=1), 0.75,
    )
    target_radius = np.quantile(np.linalg.norm(partial - partial_centre, axis=1), 0.75)
    initial_scale = float(target_radius / max(source_radius, 1e-8))
    image_norm = float(max(projector.image_shape))
    depth_norm = max(float(target_depth_q[2] - target_depth_q[0]), 1e-4)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-4)
    target_cache = prepare_visible_target(partial, projector)
    proposals = selection.get("top", [{"index": selection.get("selected_index", 0), **selection["selected"]}])
    proposals = proposals[: max(1, int(view_shortlist))]
    records = []
    for proposal in proposals:
        source_camera_basis = np.asarray(proposal["camera_pose"], dtype=np.float64)[:3, :3]
        rotation = target_camera_basis @ source_camera_basis.T
        u, _, vh = np.linalg.svd(rotation)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        oriented = (source - source_centre) @ rotation.T
        sampled = oriented[sample_ids]
        initial_translation = partial_centre - initial_scale * np.median(sampled, axis=0)

        def residual(parameters: np.ndarray) -> np.ndarray:
            scale = float(np.exp(parameters[0]))
            transformed = sampled * scale + parameters[1:4]
            uv, depth = projector.project(transformed)
            uv_q = np.quantile(uv, (0.02, 0.5, 0.98), axis=0)
            depth_q = np.quantile(depth, (0.02, 0.5, 0.98))
            return np.concatenate([
                ((uv_q - target_uv_q) / image_norm).reshape(-1),
                0.45 * (depth_q - target_depth_q) / depth_norm,
            ])

        x0 = np.concatenate([[np.log(initial_scale)], initial_translation])
        lower = np.concatenate([[np.log(initial_scale) - 1.5], initial_translation - 2.0 * diagonal])
        upper = np.concatenate([[np.log(initial_scale) + 1.5], initial_translation + 2.0 * diagonal])
        fit = least_squares(
            residual, x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.02,
            max_nfev=50, xtol=1e-7, ftol=1e-7, gtol=1e-7,
        )
        scale = float(np.exp(fit.x[0]))
        translation = fit.x[1:4]
        candidate = oriented * scale + translation
        score = visible_score(
            partial, candidate, projector, diagonal,
            pixel_radius=5.0, target_cache=target_cache,
        )
        records.append({
            "proposal_index": int(proposal.get("index", len(records))),
            "yaw_degrees": float(proposal.get("yaw_degrees", 0.0)),
            "pitch_degrees": float(proposal.get("pitch_degrees", 0.0)),
            "roll_degrees": float(proposal.get("roll_degrees", 0.0)),
            "image_joint_score": float(proposal.get("joint_score", 0.0)),
            "rotation": rotation,
            "scale": scale,
            "translation": translation,
            "fit_cost": float(fit.cost),
            "fit_optimality": float(fit.optimality),
            "fit_evaluations": int(fit.nfev),
            "visible_objective": float(score["objective"]),
            "visible_geometric": float(score["geometric"]["objective"]),
            "projection": {key: float(value) for key, value in score["projection"].items()},
        })
    image_values = np.asarray([item["image_joint_score"] for item in records])
    image_penalty = (image_values.max() - image_values) / max(float(np.ptp(image_values)), 1e-8)
    for item, penalty in zip(records, image_penalty):
        item["selection_objective"] = float(item["visible_objective"] + 0.06 * penalty)
    selected = min(records, key=lambda item: item["selection_objective"])
    rotation = selected["rotation"]
    scale = float(selected["scale"])
    translation = np.asarray(selected["translation"])
    result = (source - source_centre) @ rotation.T * scale + translation
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation - scale * rotation @ source_centre
    return result, {
        "method": "image_shortlist_camera1_visible_sim3_initialization",
        "ground_truth_used": False,
        "selection_json": str(Path(selection_json).resolve()),
        "rotation": rotation.tolist(),
        "scale": scale,
        "translation": translation.tolist(),
        "transform": transform.tolist(),
        "view_shortlist": int(len(records)),
        "selected_proposal_index": int(selected["proposal_index"]),
        "selected_view": {
            "yaw_degrees": selected["yaw_degrees"],
            "pitch_degrees": selected["pitch_degrees"],
            "roll_degrees": selected["roll_degrees"],
        },
        "least_squares_cost": float(selected["fit_cost"]),
        "least_squares_optimality": float(selected["fit_optimality"]),
        "least_squares_evaluations": int(selected["fit_evaluations"]),
        "target_uv_quantiles": target_uv_q.tolist(),
        "target_depth_quantiles": target_depth_q.tolist(),
        "candidates": [
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in item.items()
                if key != "rotation"
            }
            for item in records
        ],
    }
