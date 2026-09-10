"""No-GT complete-prior bridge for regenerated image-to-3D assets."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from src.bidirectional_cycle_registration import prepare_visible_target, visible_score


def _pca_basis(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centred.T))
    return vectors[:, np.argsort(values)[::-1]]


def _proper_signed_permutations() -> list[np.ndarray]:
    candidates = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.0:
                candidates.append(matrix)
    return candidates


def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, min(len(values), int(round(len(values) * fraction))))
    return float(np.mean(np.partition(values, count - 1)[:count]))


def bridge_regenerated_complete_prior(
    source: np.ndarray,
    reference: np.ndarray,
    partial: np.ndarray,
    projector,
    *,
    seed: int = 42,
    sample_points: int = 12_000,
    trim_fraction: float = 0.80,
    visible_shortlist: int = 8,
    full_score_slack: float = 1.15,
) -> tuple[np.ndarray, dict]:
    """Place a regenerated complete prior in a registered-prior basin.

    Full-to-full geometry resolves the canonical axis ambiguity.  Camera-1
    partial evidence then chooses among geometrically near-equivalent basins.
    The latter is intentionally restricted to a shortlist within a fixed
    factor of the best full-shape score, preventing the incomplete scan from
    rotating a complete asset into an implausible but locally convenient pose.
    """
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    partial = np.asarray(partial, dtype=np.float64)
    if min(len(source), len(reference), len(partial)) < 64:
        raise ValueError("source, reference and partial need at least 64 points")
    if not 0.5 <= trim_fraction <= 1.0:
        raise ValueError("trim_fraction must lie in [0.5, 1]")
    if visible_shortlist < 1 or full_score_slack < 1.0:
        raise ValueError("shortlist must be positive and score slack at least one")

    rng = np.random.default_rng(seed)
    source_sample = source[rng.choice(len(source), min(sample_points, len(source)), replace=False)]
    reference_sample = reference[
        rng.choice(len(reference), min(sample_points, len(reference)), replace=False)
    ]
    source_centre = np.median(source_sample, axis=0)
    reference_centre = np.median(reference_sample, axis=0)
    source_radius = float(np.quantile(np.linalg.norm(source_sample - source_centre, axis=1), 0.75))
    reference_radius = float(np.quantile(np.linalg.norm(reference_sample - reference_centre, axis=1), 0.75))
    scale = reference_radius / max(source_radius, 1e-12)
    source_basis = _pca_basis(source_sample)
    reference_basis = _pca_basis(reference_sample)
    reference_tree = cKDTree(reference_sample)
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    target_cache = prepare_visible_target(partial, projector)

    records = []
    for index, permutation in enumerate(_proper_signed_permutations()):
        rotation = reference_basis @ permutation @ source_basis.T
        transformed_sample = (source_sample - source_centre) @ rotation.T * scale + reference_centre
        forward = reference_tree.query(transformed_sample, k=1, workers=-1)[0]
        reverse = cKDTree(transformed_sample).query(reference_sample, k=1, workers=-1)[0]
        full_score = _trimmed_mean(forward, trim_fraction) + _trimmed_mean(reverse, trim_fraction)
        records.append({
            "candidate": index,
            "rotation": rotation,
            "full_score": float(full_score / diagonal),
        })
    records.sort(key=lambda item: item["full_score"])
    shortlist = records[: min(visible_shortlist, len(records))]
    best_full = float(shortlist[0]["full_score"])
    eligible = [item for item in shortlist if item["full_score"] <= best_full * full_score_slack]
    for item in shortlist:
        transformed = (source - source_centre) @ item["rotation"].T * scale + reference_centre
        score = visible_score(
            partial, transformed, projector, diagonal,
            pixel_radius=5.0, target_cache=target_cache,
        )
        item["visible_score"] = float(score["objective"])
        item["visible_geometric_score"] = float(score["geometric"]["objective"])
        item["projection"] = {key: float(value) for key, value in score["projection"].items()}
    selected = min(eligible, key=lambda item: item["visible_score"])
    result = (source - source_centre) @ selected["rotation"].T * scale + reference_centre
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * selected["rotation"]
    transform[:3, 3] = reference_centre - scale * selected["rotation"] @ source_centre
    return result, {
        "method": "full_prior_pca_basin_then_camera1_visible_selection",
        "strict_zero_shot": True,
        "ground_truth_cd_emd_used": False,
        "sample_points": int(min(sample_points, len(source), len(reference))),
        "trim_fraction": float(trim_fraction),
        "visible_shortlist": int(visible_shortlist),
        "full_score_slack": float(full_score_slack),
        "isotropic_scale": scale,
        "source_centre": source_centre.tolist(),
        "reference_centre": reference_centre.tolist(),
        "selected_candidate": int(selected["candidate"]),
        "selected_transform": transform.tolist(),
        "candidates": [
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in item.items()
            }
            for item in shortlist
        ],
    }
