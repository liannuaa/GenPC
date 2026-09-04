"""Original 2D+3D bidirectional Sim(3) registration in a MoGe camera frame."""

from __future__ import annotations

import numpy as np
import torch

from scripts.run_pixal_pca_sim3_ttt_v2 import proper_pca_rotations
from src.bidirectional_consensus_registration import bidirectional_consensus_step
from src.bidirectional_cycle_registration import partial_to_prior_inverse_step, visible_score
from src.ray_consistent_registration import apply_transform


class _IdentityCamera:
    def transform(self, points: torch.Tensor) -> torch.Tensor:
        return points


class MoGeProjector:
    """Perspective projector for MoGe's native semantic-image camera frame."""

    def __init__(self, normalized_intrinsics: np.ndarray, image_shape: tuple[int, int], *, device: str = "cpu"):
        intrinsic = np.asarray(normalized_intrinsics, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError("MoGe intrinsics must be 3x3")
        self.image_shape = tuple(map(int, image_shape))
        height, width = self.image_shape
        self.intrinsic = intrinsic.copy()
        self.intrinsic[0] *= width
        self.intrinsic[1] *= height
        self.intrinsic[2] = (0., 0., 1.)
        self.device = str(device)
        self.camera = _IdentityCamera()
        # Compatibility fields are unused by the ray-score primitives but
        # make the camera contract explicit for future renderers.
        self.center_xy = np.array((width * .5, height * .5), dtype=np.float64)
        self.scale_xy = float(max(width, height))
        self.padding = 0.

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        depth = points[:, 2].copy()
        pixel = np.full((len(points), 2), np.nan, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1) & (depth > 1e-8)
        if np.any(valid):
            homogeneous = (self.intrinsic @ points[valid].T).T
            pixel[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
        return pixel, depth


def _diag(points: np.ndarray) -> float:
    return max(float(np.linalg.norm(np.ptp(np.asarray(points), axis=0))), 1e-8)


def _pca_initial(prior: np.ndarray, target: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    source_center, target_center = np.median(prior, axis=0), np.median(target, axis=0)
    scale = _diag(target) / _diag(prior @ rotation.T)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_center - scale * rotation @ source_center
    return transform


def register_prior_to_moge_2d3d(
    prior: np.ndarray,
    moge_points: np.ndarray,
    projector: MoGeProjector,
    *,
    shortlist: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run PCA hypotheses then the established bidirectional 2D+3D steps.

    MoGe is the *only* target in this function.  The caller must later convert
    to the partial frame and run a separate hard-partial refinement.
    """
    prior, moge_points = np.asarray(prior, dtype=np.float64), np.asarray(moge_points, dtype=np.float64)
    diagonal = _diag(moge_points)
    hypotheses = []
    for pca_id, item in enumerate(proper_pca_rotations(prior, moge_points)):
        transform = _pca_initial(prior, moge_points, item["rotation"])
        moved = apply_transform(prior, transform)
        score = visible_score(moge_points, moved, projector, diagonal, pixel_radius=8.)
        hypotheses.append({"pca_id": pca_id, "transform": transform, "points": moved, "score": score})
    hypotheses.sort(key=lambda item: item["score"]["objective"])
    candidates = []
    for hypothesis in hypotheses[:min(int(shortlist), len(hypotheses))]:
        current, total = hypothesis["points"].copy(), hypothesis["transform"].copy()
        trace = []
        for radius in (10., 6., 4.):
            direct, direct_step, direct_info = partial_to_prior_inverse_step(
                current, moge_points, projector, diagonal=diagonal, pixel_radius=radius,
                max_rotation_deg=4., scale_bounds=(.94, 1.06), max_translation_ratio=.04,
                min_pairs=96, return_best_candidate=False)
            consensus, consensus_step, consensus_info = bidirectional_consensus_step(
                current, moge_points, projector, diagonal=diagonal, pixel_radius=radius,
                max_rotation_deg=4., scale_bounds=(.94, 1.06), max_translation_ratio=.04,
                min_pairs=96, max_cycle_ratio=.04, return_best_candidate=False)
            choices = [("identity", current, np.eye(4), visible_score(moge_points, current, projector, diagonal, radius))]
            if direct_info["accepted"]:
                choices.append(("partial_to_prior_inverse", direct, direct_step, direct_info["after"]))
            if consensus_info["accepted"]:
                choices.append(("bidirectional_consensus", consensus, consensus_step, consensus_info["after"]))
            action, current, step, score = min(choices, key=lambda item: item[3]["objective"])
            total = step @ total
            trace.append({"pixel_radius": radius, "selected_action": action, "score": score,
                          "direct": direct_info, "consensus": consensus_info})
        candidates.append({"pca_id": hypothesis["pca_id"], "points": current, "transform": total,
                           "before": hypothesis["score"],
                           "after": visible_score(moge_points, current, projector, diagonal, pixel_radius=5.),
                           "trace": trace})
    selected = min(candidates, key=lambda item: item["after"]["objective"])
    return selected["points"], selected["transform"], {
        "method": "moge_camera_saved_view_2d3d_bidirectional_proper_sim3",
        "target_frame": "native MoGe semantic-image camera", "selected_pca_id": selected["pca_id"],
        "before": selected["before"], "after": selected["after"],
        "candidates": [{"pca_id": item["pca_id"], "before": item["before"], "after": item["after"],
                        "trace": item["trace"]} for item in candidates],
    }
