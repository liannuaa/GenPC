#!/usr/bin/env python3
"""Build three-view edit cards with one shared 3-D low-frequency update.

The cards decompose each partial-to-prior displacement into a shared smooth
3-D field plus a remaining local residual.  The same field is projected into
all views, so an image editor cannot independently re-apply global scale or
translation in each view.  No class label, part label, ground truth, or fixed
world axis enters the decomposition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pointcloud_io import load_points
from src.trellis_multiview_probe import (
    build_local_residual_prompt,
    build_shared_low_frequency_prompt,
)
from src.zbuffer import zbuffer_depth_with_indices


def _robust_diagonal(points: np.ndarray) -> float:
    lower, upper = np.quantile(points, (0.01, 0.99), axis=0)
    return max(float(np.linalg.norm(upper - lower)), 1e-8)


def _median_spacing(points: np.ndarray, limit: int = 16_384) -> float:
    if len(points) > limit:
        ids = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        points = points[ids]
    distance = cKDTree(points).query(points, k=2, workers=-1)[0][:, 1]
    positive = distance[np.isfinite(distance) & (distance > 0)]
    return float(np.median(positive)) if len(positive) else 1e-6


def _project(points: np.ndarray, pose: np.ndarray, fov: float, resolution: int):
    inverse = np.linalg.inv(pose)
    camera = points @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -camera[:, 2]
    focal = 0.5 * resolution / math.tan(math.radians(fov) * 0.5)
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = focal * camera[:, 0] / np.maximum(depth, 1e-8) + resolution * 0.5
    uv[:, 1] = -focal * camera[:, 1] / np.maximum(depth, 1e-8) + resolution * 0.5
    return uv, depth


def _depth_image(uv: np.ndarray, depth: np.ndarray, resolution: int) -> Image.Image:
    rendered, mask, _ = zbuffer_depth_with_indices(
        uv, depth, (resolution, resolution), splat_radius=2,
    )
    panel = np.full((resolution, resolution, 3), 18, dtype=np.uint8)
    if mask.any():
        low, high = np.quantile(rendered[mask], (0.02, 0.98))
        value = np.clip((rendered - low) / max(float(high - low), 1e-8), 0, 1)
        colour = cv2.applyColorMap((255 * (1 - value)).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel[mask] = colour[mask, ::-1]
    return Image.fromarray(panel)


def _label(panel: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    width = min(panel.width - 16, max(190, 8 * len(text)))
    draw.rounded_rectangle((8, 8, 8 + width, 36), radius=5, fill=(0, 0, 0))
    draw.text((15, 15), text, fill="white")
    return panel


def _voxel_medians(
    source: np.ndarray,
    displacement: np.ndarray,
    residual: np.ndarray,
    voxel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor(source / max(voxel, 1e-8)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    centres, targets, distances, counts = [], [], [], []
    for group in range(int(inverse.max()) + 1):
        ids = np.flatnonzero(inverse == group)
        centres.append(np.median(source[ids], axis=0))
        targets.append(np.median(displacement[ids], axis=0))
        distances.append(np.median(residual[ids]))
        counts.append(len(ids))
    return (
        np.asarray(centres), np.asarray(targets),
        np.asarray(distances), np.asarray(counts),
    )


def _blend_matrix(points: np.ndarray, nodes: np.ndarray, sigma: float):
    neighbours = min(4, len(nodes))
    distance, index = cKDTree(nodes).query(points, k=neighbours)
    if neighbours == 1:
        distance, index = distance[:, None], index[:, None]
    weight = np.exp(-0.5 * np.square(distance / max(sigma, 1e-8)))
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    row = np.repeat(np.arange(len(points)), neighbours)
    return coo_matrix(
        (weight.ravel(), (row, index.ravel())),
        shape=(len(points), len(nodes)),
    ).tocsr()


def _shared_low_frequency_field(
    prior: np.ndarray,
    partial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    diagonal = _robust_diagonal(partial)
    spacing = _median_spacing(partial)
    residual, nearest = cKDTree(prior).query(partial, workers=-1)
    source = prior[nearest]
    displacement = partial - source
    stable_threshold = max(float(np.quantile(residual, 0.55)), 3.0 * spacing)

    node_voxel = 0.055 * diagonal
    node_keys = np.floor(prior / max(node_voxel, 1e-8)).astype(np.int64)
    unique_keys, node_inverse = np.unique(node_keys, axis=0, return_inverse=True)
    nodes = np.stack([
        np.mean(prior[node_inverse == index], axis=0)
        for index in range(len(unique_keys))
    ])
    controls, target_displacement, control_residual, control_count = _voxel_medians(
        source, displacement, residual, 0.025 * diagonal,
    )
    sigma = 0.09 * diagonal
    data_matrix = _blend_matrix(controls, nodes, sigma)

    neighbour_count = min(9, len(nodes))
    _, graph_neighbours = cKDTree(nodes).query(nodes, k=neighbour_count)
    edge_set = {
        tuple(sorted((source_id, int(target_id))))
        for source_id in range(len(nodes))
        for target_id in np.atleast_1d(graph_neighbours[source_id])[1:]
        if source_id != int(target_id)
    }
    rows, columns, values = [], [], []
    for row_id, (source_id, target_id) in enumerate(sorted(edge_set)):
        rows.extend((row_id, row_id))
        columns.extend((source_id, target_id))
        values.extend((1.0, -1.0))
    graph = coo_matrix(
        (values, (rows, columns)), shape=(len(edge_set), len(nodes)),
    ).tocsr()

    # Stable observations are stiff anchors. Corrective observations receive
    # increasing evidence weight, while graph smoothing turns their motion
    # into one continuous low-frequency field rather than detached shifts.
    data_weight = np.where(
        control_residual < stable_threshold,
        12.0,
        1.0 + np.clip(control_residual / stable_threshold, 1.0, 5.0),
    ) * np.sqrt(control_count)
    weighted_data = data_matrix.multiply(np.sqrt(data_weight)[:, None])
    smoothness = 3.0
    system = vstack((weighted_data, math.sqrt(smoothness) * graph)).tocsr()
    right_padding = np.zeros(graph.shape[0], dtype=np.float64)
    translations = np.stack([
        lsqr(
            system,
            np.r_[np.sqrt(data_weight) * target_displacement[:, axis], right_padding],
            atol=1e-8, btol=1e-8, iter_lim=400,
        )[0]
        for axis in range(3)
    ], axis=1)

    prior_blend = _blend_matrix(prior, nodes, sigma)
    pair_blend = _blend_matrix(source, nodes, sigma)
    prior_low = np.asarray(prior_blend @ translations)
    pair_low = np.asarray(pair_blend @ translations)
    shared_prior = prior + prior_low
    shared_pair = source + pair_low
    local_residual = partial - shared_pair
    local_norm = np.linalg.norm(local_residual, axis=1)
    stable = residual < stable_threshold
    diagnostics = {
        "partial_diagonal": diagonal,
        "median_spacing": spacing,
        "stable_threshold": stable_threshold,
        "node_voxel_over_diagonal": 0.055,
        "control_voxel_over_diagonal": 0.025,
        "kernel_sigma_over_diagonal": 0.09,
        "smoothness": smoothness,
        "stable_data_weight": 12.0,
        "nodes": int(len(nodes)),
        "controls": int(len(controls)),
        "stable_count": int(np.count_nonzero(stable)),
        "corrective_count": int(np.count_nonzero(~stable)),
        "total_residual_quantiles": np.quantile(residual, (0.5, 0.7, 0.85, 0.95)).tolist(),
        "shared_motion_quantiles": np.quantile(np.linalg.norm(pair_low, axis=1), (0.5, 0.7, 0.85, 0.95)).tolist(),
        "remaining_local_quantiles": np.quantile(local_norm, (0.5, 0.7, 0.85, 0.95)).tolist(),
        "stable_shared_motion_quantiles": np.quantile(
            np.linalg.norm(pair_low[stable], axis=1), (0.5, 0.9, 0.99),
        ).tolist(),
    }
    return shared_prior, shared_pair, residual, nearest, diagnostics


def _valid_pairs(
    source_uv: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_depth: np.ndarray,
    resolution: int,
) -> np.ndarray:
    return (
        np.isfinite(source_uv).all(axis=1) & np.isfinite(target_uv).all(axis=1)
        & (source_depth > 0) & (target_depth > 0)
        & (source_uv[:, 0] >= 0) & (source_uv[:, 0] < resolution)
        & (source_uv[:, 1] >= 0) & (source_uv[:, 1] < resolution)
        & (target_uv[:, 0] >= 0) & (target_uv[:, 0] < resolution)
        & (target_uv[:, 1] >= 0) & (target_uv[:, 1] < resolution)
    )


def _arrow_panel(
    source_uv: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_depth: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
    resolution: int,
    source_colour: tuple[int, int, int],
    target_colour: tuple[int, int, int],
) -> Image.Image:
    panel = np.full((resolution, resolution, 3), 18, dtype=np.uint8)
    selected = magnitude >= threshold
    valid = selected & _valid_pairs(
        source_uv, source_depth, target_uv, target_depth, resolution,
    )
    source, target = source_uv[valid], target_uv[valid]
    values = magnitude[valid]
    stride = max(1, len(source) // 5000)
    for point in np.rint(source[::stride]).astype(int):
        cv2.circle(panel, tuple(point), 1, source_colour, -1, cv2.LINE_AA)
    for point in np.rint(target[::stride]).astype(int):
        cv2.circle(panel, tuple(point), 2, target_colour, -1, cv2.LINE_AA)
    order = np.argsort(values)[::-1]
    occupied: set[tuple[int, int]] = set()
    cell_size = max(8, resolution // 28)
    for index in order:
        cell = tuple(np.floor(target[index] / cell_size).astype(int))
        if cell in occupied:
            continue
        occupied.add(cell)
        cv2.arrowedLine(
            panel,
            tuple(np.rint(source[index]).astype(int)),
            tuple(np.rint(target[index]).astype(int)),
            target_colour, 1, cv2.LINE_AA, tipLength=0.25,
        )
        if len(occupied) >= 112:
            break
    return Image.fromarray(panel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registered-prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--object-type", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prior = load_points(args.registered_prior)
    partial = load_points(args.partial)
    shared_prior, shared_pair, total_residual, nearest, diagnostics = (
        _shared_low_frequency_field(prior, partial)
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(shared_prior).export(output / "shared_low_frequency_prior.ply")

    source_pair = prior[nearest]
    shared_motion = np.linalg.norm(shared_pair - source_pair, axis=1)
    local_motion = np.linalg.norm(partial - shared_pair, axis=1)
    shared_threshold = max(
        2.0 * diagnostics["median_spacing"],
        float(np.quantile(shared_motion, 0.55)),
    )
    local_threshold = max(
        2.0 * diagnostics["median_spacing"],
        float(np.quantile(local_motion, 0.55)),
    )
    diagnostics["shared_visual_threshold"] = shared_threshold
    diagnostics["local_visual_threshold"] = local_threshold

    fov = float(manifest["field_of_view_degrees"])
    gutter = 8
    records = []
    cards = []
    for view in manifest["views"]:
        name = str(view["name"])
        pose = np.asarray(view["camera_pose"], dtype=np.float64)
        prior_uv, prior_depth = _project(prior, pose, fov, args.resolution)
        shared_uv, shared_depth = _project(shared_prior, pose, fov, args.resolution)
        source_uv, source_depth = _project(source_pair, pose, fov, args.resolution)
        shared_pair_uv, shared_pair_depth = _project(shared_pair, pose, fov, args.resolution)
        partial_uv, partial_depth = _project(partial, pose, fov, args.resolution)

        shared_panel = _arrow_panel(
            source_uv, source_depth, shared_pair_uv, shared_pair_depth,
            shared_motion, shared_threshold, args.resolution,
            source_colour=(35, 90, 130), target_colour=(70, 235, 130),
        )
        local_panel = _arrow_panel(
            shared_pair_uv, shared_pair_depth, partial_uv, partial_depth,
            local_motion, local_threshold, args.resolution,
            source_colour=(35, 125, 80), target_colour=(255, 85, 205),
        )
        panels = [
            _label(shared_panel, f"{name.upper()} - CURRENT TO SHARED LOW-FREQUENCY"),
            _label(_depth_image(shared_uv, shared_depth, args.resolution), f"{name.upper()} - SHARED LOW-FREQUENCY DEPTH"),
            _label(_depth_image(partial_uv, partial_depth, args.resolution), f"{name.upper()} - PARTIAL DEPTH TARGET"),
            _label(local_panel, f"{name.upper()} - REMAINING LOCAL RESIDUAL"),
        ]
        card = Image.new(
            "RGB", (args.resolution, args.resolution * 4 + gutter * 3),
            (245, 245, 245),
        )
        for row, panel in enumerate(panels):
            card.paste(panel, (0, row * (args.resolution + gutter)))
        card_path = output / f"{name}_shared_local_residual_card.png"
        card.save(card_path)
        cards.append(card)
        records.append({
            "name": name,
            "source_image": str(Path(view["image"]).resolve()),
            "camera_pose": view["camera_pose"],
            "card": str(card_path),
        })

        prompt = build_local_residual_prompt(args.object_type, name)
        (output / f"{name}_prompt.txt").write_text(prompt, encoding="utf-8")

    board = Image.new(
        "RGB",
        (args.resolution * len(cards) + gutter * (len(cards) - 1), cards[0].height),
        (245, 245, 245),
    )
    for column, card in enumerate(cards):
        board.paste(card, (column * (args.resolution + gutter), 0))
    board_path = output / "shared_local_residual_board.png"
    board.save(board_path)
    stage1_prompt_path = output / "stage1_shared_low_frequency_prompt.txt"
    stage1_prompt_path.write_text(
        build_shared_low_frequency_prompt(args.object_type), encoding="utf-8",
    )
    ledger = {
        "method": "shared_low_frequency_3d_plus_local_residual",
        "ground_truth_used": False,
        "object_type": args.object_type,
        "registered_prior": str(args.registered_prior.resolve()),
        "partial": str(args.partial.resolve()),
        "manifest": str(args.manifest.resolve()),
        "diagnostics": diagnostics,
        "views": records,
        "board": str(board_path),
        "stage1_prompt": str(stage1_prompt_path),
        "shared_prior": str(output / "shared_low_frequency_prior.ply"),
    }
    (output / "shared_local_residual_ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "board": str(board_path),
        "ledger": str(output / "shared_local_residual_ledger.json"),
        "diagnostics": diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
