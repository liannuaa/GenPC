#!/usr/bin/env python3
"""Build category-independent, component-consistent multi-view edit evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pointcloud_io import load_points
from src.zbuffer import zbuffer_depth_with_indices


PALETTE = (
    (255, 92, 205),
    (108, 238, 94),
    (255, 168, 48),
    (77, 166, 255),
    (245, 226, 66),
    (174, 112, 255),
    (52, 211, 153),
    (251, 113, 133),
    (96, 165, 250),
    (250, 204, 21),
    (192, 132, 252),
    (45, 212, 191),
)


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


def _residual_components(
    partial: np.ndarray,
    prior: np.ndarray,
    max_components: int,
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    diagonal = _robust_diagonal(partial)
    spacing = _median_spacing(partial)
    distance, nearest = cKDTree(prior).query(partial, workers=-1)
    # The old ledger kept only the upper residual tail and then truncated the
    # connected components.  That made visually dominant end structures the
    # only actionable evidence.  Here every non-stable residual receives a
    # continuous confidence; components are merely explanatory groupings.
    stable_threshold = max(float(np.quantile(distance, 0.55)), 3.0 * spacing)
    strong_threshold = max(float(np.quantile(distance, 0.85)), 2.0 * stable_threshold)
    candidate_ids = np.flatnonzero(distance >= stable_threshold)
    if len(candidate_ids) == 0:
        return [], distance, nearest, {
            "partial_diagonal": diagonal,
            "median_spacing": spacing,
            "stable_threshold": stable_threshold,
            "strong_threshold": strong_threshold,
        }
    candidate = partial[candidate_ids]
    radius = max(6.0 * spacing, 0.0125 * diagonal)
    edges = cKDTree(candidate).query_pairs(radius, output_type="ndarray")
    if len(edges):
        row = np.concatenate((edges[:, 0], edges[:, 1], np.arange(len(candidate))))
        col = np.concatenate((edges[:, 1], edges[:, 0], np.arange(len(candidate))))
    else:
        row = col = np.arange(len(candidate))
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(candidate), len(candidate)))
    count, labels = connected_components(graph, directed=False)
    minimum_size = max(8, int(math.ceil(0.0005 * len(partial))))
    components = []
    for label in range(count):
        local = np.flatnonzero(labels == label)
        if len(local) < minimum_size:
            continue
        ids = candidate_ids[local]
        residual = distance[ids]
        displacement = partial[ids] - prior[nearest[ids]]
        components.append({
            "partial_ids": ids,
            "prior_ids": nearest[ids],
            "size": int(len(ids)),
            "median_residual": float(np.median(residual)),
            "p90_residual": float(np.quantile(residual, 0.90)),
            "median_displacement": np.median(displacement, axis=0),
            "score": float(len(ids) * np.median(residual)),
        })
    components.sort(key=lambda item: item["score"], reverse=True)
    retained = components[:max_components]
    retained_ids = sum((len(item["partial_ids"]) for item in retained), 0)
    return retained, distance, nearest, {
        "partial_diagonal": diagonal,
        "median_spacing": spacing,
        "stable_threshold": stable_threshold,
        "strong_threshold": strong_threshold,
        "component_radius": radius,
        "minimum_component_size": minimum_size,
        "partial_count": int(len(partial)),
        "stable_count": int(np.count_nonzero(distance < stable_threshold)),
        "corrective_count": int(np.count_nonzero(distance >= stable_threshold)),
        "strong_count": int(np.count_nonzero(distance >= strong_threshold)),
        "corrective_coverage": float(np.mean(distance >= stable_threshold)),
        "annotated_component_count": int(len(retained)),
        "annotated_point_count": int(retained_ids),
        "annotated_fraction_of_corrective": float(retained_ids / max(len(candidate_ids), 1)),
        "residual_quantiles": {
            str(quantile): float(np.quantile(distance, quantile))
            for quantile in (0.25, 0.50, 0.55, 0.70, 0.85, 0.95)
        },
    }


def _component_panel(
    prior_uv: np.ndarray,
    prior_depth: np.ndarray,
    partial_uv: np.ndarray,
    partial_depth: np.ndarray,
    components: list[dict],
    distance: np.ndarray,
    nearest: np.ndarray,
    stable_threshold: float,
    strong_threshold: float,
    resolution: int,
) -> tuple[Image.Image, list[dict]]:
    panel = np.full((resolution, resolution, 3), 18, dtype=np.uint8)
    # Dense continuous evidence: every residual above the automatically
    # estimated stable threshold is shown, even if it belongs to a small or
    # low-ranked component. Cyan denotes moderate correction and red denotes
    # strong correction; dim points are prior locations and bright points are
    # partial-supported targets.
    corrective_ids = np.flatnonzero(distance >= float(stable_threshold))
    source = prior_uv[nearest[corrective_ids]]
    target = partial_uv[corrective_ids]
    valid = (
        np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        & (prior_depth[nearest[corrective_ids]] > 0) & (partial_depth[corrective_ids] > 0)
        & (source[:, 0] >= 0) & (source[:, 0] < resolution)
        & (source[:, 1] >= 0) & (source[:, 1] < resolution)
        & (target[:, 0] >= 0) & (target[:, 0] < resolution)
        & (target[:, 1] >= 0) & (target[:, 1] < resolution)
    )
    ids = corrective_ids[valid]
    source, target = source[valid], target[valid]
    denominator = max(float(strong_threshold - stable_threshold), 1e-8)
    confidence = np.clip((distance[ids] - stable_threshold) / denominator, 0.0, 1.0)
    moderate = np.array([50, 205, 255], dtype=np.float64)
    strong = np.array([255, 70, 70], dtype=np.float64)
    colours = np.rint(
        (1.0 - confidence[:, None]) * moderate[None]
        + confidence[:, None] * strong[None]
    ).astype(np.uint8)
    stride = max(1, len(source) // 4500)
    for point, colour in zip(np.rint(source[::stride]).astype(int), colours[::stride]):
        cv2.circle(panel, tuple(point), 1, tuple(int(v * 0.38) for v in colour), -1, cv2.LINE_AA)
    for point, colour in zip(np.rint(target[::stride]).astype(int), colours[::stride]):
        cv2.circle(panel, tuple(point), 2, tuple(int(v) for v in colour), -1, cv2.LINE_AA)
    delta = target - source
    order = np.argsort(distance[ids])[::-1]
    occupied: set[tuple[int, int]] = set()
    cell_size = max(8, resolution // 28)
    for index in order:
        cell = tuple(np.floor(target[index] / cell_size).astype(int))
        if cell in occupied:
            continue
        occupied.add(cell)
        colour = tuple(int(v) for v in colours[index])
        cv2.arrowedLine(
            panel, tuple(np.rint(source[index]).astype(int)),
            tuple(np.rint(target[index]).astype(int)), colour,
            1, cv2.LINE_AA, tipLength=0.25,
        )
        if len(occupied) >= 112:
            break

    records = []
    for component_id, component in enumerate(components):
        colour = PALETTE[component_id % len(PALETTE)]
        partial_ids = component["partial_ids"]
        prior_ids = component["prior_ids"]
        source = prior_uv[prior_ids]
        target = partial_uv[partial_ids]
        valid = (
            np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
            & (prior_depth[prior_ids] > 0) & (partial_depth[partial_ids] > 0)
            & (source[:, 0] >= 0) & (source[:, 0] < resolution)
            & (source[:, 1] >= 0) & (source[:, 1] < resolution)
            & (target[:, 0] >= 0) & (target[:, 0] < resolution)
            & (target[:, 1] >= 0) & (target[:, 1] < resolution)
        )
        source, target = source[valid], target[valid]
        if len(source) == 0:
            records.append({"visible_pairs": 0})
            continue
        delta = target - source
        depth_delta = partial_depth[partial_ids][valid] - prior_depth[prior_ids][valid]
        # Major connected components remain colour-coded only as a structural
        # cross-view reference; they no longer suppress unannotated evidence.
        source_colour = tuple(int(channel * 0.42) for channel in colour)
        for point in np.rint(source[::max(1, len(source) // 900)]).astype(int):
            cv2.circle(panel, tuple(point), 1, source_colour, -1, cv2.LINE_AA)
        for point in np.rint(target[::max(1, len(target) // 900)]).astype(int):
            cv2.circle(panel, tuple(point), 2, colour, -1, cv2.LINE_AA)
        order = np.argsort(np.linalg.norm(delta, axis=1))[::-1]
        occupied: set[tuple[int, int]] = set()
        for index in order:
            cell = tuple(np.floor(target[index] / max(10, resolution // 24)).astype(int))
            if cell in occupied:
                continue
            occupied.add(cell)
            cv2.arrowedLine(
                panel,
                tuple(np.rint(source[index]).astype(int)),
                tuple(np.rint(target[index]).astype(int)),
                colour,
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            if len(occupied) >= 12:
                break
        records.append({
            "visible_pairs": int(len(source)),
            "median_pixel_shift": np.median(delta, axis=0).tolist(),
            "p90_pixel_shift": float(np.quantile(np.linalg.norm(delta, axis=1), 0.90)),
            "median_depth_shift": float(np.median(depth_delta)),
        })
    return Image.fromarray(panel), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registered-prior", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--object-type", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--max-components", type=int, default=12,
        help="Maximum explanatory components; dense residual evidence is never truncated.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prior, partial = load_points(args.registered_prior), load_points(args.partial)
    components, distance, nearest, geometry = _residual_components(
        partial, prior, args.max_components,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = []
    view_records = []
    fov = float(manifest["field_of_view_degrees"])
    for view in manifest["views"]:
        pose = np.asarray(view["camera_pose"], dtype=np.float64)
        prior_uv, prior_depth = _project(prior, pose, fov, args.resolution)
        partial_uv, partial_depth = _project(partial, pose, fov, args.resolution)
        rgb = Image.open(view["image"]).convert("RGB").resize(
            (args.resolution, args.resolution), Image.Resampling.LANCZOS,
        )
        component_panel, component_records = _component_panel(
            prior_uv, prior_depth, partial_uv, partial_depth, components,
            distance, nearest, geometry["stable_threshold"],
            geometry["strong_threshold"], args.resolution,
        )
        columns.append([
            _label(rgb, f"{view['name'].upper()} - PRIOR RGB"),
            _label(_depth_image(prior_uv, prior_depth, args.resolution), f"{view['name'].upper()} - PRIOR DEPTH"),
            _label(_depth_image(partial_uv, partial_depth, args.resolution), f"{view['name'].upper()} - PARTIAL DEPTH"),
            _label(component_panel, f"{view['name'].upper()} - CONTINUOUS 3D RESIDUAL"),
        ])
        view_records.append({
            "name": view["name"],
            "camera_pose": view["camera_pose"],
            "components": component_records,
        })

    gutter = 8
    board = Image.new(
        "RGB",
        (args.resolution * len(columns) + gutter * (len(columns) - 1),
         args.resolution * 4 + gutter * 3),
        (245, 245, 245),
    )
    for column_id, panels in enumerate(columns):
        for row_id, panel in enumerate(panels):
            board.paste(panel, (
                column_id * (args.resolution + gutter),
                row_id * (args.resolution + gutter),
            ))
    board_path = output / "cross_view_residual_ledger_board.png"
    board.save(board_path)
    for record, panels in zip(view_records, columns, strict=True):
        evidence = Image.new(
            "RGB",
            (args.resolution, args.resolution * 4 + gutter * 3),
            (245, 245, 245),
        )
        for row_id, panel in enumerate(panels):
            evidence.paste(panel, (0, row_id * (args.resolution + gutter)))
        evidence_path = output / f"{record['name']}_continuous_residual_evidence.png"
        evidence.save(evidence_path)
        record["evidence_image"] = str(evidence_path)

    serialised_components = []
    strong_threshold = geometry["strong_threshold"] / geometry["partial_diagonal"]
    for component_id, component in enumerate(components):
        median_normalised = component["median_residual"] / geometry["partial_diagonal"]
        serialised_components.append({
            "id": component_id + 1,
            "colour_rgb": PALETTE[component_id % len(PALETTE)],
            "size": component["size"],
            "priority": "strong" if median_normalised >= strong_threshold else "moderate",
            "median_residual_over_partial_diagonal": median_normalised,
            "p90_residual_over_partial_diagonal": component["p90_residual"] / geometry["partial_diagonal"],
            "median_displacement_over_partial_diagonal": (
                component["median_displacement"] / geometry["partial_diagonal"]
            ).tolist(),
        })
    ledger = {
        "method": "continuous_cross_view_3d_residual_ledger",
        "ground_truth_used": False,
        "object_type": args.object_type,
        "registered_prior": str(args.registered_prior.resolve()),
        "partial": str(args.partial.resolve()),
        "manifest": str(args.manifest.resolve()),
        "geometry": geometry,
        "components": serialised_components,
        "views": view_records,
        "board": str(board_path),
    }
    ledger_path = output / "cross_view_residual_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    component_summary = "\n".join(
        f"- C{item['id']} colour RGB {tuple(item['colour_rgb'])}, {item['priority']} correction: "
        f"median/p90 residual {item['median_residual_over_partial_diagonal']:.4f}/"
        f"{item['p90_residual_over_partial_diagonal']:.4f} of the observed diagonal."
        for item in serialised_components
    ) or "- No reliable residual component was found; preserve the input exactly."
    prompt = f"""Use case: precise-object-edit
Asset type: cross-view geometry-conditioned input for 3D regeneration

Image 1 is the edit target: one clean FRONT / SIDE / BACK strip of the same complete {args.object_type}.
Image 2 is a four-row diagnostic board for the same cameras. In each view column the rows are prior RGB, prior depth, partial depth, and a dense continuous 3-D residual field. In the final row, dim points are current prior locations, bright points are observed target locations, and arrows point from prior to target. Cyan evidence denotes moderate supported correction and red evidence denotes strong supported correction. Every coloured residual point is actionable, including points outside the annotated components. Repeated component colours provide cross-view structural references only.

Component ledger:
{component_summary}

Refine Image 1 according to the complete residual field, not only the largest components. Strong residuals require a clearly visible correction in every view where evidence is visible. Moderate residuals also require a real contour or depth correction when spatially coherent; they are not merely cosmetic hints. Preserve only the uncoloured low-residual regions. Apply changes only where their targets are supported by partial depth, and express each connected 3-D correction consistently in every view where it is visible. Use the depth rows to distinguish an image-plane contour shift from motion toward or away from the camera. Preserve geometry without positive partial evidence. Propagate every accepted change smoothly through its attachment; never translate a structure as a detached piece.

Keep object identity, topology, material, texture, pose, view cameras, framing, lighting, complete unobserved support, and the FRONT / SIDE / BACK order unchanged. Output exactly one clean three-panel strip on pure white, with no labels, component colours, arrows, depth maps, text, borders, watermark, duplicated structures, missing structures, tears, or intersections.
"""
    prompt_path = output / "cross_view_residual_refinement_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(json.dumps({
        "board": str(board_path),
        "ledger": str(ledger_path),
        "prompt": str(prompt_path),
        "components": len(components),
    }, indent=2))


if __name__ == "__main__":
    main()
