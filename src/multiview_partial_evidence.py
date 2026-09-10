"""Four-view positive evidence for camera-consistent prior regeneration.

Missing partial pixels are unknown rather than negative silhouette evidence.
The module contains no category, part, sample, or ground-truth rule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from src.pointcloud_io import load_points
from src.zbuffer import zbuffer_depth_with_indices


VIEW_ORDER = ("front", "side", "back", "right")


class PoseProjector:
    """Perspective projector reconstructed from a saved diagnostic camera."""

    def __init__(self, pose: np.ndarray, field_of_view_degrees: float, resolution: int):
        self.pose = np.asarray(pose, dtype=np.float64)
        self.field_of_view_degrees = float(field_of_view_degrees)
        self.image_shape = (int(resolution), int(resolution))
        if self.pose.shape != (4, 4) or min(self.image_shape) < 8:
            raise ValueError("pose must be 4x4 and resolution must be at least eight")

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _project(
            points, self.pose, self.field_of_view_degrees, self.image_shape[0],
        )


def load_manifest_projectors(manifest_path: Path, *, resolution: int = 512) -> list[PoseProjector]:
    """Load ordered, GT-free camera projectors from a render manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fov = float(manifest["field_of_view_degrees"])
    views = list(manifest.get("views", []))
    if not views:
        raise ValueError("render manifest contains no views")
    return [
        PoseProjector(np.asarray(view["camera_pose"], dtype=np.float64), fov, int(resolution))
        for view in views
    ]


def _project(
    points: np.ndarray,
    pose: np.ndarray,
    field_of_view_degrees: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(np.asarray(pose, dtype=np.float64))
    camera = np.asarray(points, dtype=np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -camera[:, 2]
    focal = 0.5 * resolution / math.tan(math.radians(field_of_view_degrees) * 0.5)
    uv = np.column_stack((
        focal * camera[:, 0] / np.maximum(depth, 1e-8) + resolution * 0.5,
        -focal * camera[:, 1] / np.maximum(depth, 1e-8) + resolution * 0.5,
    ))
    return uv, depth


def compose_grid(images: Iterable[Image.Image | Path], output: Path) -> Path:
    """Compose exactly four equally sized images in row-major 2x2 order."""
    loaded = [
        Image.open(item).convert("RGB") if isinstance(item, (str, Path)) else item.convert("RGB")
        for item in images
    ]
    if len(loaded) != 4:
        raise ValueError("a four-view grid requires exactly four images")
    width = max(image.width for image in loaded)
    height = max(image.height for image in loaded)
    board = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, image in enumerate(loaded):
        x = (index % 2) * width + (width - image.width) // 2
        y = (index // 2) * height + (height - image.height) // 2
        board.paste(image, (x, y))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    return output


def compose_contact_sheet(
    images: Iterable[Image.Image | Path],
    output: Path,
    *,
    columns: int,
) -> Path:
    """Compose any number of equal-camera panels without changing their pixels."""
    loaded = [
        Image.open(item).convert("RGB") if isinstance(item, (str, Path)) else item.convert("RGB")
        for item in images
    ]
    if not loaded:
        raise ValueError("a contact sheet requires at least one image")
    if int(columns) < 1:
        raise ValueError("columns must be positive")
    width = max(image.width for image in loaded)
    height = max(image.height for image in loaded)
    rows = int(math.ceil(len(loaded) / int(columns)))
    board = Image.new("RGB", (int(columns) * width, rows * height), "white")
    for index, image in enumerate(loaded):
        x = (index % int(columns)) * width + (width - image.width) // 2
        y = (index // int(columns)) * height + (height - image.height) // 2
        board.paste(image, (x, y))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    return output


def split_grid(board: Path, output_dir: Path) -> dict[str, Path]:
    """Split a row-major 2x2 board into FRONT/SIDE/BACK/RIGHT views."""
    image = Image.open(board).convert("RGB")
    if image.width % 2 or image.height % 2:
        raise ValueError("four-view board dimensions must be divisible by two")
    width, height = image.width // 2, image.height // 2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for index, name in enumerate(VIEW_ORDER):
        left, top = (index % 2) * width, (index // 2) * height
        path = output_dir / f"{name}.png"
        image.crop((left, top, left + width, top + height)).save(path)
        outputs[name] = path
    return outputs


def build_prompt(object_type: str) -> str:
    category = str(object_type).strip()
    if not category:
        raise ValueError("object_type must be non-empty")
    return f"""Use case: camera-consistent multi-view geometric correction

Image 1 is a 2x2 grid of clean RGB views of the same complete {category}. The fixed row-major camera order is FRONT, SIDE, BACK, RIGHT.
Image 2 uses exactly the same cameras and panel order. It keeps the complete RGB prior visible and overlays visibility-valid correspondences from the current prior to a partial 3-D scan. Red points are current prior surface locations, cyan points are observed target locations, and yellow arrows point from red to cyan. Unmarked prior pixels are UNKNOWN because the scan is incomplete; they are not empty space and must never cause deletion.

Produce one corrected 2x2 RGB grid in exactly the same FRONT, SIDE, BACK, RIGHT order. Apply one shared three-dimensional shape correction: move the corresponding connected surface from each red source toward its cyan target, following the yellow direction consistently across views. Make coherent component displacement and contour correction clearly visible rather than merely recolouring the original. Preserve Image 1 wherever evidence is absent or already compatible. Keep the object complete and connected, including all unobserved geometry. Preserve its identity, topology, number of structures, material and texture.

Keep every camera, viewpoint, framing, object orientation, lighting and pure white background exactly fixed. Do not independently redesign panels, swap views, mirror the object, invent or remove structures, copy diagnostic colours, or render depth. Output only the clean 2x2 RGB grid with no text, labels, arrows, borders or watermark."""


def build_partial_evidence(
    *,
    manifest_path: Path,
    registered_prior_path: Path,
    partial_path: Path,
    output_dir: Path,
    resolution: int = 512,
    occlusion_tolerance_ratio: float = 0.02,
    view_order: Iterable[str] | None = None,
    grid_columns: int | None = None,
) -> dict:
    """Build prior RGB and positive partial-evidence grids in matching cameras."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    views = {str(item["name"]): item for item in manifest["views"]}
    order = VIEW_ORDER if view_order is None else tuple(str(name) for name in view_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("view_order must contain unique view names")
    missing = tuple(name for name in order if name not in views)
    if missing:
        raise ValueError(f"manifest is missing requested views: {missing}")
    prior = load_points(Path(registered_prior_path))
    partial = load_points(Path(partial_path))
    pair_distance, nearest_prior = cKDTree(prior).query(partial, workers=-1)
    paired_prior = prior[nearest_prior]
    lower, upper = np.quantile(partial, (0.01, 0.99), axis=0)
    diagonal = max(float(np.linalg.norm(upper - lower)), 1e-8)
    tolerance = float(occlusion_tolerance_ratio) * diagonal
    fov = float(manifest["field_of_view_degrees"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_images: list[Path] = []
    evidence_images: list[Path] = []
    overlay_images: list[Path] = []
    correspondence_images: list[Path] = []
    paper_residual_images: list[Path] = []
    records = []
    for name in order:
        view = views[name]
        prior_image = Path(view["image"]).resolve()
        if not prior_image.is_file():
            raise FileNotFoundError(prior_image)
        prior_images.append(prior_image)
        pose = np.asarray(view["camera_pose"], dtype=np.float64)
        prior_uv, prior_depth = _project(prior, pose, fov, resolution)
        partial_uv, partial_depth = _project(partial, pose, fov, resolution)
        source_uv, source_depth = _project(paired_prior, pose, fov, resolution)
        prior_z, prior_mask, _ = zbuffer_depth_with_indices(
            prior_uv, prior_depth, (resolution, resolution), splat_radius=2,
        )
        partial_z, partial_mask, _ = zbuffer_depth_with_indices(
            partial_uv, partial_depth, (resolution, resolution), splat_radius=2,
        )
        visible = partial_mask & (
            ~prior_mask | (partial_z <= prior_z + tolerance)
        )
        panel = np.full((resolution, resolution, 3), 18, dtype=np.uint8)
        if np.any(visible):
            low, high = np.quantile(partial_z[visible], (0.02, 0.98))
            normalized = np.clip(
                (partial_z - low) / max(float(high - low), 1e-8), 0.0, 1.0,
            )
            colour = cv2.applyColorMap(
                np.rint(255.0 * (1.0 - normalized)).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )[:, :, ::-1]
            panel[visible] = colour[visible]
        evidence_path = output_dir / f"partial_evidence_{name}.png"
        Image.fromarray(panel).save(evidence_path)
        evidence_images.append(evidence_path)
        prior_rgb = np.asarray(
            Image.open(prior_image).convert("RGB").resize(
                (resolution, resolution), Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        ).copy()
        overlay = prior_rgb.copy()
        overlay[visible] = np.rint(
            0.25 * prior_rgb[visible].astype(np.float32)
            + 0.75 * panel[visible].astype(np.float32)
        ).astype(np.uint8)
        overlay_path = output_dir / f"partial_positive_overlay_{name}.png"
        Image.fromarray(overlay).save(overlay_path)
        overlay_images.append(overlay_path)

        correspondence = prior_rgb.copy()
        prior_gray = cv2.cvtColor(prior_rgb, cv2.COLOR_RGB2GRAY)
        paper_residual = np.repeat(prior_gray[:, :, None], 3, axis=2)
        foreground = np.any(prior_rgb < 245, axis=2)
        paper_residual[foreground] = np.rint(
            0.62 * paper_residual[foreground].astype(np.float32) + 0.38 * 255.0
        ).astype(np.uint8)
        valid_points = (
            np.isfinite(source_uv).all(axis=1)
            & np.isfinite(partial_uv).all(axis=1)
            & (source_depth > 0.0) & (partial_depth > 0.0)
            & (source_uv[:, 0] >= 0.0) & (source_uv[:, 0] < resolution)
            & (source_uv[:, 1] >= 0.0) & (source_uv[:, 1] < resolution)
            & (partial_uv[:, 0] >= 0.0) & (partial_uv[:, 0] < resolution)
            & (partial_uv[:, 1] >= 0.0) & (partial_uv[:, 1] < resolution)
        )
        target_pixel = np.rint(partial_uv).astype(np.int64)
        target_pixel[:, 0] = np.clip(target_pixel[:, 0], 0, resolution - 1)
        target_pixel[:, 1] = np.clip(target_pixel[:, 1], 0, resolution - 1)
        point_visible = visible[target_pixel[:, 1], target_pixel[:, 0]]
        valid_points &= point_visible
        projected_motion = np.linalg.norm(partial_uv - source_uv, axis=1)
        valid_ids = np.flatnonzero(valid_points)
        selected_vectors: list[int] = []
        if len(valid_ids):
            # Draw every target and source at a density bounded only for
            # readability. Correspondence eligibility itself has no residual
            # magnitude gate.
            stride = max(1, len(valid_ids) // 12_000)
            for point in np.rint(source_uv[valid_ids[::stride]]).astype(int):
                cv2.circle(correspondence, tuple(point), 1, (235, 45, 45), -1, cv2.LINE_AA)
            for point in np.rint(partial_uv[valid_ids[::stride]]).astype(int):
                cv2.circle(correspondence, tuple(point), 2, (40, 225, 235), -1, cv2.LINE_AA)
            ranked_ids = valid_ids[np.argsort(projected_motion[valid_ids])[::-1]]
            occupied: set[tuple[int, int]] = set()
            cell = max(10, resolution // 26)
            for index in ranked_ids:
                key = tuple(np.floor(partial_uv[index] / cell).astype(int))
                if key in occupied or projected_motion[index] < 1.0:
                    continue
                occupied.add(key)
                selected_vectors.append(int(index))
                cv2.arrowedLine(
                    correspondence,
                    tuple(np.rint(source_uv[index]).astype(int)),
                    tuple(np.rint(partial_uv[index]).astype(int)),
                    (245, 210, 30), 2, cv2.LINE_AA, tipLength=0.28,
                )
                if len(occupied) >= 96:
                    break

            # A separate publication-oriented diagnostic uses only spatially
            # distributed vectors. Red is the registered prior source, cyan
            # is the physical partial target, and the yellow arrowhead points
            # in the actual edit direction: prior -> partial.
            for index in selected_vectors[:64]:
                source = tuple(np.rint(source_uv[index]).astype(int))
                target = tuple(np.rint(partial_uv[index]).astype(int))
                cv2.circle(paper_residual, source, 4, (230, 45, 45), -1, cv2.LINE_AA)
                cv2.circle(paper_residual, target, 4, (25, 205, 225), -1, cv2.LINE_AA)
                cv2.arrowedLine(
                    paper_residual, source, target,
                    (35, 35, 35), 5, cv2.LINE_AA, tipLength=0.34,
                )
                cv2.arrowedLine(
                    paper_residual, source, target,
                    (250, 195, 25), 2, cv2.LINE_AA, tipLength=0.34,
                )
        correspondence_path = output_dir / f"partial_correspondence_overlay_{name}.png"
        Image.fromarray(correspondence).save(correspondence_path)
        correspondence_images.append(correspondence_path)
        paper_residual_path = output_dir / f"prior_to_partial_residual_{name}.png"
        Image.fromarray(paper_residual).save(paper_residual_path)
        paper_residual_images.append(paper_residual_path)
        records.append({
            "name": name,
            "prior_rgb": str(prior_image),
            "partial_evidence": str(evidence_path.resolve()),
            "positive_overlay": str(overlay_path.resolve()),
            "correspondence_overlay": str(correspondence_path.resolve()),
            "paper_residual": str(paper_residual_path.resolve()),
            "residual_arrow_direction": "registered_prior_to_physical_partial",
            "displayed_residual_vectors": int(min(len(selected_vectors), 64)),
            "visible_correspondences": int(np.count_nonzero(valid_points)),
            "projected_pixels": int(np.count_nonzero(partial_mask)),
            "visible_positive_pixels": int(np.count_nonzero(visible)),
            "occluded_unknown_pixels": int(np.count_nonzero(partial_mask & ~visible)),
            "camera_pose": view["camera_pose"],
        })

    suffix = "front_side_back_right" if order == VIEW_ORDER else f"{len(order)}view"
    if order == VIEW_ORDER and grid_columns is None:
        compose = compose_grid
    else:
        columns = int(grid_columns or math.ceil(math.sqrt(len(order))))
        compose = lambda images, output: compose_contact_sheet(images, output, columns=columns)
    prior_board = compose(prior_images, output_dir / f"prior_rgb_{suffix}.png")
    evidence_board = compose(evidence_images, output_dir / f"partial_evidence_{suffix}.png")
    overlay_board = compose(overlay_images, output_dir / f"partial_positive_overlay_{suffix}.png")
    correspondence_board = compose(
        correspondence_images, output_dir / f"partial_correspondence_overlay_{suffix}.png",
    )
    paper_residual_board = compose(
        paper_residual_images, output_dir / f"prior_to_partial_residual_{suffix}.png",
    )
    record = {
        "method": "visibility_aware_positive_partial_evidence",
        "ground_truth_used": False,
        "view_order": list(order),
        "registered_prior": str(Path(registered_prior_path).resolve()),
        "partial": str(Path(partial_path).resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "partial_diagonal": diagonal,
        "occlusion_tolerance_ratio": float(occlusion_tolerance_ratio),
        "occlusion_tolerance": tolerance,
        "prior_board": str(prior_board.resolve()),
        "evidence_board": str(evidence_board.resolve()),
        "positive_overlay_board": str(overlay_board.resolve()),
        "correspondence_overlay_board": str(correspondence_board.resolve()),
        "paper_residual_board": str(paper_residual_board.resolve()),
        "residual_arrow_direction": "registered_prior_to_physical_partial",
        "pair_distance_quantiles": np.quantile(pair_distance, (0.5, 0.75, 0.9, 0.95)).tolist(),
        "views": records,
    }
    ledger = output_dir / "partial_evidence_manifest.json"
    ledger.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["ledger"] = str(ledger.resolve())
    return record
