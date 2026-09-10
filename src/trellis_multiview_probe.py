"""Camera-consistent multi-view utilities for the object mainline.

The helpers turn an initially registered complete prior into auditable
front/side/back/right image conditions. They never inspect ground truth or offline
completion metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from PIL import Image

from src.pointcloud_io import load_points
from src.zbuffer import zbuffer_depth_with_indices


@dataclass(frozen=True)
class RenderedView:
    name: str
    image_path: Path
    camera_pose: np.ndarray


def estimate_ordered_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Recover the exact row-vector affine map between ordered carrier points.

    The mainline preserves the ordering of the sampled Pixal carrier.  Fitting
    the map from that carrier to the registered carrier therefore provides the
    exact proper Sim(3) needed to place every vertex of the textured GLB in the
    partial coordinate frame.  The returned matrix uses the conventional
    column-vector homogeneous representation.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("ordered source and target carriers must share shape (N, 3)")
    if len(source) < 4:
        raise ValueError("at least four ordered carrier points are required")
    design = np.concatenate([source, np.ones((len(source), 1), dtype=np.float64)], axis=1)
    row_affine, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ row_affine
    rmse = float(np.sqrt(np.mean(np.square(predicted - target))))
    linear = row_affine[:3].T
    singular = np.linalg.svd(linear, compute_uv=False)
    if singular[-1] <= 1e-12 or float(singular.max() / singular.min()) > 1.0005:
        raise ValueError("ordered carrier transform is not a similarity")
    if np.linalg.det(linear) <= 0.0:
        raise ValueError("ordered carrier transform is not a proper similarity")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = row_affine[3]
    return transform, rmse


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def _normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("cannot normalise a zero-length vector")
    return vector / length


def rotate_about_axis(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a world-space direction with Rodrigues' formula."""
    vector, axis = np.asarray(vector, dtype=np.float64), _normalise(axis)
    angle = math.radians(float(degrees))
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle))
    )


def look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return an OpenGL camera-to-world matrix looking along local ``-z``."""
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    backward = _normalise(eye - target)
    right = _normalise(np.cross(up, backward))
    true_up = _normalise(np.cross(backward, right))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([right, true_up, backward], axis=1)
    pose[:3, 3] = eye
    return pose


def _circular_yaw_distance(first: float, second: float) -> float:
    difference = abs((float(first) - float(second)) % 360.0)
    return min(difference, 360.0 - difference)


def _visible_point_ids(
    points: np.ndarray,
    pose: np.ndarray,
    field_of_view_degrees: float,
    resolution: int,
) -> np.ndarray:
    """Return point ids surviving a deterministic point z-buffer."""
    inverse = np.linalg.inv(np.asarray(pose, dtype=np.float64))
    camera = np.asarray(points, dtype=np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -camera[:, 2]
    focal = 0.5 * int(resolution) / math.tan(
        math.radians(float(field_of_view_degrees)) * 0.5
    )
    uv = np.column_stack((
        focal * camera[:, 0] / np.maximum(depth, 1e-8) + int(resolution) * 0.5,
        -focal * camera[:, 1] / np.maximum(depth, 1e-8) + int(resolution) * 0.5,
    ))
    _, mask, indices = zbuffer_depth_with_indices(
        uv, depth, (int(resolution), int(resolution)), splat_radius=1,
    )
    return np.unique(indices[mask])


def select_informative_orbit_yaws(
    *,
    partial: np.ndarray,
    centre: np.ndarray,
    front_direction: np.ndarray,
    up_direction: np.ndarray,
    camera_distance: float,
    field_of_view_degrees: float,
    num_views: int = 4,
    candidate_yaw_step_degrees: float = 15.0,
    min_yaw_separation_degrees: float = 45.0,
    resolution: int = 256,
) -> dict:
    """Select Camera-1 plus views with maximal visible partial support.

    Selection is category-free and GT-free. Camera-1 (yaw zero) is immutable;
    auxiliary views are ranked by their total number of z-buffer-visible
    partial points. A minimum angular separation prevents near-duplicate
    cameras. Incremental coverage is recorded only as a diagnostic.
    """
    points = np.asarray(partial, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("partial must be a non-empty array with shape (N, 3)")
    if int(num_views) < 1:
        raise ValueError("num_views must be positive")
    step = float(candidate_yaw_step_degrees)
    separation = float(min_yaw_separation_degrees)
    if not 0.0 < step <= 180.0:
        raise ValueError("candidate_yaw_step_degrees must be in (0, 180]")
    if not 0.0 <= separation <= 180.0:
        raise ValueError("min_yaw_separation_degrees must be in [0, 180]")

    centre = np.asarray(centre, dtype=np.float64)
    front = _normalise(front_direction)
    up = _normalise(up_direction)
    candidates: dict[float, np.ndarray] = {}
    yaw_values = np.arange(0.0, 360.0, step, dtype=np.float64)
    if not np.any(np.isclose(yaw_values, 0.0)):
        yaw_values = np.r_[0.0, yaw_values]
    for raw_yaw in yaw_values:
        yaw = float(raw_yaw % 360.0)
        direction = _normalise(rotate_about_axis(front, up, yaw))
        pose = look_at_pose(centre + direction * float(camera_distance), centre, up)
        candidates[yaw] = _visible_point_ids(
            points, pose, field_of_view_degrees, resolution,
        )

    selected = [0.0]
    covered = set(int(index) for index in candidates[0.0])
    decisions = [{
        "rank": 1,
        "yaw_degrees": 0.0,
        "visible_points": int(len(candidates[0.0])),
        "new_visible_points": int(len(candidates[0.0])),
        "camera1_fixed": True,
    }]
    while len(selected) < int(num_views):
        eligible = [
            yaw for yaw in candidates if yaw not in selected
            and all(
                _circular_yaw_distance(yaw, previous) >= separation - 1e-9
                for previous in selected
            )
        ]
        if not eligible:
            raise ValueError(
                "candidate step and minimum separation cannot provide the requested views"
            )

        def score(yaw: float) -> tuple[int, float, float]:
            ids = candidates[yaw]
            diversity = min(_circular_yaw_distance(yaw, value) for value in selected)
            # The user-facing rule is literal and auditable: maximise visible
            # physical partial points, using diversity and yaw only as ties.
            return int(len(ids)), diversity, -yaw

        chosen = max(eligible, key=score)
        chosen_ids = candidates[chosen]
        new_count = sum(int(index) not in covered for index in chosen_ids)
        selected.append(chosen)
        covered.update(int(index) for index in chosen_ids)
        decisions.append({
            "rank": int(len(selected)),
            "yaw_degrees": float(chosen),
            "visible_points": int(len(chosen_ids)),
            "new_visible_points": int(new_count),
            "camera1_fixed": False,
        })

    return {
        "method": "camera1_plus_ranked_visible_partial_support",
        "ground_truth_used": False,
        "num_partial_points": int(len(points)),
        "candidate_yaw_step_degrees": step,
        "min_yaw_separation_degrees": separation,
        "selection_resolution": int(resolution),
        "selected": decisions,
        "covered_partial_points": int(len(covered)),
        "candidate_visible_points": {
            f"{yaw:.6g}": int(len(indices))
            for yaw, indices in sorted(candidates.items())
        },
    }


def _load_world_mesh(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(glb_path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    meshes = [item for item in loaded.dump() if isinstance(item, trimesh.Trimesh)]
    if not meshes:
        raise ValueError(f"no triangle mesh found in {glb_path}")
    return trimesh.util.concatenate(meshes)


def _camera_vectors(camera_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Kaolin is imported lazily so geometry-only unit tests do not require it.
    import torch

    camera = torch.load(str(camera_path), map_location="cpu", weights_only=False)
    camera_position = camera.cam_pos().detach().float().cpu().numpy().reshape(-1, 3)[0]
    rotation = camera.R.detach().float().cpu().numpy().reshape(-1, 3, 3)[0]
    return camera_position.astype(np.float64), _normalise(rotation[1])


def render_registered_multiview(
    *,
    glb_path: Path,
    source_carrier_path: Path,
    registered_carrier_path: Path,
    camera_path: Path,
    output_dir: Path,
    resolution: int = 768,
    field_of_view_degrees: float = 38.0,
    view_yaws: Iterable[tuple[str, float]] | None = None,
    informative_partial_path: Path | None = None,
    candidate_yaw_step_degrees: float = 15.0,
    min_yaw_separation_degrees: float = 45.0,
    selection_resolution: int = 256,
    num_views: int = 4,
) -> dict:
    """Render a registered textured Pixal mesh from consistent orbit views."""
    if resolution < 128:
        raise ValueError("resolution must be at least 128")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cloud = trimesh.load(str(source_carrier_path), process=False)
    target_cloud = trimesh.load(str(registered_carrier_path), process=False)
    source = np.asarray(source_cloud.vertices, dtype=np.float64)
    target = np.asarray(target_cloud.vertices, dtype=np.float64)
    transform, fit_rmse = estimate_ordered_similarity(source, target)

    mesh = _load_world_mesh(Path(glb_path))
    mesh.vertices = apply_transform(mesh.vertices, transform)
    centre = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    radius = max(float(np.linalg.norm(mesh.vertices - centre, axis=1).max()), 1e-4)
    saved_eye, saved_up = _camera_vectors(Path(camera_path))
    front_direction = _normalise(saved_eye - centre)
    # Avoid a degenerate up vector if an unusual acquisition camera points
    # almost exactly along its own nominal vertical axis.
    saved_up = saved_up - front_direction * float(np.dot(saved_up, front_direction))
    if np.linalg.norm(saved_up) < 1e-5:
        saved_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        saved_up -= front_direction * float(np.dot(saved_up, front_direction))
    saved_up = _normalise(saved_up)
    fov = math.radians(float(field_of_view_degrees))
    camera_distance = radius / max(math.sin(fov * 0.5), 1e-3) * 1.12
    view_selection = None
    if informative_partial_path is not None:
        if view_yaws is not None:
            raise ValueError("explicit view_yaws and informative_partial_path are mutually exclusive")
        view_selection = select_informative_orbit_yaws(
            partial=load_points(Path(informative_partial_path)),
            centre=centre,
            front_direction=front_direction,
            up_direction=saved_up,
            camera_distance=camera_distance,
            field_of_view_degrees=field_of_view_degrees,
            num_views=int(num_views),
            candidate_yaw_step_degrees=candidate_yaw_step_degrees,
            min_yaw_separation_degrees=min_yaw_separation_degrees,
            resolution=selection_resolution,
        )
        if int(num_views) == 4:
            # Preserve the established four-view names for existing tools.
            names = ("front", "side", "back", "right")
        else:
            names = ("front",) + tuple(
                f"view_{index:02d}" for index in range(2, int(num_views) + 1)
            )
        view_yaws = tuple(
            (name, record["yaw_degrees"])
            for name, record in zip(names, view_selection["selected"])
        )
    elif view_yaws is None:
        view_yaws = (
            ("front", 0.0), ("side", 90.0), ("back", 180.0), ("right", 270.0),
        )
    else:
        view_yaws = tuple(view_yaws)

    # Imported lazily because pyrender is supplied by the isolated TRELLIS
    # environment rather than the GenPC main environment.
    import pyrender

    scene = pyrender.Scene(
        bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.array([0.55, 0.55, 0.55, 1.0], dtype=np.float32),
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0, znear=max(radius * 0.01, 1e-4))
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    rendered: list[RenderedView] = []
    try:
        for name, yaw in view_yaws:
            direction = _normalise(rotate_about_axis(front_direction, saved_up, float(yaw)))
            eye = centre + direction * camera_distance
            pose = look_at_pose(eye, centre, saved_up)
            camera_node = scene.add(camera, pose=pose)
            light_nodes = [
                scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.4), pose=pose),
                scene.add(
                    pyrender.DirectionalLight(color=np.ones(3), intensity=1.2),
                    pose=look_at_pose(centre - direction * camera_distance, centre, saved_up),
                ),
            ]
            colour, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            image_path = output_dir / f"pixal_registered_{name}.png"
            Image.fromarray(colour, mode="RGBA").convert("RGB").save(image_path)
            rendered.append(RenderedView(name, image_path, pose))
            scene.remove_node(camera_node)
            for node in light_nodes:
                scene.remove_node(node)
    finally:
        renderer.delete()

    gutter = max(8, resolution // 32)
    board = Image.new("RGB", (resolution * len(rendered) + gutter * (len(rendered) - 1), resolution), "white")
    for index, item in enumerate(rendered):
        board.paste(Image.open(item.image_path).convert("RGB"), (index * (resolution + gutter), 0))
    board_path = output_dir / (
        "pixal_registered_front_side_back_right_board.png"
        if len(rendered) == 4 else f"pixal_registered_{len(rendered)}view_board.png"
    )
    board.save(board_path)

    manifest = {
        "method": (
            "registered_pixal_visible_partial_ranked_multiview"
            if view_selection is not None
            else "registered_pixal_camera_consistent_multiview"
        ),
        "ground_truth_used": False,
        "glb": str(Path(glb_path).resolve()),
        "source_carrier": str(Path(source_carrier_path).resolve()),
        "registered_carrier": str(Path(registered_carrier_path).resolve()),
        "saved_partial_camera": str(Path(camera_path).resolve()),
        "ordered_similarity": transform.tolist(),
        "ordered_similarity_fit_rmse": fit_rmse,
        "resolution": int(resolution),
        "field_of_view_degrees": float(field_of_view_degrees),
        "centre": centre.tolist(),
        "radius": radius,
        "front_direction": front_direction.tolist(),
        "up_direction": saved_up.tolist(),
        "views": [
            {
                "name": item.name,
                "image": str(item.image_path.resolve()),
                "camera_pose": item.camera_pose.tolist(),
                "orbit_yaw_degrees": float(dict(view_yaws)[item.name]),
            }
            for item in rendered
        ],
        "view_selection": view_selection,
        "board": str(board_path.resolve()),
    }
    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def split_equal_view_board(board_path: Path, output_dir: Path, names: Iterable[str]) -> list[Path]:
    """Split a horizontal edited board into equal square view conditions."""
    names = tuple(names)
    image = Image.open(board_path).convert("RGB")
    width, height = image.size
    if not names or width < height * len(names):
        raise ValueError("edited board is not wide enough for the requested square views")
    # Image generation may slightly alter gutter width.  Anchor the crops to
    # evenly spaced panel centres and crop one image-height square per panel.
    centres = (np.arange(len(names), dtype=np.float64) + 0.5) * width / len(names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, centre_x in zip(names, centres):
        left = int(round(centre_x - height * 0.5))
        left = min(max(left, 0), width - height)
        output = output_dir / f"trellis_condition_{name}.png"
        image.crop((left, 0, left + height, height)).save(output)
        paths.append(output)
    return paths


def foreground_bbox(image: Image.Image, white_threshold: int = 245) -> tuple[int, int, int, int]:
    """Return a robust non-white foreground box for a studio-style condition."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.any(array < int(white_threshold), axis=2)
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("condition image has no detectable foreground")
    return int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)


def normalise_edited_view_framing(
    edited_path: Path,
    reference_path: Path,
    output_path: Path,
    white_threshold: int = 245,
) -> dict[str, object]:
    """Restore an edited view's object centre and isotropic image scale.

    Image editing models may improve a local contour while silently zooming or
    translating the whole object.  This deterministic operation retains the
    edited shape but maps its foreground box back to the original view frame.
    """
    edited = Image.open(edited_path).convert("RGB")
    reference = Image.open(reference_path).convert("RGB")
    edit_box = foreground_bbox(edited, white_threshold)
    reference_box = foreground_bbox(reference, white_threshold)
    edit_width, edit_height = edit_box[2] - edit_box[0], edit_box[3] - edit_box[1]
    ref_width, ref_height = reference_box[2] - reference_box[0], reference_box[3] - reference_box[1]
    scale = min(ref_width / max(edit_width, 1), ref_height / max(edit_height, 1))
    resized_size = (
        max(1, int(round(edited.width * scale))),
        max(1, int(round(edited.height * scale))),
    )
    resized = edited.resize(resized_size, Image.Resampling.LANCZOS)
    edit_centre = np.array(
        [(edit_box[0] + edit_box[2]) * 0.5, (edit_box[1] + edit_box[3]) * 0.5],
        dtype=np.float64,
    ) * scale
    ref_centre = np.array(
        [(reference_box[0] + reference_box[2]) * 0.5, (reference_box[1] + reference_box[3]) * 0.5],
        dtype=np.float64,
    )
    offset = np.rint(ref_centre - edit_centre).astype(int)
    canvas = Image.new("RGB", reference.size, "white")
    canvas.paste(resized, tuple(offset))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "edited": str(Path(edited_path).resolve()),
        "reference": str(Path(reference_path).resolve()),
        "output": str(output_path.resolve()),
        "edited_foreground_box": edit_box,
        "reference_foreground_box": reference_box,
        "isotropic_scale": float(scale),
        "paste_offset": offset.tolist(),
    }


def compose_equal_view_board(view_paths: Iterable[Path], output_path: Path) -> Path:
    """Compose view conditions into equal, white-padded horizontal panels."""
    images = [Image.open(path).convert("RGB") for path in view_paths]
    if not images:
        raise ValueError("at least one view is required")
    size = (max(image.width for image in images), max(image.height for image in images))
    board = Image.new("RGB", (size[0] * len(images), size[1]), "white")
    for index, image in enumerate(images):
        offset = (
            index * size[0] + (size[0] - image.width) // 2,
            (size[1] - image.height) // 2,
        )
        board.paste(image, offset)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)
    return output_path


def build_residual_view_refinement_prompt(
    object_type: str,
    target_view: str,
    reference_view: str,
) -> str:
    """Build a category-parameterized prompt without category/part rules."""
    target = target_view.strip().upper()
    reference = reference_view.strip().upper()
    category = object_type.strip()
    if not target or not reference or not category:
        raise ValueError("object_type, target_view and reference_view must be non-empty")
    return f"""Use case: precise-object-edit
Asset type: auxiliary-view condition for image-conditioned 3D regeneration

Image 1 is the only edit target: the current clean {target} view of one {category}.
Image 2 is {target}-view diagnostic evidence for exactly the same camera. Its top panel is the original prior RGB, middle panel is observed partial depth, and bottom panel is the residual overlay: white is already consistent, red is prior-only silhouette, cyan is partial-only evidence, and yellow arrows point from the prior surface toward supported partial geometry.
Image 3 is an accepted {reference} view of the same object after correction. Use it only to understand the corrected three-dimensional extent, connected geometry, identity, and material; do not output the reference view.

Edit only Image 1. Refine its exterior contours and attached surface regions wherever Image 2 shows clear cyan-versus-red separation. Make the supported correction visibly stronger than the current target view, but local and evidence-bound. Preserve every white-overlap and low-residual region. Propagate every moved contour smoothly into connected structure so there are no detached parts, tears, duplicated structures, intersections, or abrupt seams. The correction must represent the same 3-D shape already supported by Image 3.

Keep exactly the same {target} camera, framing, pose, object identity, topology, number of structures, texture, lighting, and pure white background. Do not globally redesign, restyle, shrink, thicken, rotate, or rescale the object. Output one clean {target}-view image only, with no labels, arrows, diagnostic colors, depth maps, borders, text, watermark, or additional objects.
"""


def build_shared_low_frequency_prompt(object_type: str) -> str:
    """Describe the first, view-shared deformation without naming parts."""
    category = object_type.strip()
    if not category:
        raise ValueError("object_type must be non-empty")
    return f"""Use case: precise-object-edit
Asset type: camera-consistent three-view condition for 3D regeneration

Image 1 is the original unedited FRONT / SIDE / BACK strip of the same complete {category}. Image 2 is a four-row-per-view geometric evidence board for the same three cameras.

This is stage 1 of a two-stage edit. Apply only the shared low-frequency 3-D update shown in the first two rows of Image 2. In row 1, dim blue points are the current prior and green points are the shared low-frequency target; arrows point from current to target. Row 2 shows the expected depth after this single shared update. Express this one continuous 3-D change consistently in FRONT, SIDE, and BACK. Smoothly update the global extent and every attached structure as one connected object.

Do not yet apply the magenta remaining-local-residual row. Do not independently repeat the same global change in each view. Preserve identity, topology, material, texture, camera poses, framing, panel order, pure white background, and complete unobserved support. Do not detach, tear, duplicate, remove, or intersect structures.

Output exactly one clean three-panel FRONT / SIDE / BACK strip with no labels, arrows, diagnostic colours, depth maps, text, borders, or watermark.
"""


def build_local_residual_prompt(object_type: str, view_name: str) -> str:
    """Describe the residual-only edit using category-neutral geometry rules."""
    category = object_type.strip()
    view = view_name.strip().upper()
    if not category or not view:
        raise ValueError("object_type and view_name must be non-empty")
    return f"""Use case: precise-object-edit
Asset type: one camera-consistent view for multi-view 3D regeneration

Image 1 is the {view} RGB view of one {category} after a shared low-frequency 3-D deformation has already been applied. Image 2 is the {view} shared/local residual card. Image 3 is the clean FRONT / SIDE / BACK strip after that same shared deformation.

Do not apply the shared deformation again. Use only row 4 of Image 2, {view}-REMAINING LOCAL RESIDUAL: dim green points represent the current stage-1 geometry and bright magenta points represent the remaining partial-supported local targets; sparse arrows indicate direction. Use row 3, {view}-PARTIAL DEPTH TARGET, to resolve depth ordering and local silhouette. Make a conservative local contour correction only where the residual has coherent positive support, while keeping already aligned regions fixed.

Preserve object identity, topology, texture, camera pose, framing, white background, complete unobserved support, and consistency with Image 3. Propagate every accepted correction smoothly through connected geometry. Do not change the global extent again; do not detach, tear, duplicate, remove, or intersect structures.

Output exactly one clean square {view} RGB image with no labels, arrows, diagnostic colours, depth maps, text, panels, borders, or watermark.
"""
