"""Camera-consistent multi-view diagnostics for posterior adaptation.

The helpers turn an initially registered complete prior into auditable
informative RGB/depth observations. They never inspect ground truth or offline
completion metrics. Missing partial pixels are treated as unknown.
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
    sidecar = Path(camera_path).with_name("camera.json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        world_to_camera = np.asarray(
            metadata.get("extrinsic_world_to_camera"), dtype=np.float64,
        )
        if world_to_camera.shape == (4, 4):
            rotation = world_to_camera[:3, :3]
            translation = world_to_camera[:3, 3]
            camera_position = -rotation.T @ translation
            # Open3D pinhole coordinates use +y down, whereas the orbit
            # renderer expects a world-space camera-up vector.
            return camera_position, _normalise(-rotation[1])
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

    # Imported lazily so the posterior solver itself has no OpenGL dependency.
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


def render_registered_pointcloud_multiview(
    *,
    registered_prior_path: Path,
    partial_path: Path,
    camera_path: Path,
    output_dir: Path,
    resolution: int = 768,
    field_of_view_degrees: float = 38.0,
    num_views: int = 4,
) -> dict:
    """Render the actual registered carrier when no rigid GLB map exists."""
    if resolution < 128:
        raise ValueError("resolution must be at least 128")
    prior = load_points(Path(registered_prior_path))
    partial = load_points(Path(partial_path))
    lower, upper = np.quantile(prior, (0.005, 0.995), axis=0)
    centre = 0.5 * (lower + upper)
    radius = max(float(np.quantile(np.linalg.norm(prior - centre, axis=1), 0.995)), 1e-4)
    saved_eye, saved_up = _camera_vectors(Path(camera_path))
    front_direction = _normalise(saved_eye - centre)
    saved_up -= front_direction * float(np.dot(saved_up, front_direction))
    if np.linalg.norm(saved_up) < 1e-5:
        saved_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        saved_up -= front_direction * float(np.dot(saved_up, front_direction))
    saved_up = _normalise(saved_up)
    fov_radians = math.radians(float(field_of_view_degrees))
    camera_distance = radius / max(math.sin(fov_radians * 0.5), 1e-3) * 1.12
    selection = select_informative_orbit_yaws(
        partial=partial,
        centre=centre,
        front_direction=front_direction,
        up_direction=saved_up,
        camera_distance=camera_distance,
        field_of_view_degrees=field_of_view_degrees,
        num_views=int(num_views),
        resolution=min(256, int(resolution)),
    )
    names = ("front", "side", "back", "right") if int(num_views) == 4 else tuple(
        ["front"] + [f"view_{index:02d}" for index in range(2, int(num_views) + 1)]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = []
    images = []
    focal = 0.5 * int(resolution) / math.tan(fov_radians * 0.5)
    for name, decision in zip(names, selection["selected"]):
        yaw = float(decision["yaw_degrees"])
        direction = _normalise(rotate_about_axis(front_direction, saved_up, yaw))
        pose = look_at_pose(centre + direction * camera_distance, centre, saved_up)
        inverse = np.linalg.inv(pose)
        camera = prior @ inverse[:3, :3].T + inverse[:3, 3]
        depth = -camera[:, 2]
        uv = np.column_stack((
            focal * camera[:, 0] / np.maximum(depth, 1e-8) + resolution * 0.5,
            -focal * camera[:, 1] / np.maximum(depth, 1e-8) + resolution * 0.5,
        ))
        rendered_depth, mask, _ = zbuffer_depth_with_indices(
            uv, depth, (int(resolution), int(resolution)), splat_radius=2,
        )
        image = np.full((int(resolution), int(resolution), 3), 255, dtype=np.uint8)
        image[mask] = np.array([55, 55, 55], dtype=np.uint8)
        image_path = output_dir / f"registered_prior_{name}.png"
        Image.fromarray(image, mode="RGB").save(image_path)
        mask_path = output_dir / f"registered_prior_{name}_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        depth_path = output_dir / f"registered_prior_{name}_depth.npy"
        np.save(depth_path, rendered_depth.astype(np.float32))
        depth_image = np.zeros_like(rendered_depth, dtype=np.uint8)
        if np.any(mask):
            near, far = np.quantile(rendered_depth[mask], (0.01, 0.99))
            span = max(float(far - near), 1e-8)
            depth_image[mask] = np.clip(
                255.0 * (1.0 - (rendered_depth[mask] - near) / span), 0.0, 255.0,
            ).astype(np.uint8)
        depth_png = output_dir / f"registered_prior_{name}_depth.png"
        Image.fromarray(depth_image, mode="L").save(depth_png)
        images.append(image_path)
        views.append({
            "name": name,
            "image": str(image_path.resolve()),
            "mask": str(mask_path.resolve()),
            "depth": str(depth_path.resolve()),
            "depth_preview": str(depth_png.resolve()),
            "camera_pose": pose.tolist(),
            "orbit_yaw_degrees": yaw,
        })

    gutter = max(8, int(resolution) // 32)
    board = Image.new(
        "RGB",
        (int(resolution) * len(images) + gutter * (len(images) - 1), int(resolution)),
        "white",
    )
    for index, path in enumerate(images):
        board.paste(Image.open(path).convert("RGB"), (index * (int(resolution) + gutter), 0))
    board_path = output_dir / "registered_prior_front_side_back_right_board.png"
    board.save(board_path)
    manifest = {
        "method": "registered_pointcloud_visible_partial_ranked_multiview",
        "ground_truth_used": False,
        "registered_carrier": str(Path(registered_prior_path).resolve()),
        "saved_partial_camera": str(Path(camera_path).resolve()),
        "resolution": int(resolution),
        "field_of_view_degrees": float(field_of_view_degrees),
        "centre": centre.tolist(),
        "radius": radius,
        "front_direction": front_direction.tolist(),
        "up_direction": saved_up.tolist(),
        "views": views,
        "view_selection": selection,
        "board": str(board_path.resolve()),
    }
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
