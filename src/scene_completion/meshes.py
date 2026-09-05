"""Compose registered Pixal meshes in the shared scene-MoGe coordinate frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from src.scene_completion.contracts import SceneManifest
from src.scene_completion.io import load_colored_points, write_colored_points


_REGISTRATION_STAGES = ("amplified", "wide_tilt", "final")
_BRIDGE_TRANSFORM = "two_camera_pixal_moge_pixal_to_partial.npy"
_BRIDGE_PLACEMENT_DIR = "scene_bridge_placement"


def _as_transform(value: object, *, source: Path) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"invalid 4x4 transform: {source}")
    if not np.allclose(transform[3], (0., 0., 0., 1.)):
        raise ValueError(f"non-homogeneous transform: {source}")
    return transform


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected [N,3] points, got {values.shape}")
    return values @ transform[:3, :3].T + transform[:3, 3]


def estimate_scene_table_world_transform(scene_context_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Canonicalize the shared scene-camera frame using its dominant table plane.

    Instance positions remain untouched relative to one another.  This is a
    single global rigid transform from the scene-MoGe camera convention
    (image-down ``y``) to a GLB-friendly table world (``y`` up).  Without it,
    a free-view GLB renderer can look almost along the tabletop and visually
    collapse an otherwise correct camera-frame layout.
    """
    scene_context_path = Path(scene_context_path)
    points, _ = load_colored_points(scene_context_path)
    if len(points) < 512:
        raise ValueError(f"need at least 512 scene-context points for table alignment: {scene_context_path}")
    # Bound RANSAC cost while retaining deterministic scene-wide coverage.
    if len(points) > 250_000:
        points = points[np.linspace(0, len(points) - 1, 250_000, dtype=np.int64)]
    scene_scale = float(np.ptp(points, axis=0).max())
    if not np.isfinite(scene_scale) or scene_scale <= 1e-8:
        raise ValueError(f"degenerate scene context: {scene_context_path}")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.utility.random.seed(0)
    plane, inlier_ids = cloud.segment_plane(
        distance_threshold=max(1e-5, .004 * scene_scale), ransac_n=3, num_iterations=2_000,
    )
    inlier_ids = np.asarray(inlier_ids, dtype=np.int64)
    if len(inlier_ids) < 512:
        raise RuntimeError("dominant scene plane has too few inliers")
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    # MoGe's source-camera y increases downward. Pick the table normal whose
    # positive side points toward image-up, then retain camera-right as the
    # horizontal world x direction for a stable free-view orientation.
    camera_up = np.array((0., -1., 0.))
    if float(normal @ camera_up) < 0.:
        normal = -normal
    world_x = np.array((1., 0., 0.))
    world_x -= normal * float(world_x @ normal)
    if np.linalg.norm(world_x) < 1e-6:
        world_x = np.array((0., 0., 1.))
        world_x -= normal * float(world_x @ normal)
    world_x /= np.linalg.norm(world_x)
    world_z = np.cross(world_x, normal)
    world_z /= np.linalg.norm(world_z)
    rotation = np.stack((world_x, normal, world_z), axis=0)
    if np.linalg.det(rotation) <= 0.:
        raise RuntimeError("failed to construct a right-handed scene world frame")
    plane_center = points[inlier_ids].mean(axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    # The table plane is world y=0. Keep its in-plane origin arbitrary to
    # avoid changing the relative lateral layout that MoGe recovered.
    transform[1, 3] = -float(normal @ plane_center)
    return transform, {
        "method": "scene_moge_dominant_table_plane_to_y_up_world",
        "scene_context": str(scene_context_path.resolve()),
        "input_points": int(len(points)),
        "plane": np.asarray(plane, dtype=np.float64).tolist(),
        "inliers": int(len(inlier_ids)),
        "plane_center": plane_center.tolist(),
        "scene_camera_to_table_world": transform.tolist(),
    }


def _bbox_center_scale(
    points: np.ndarray, *, source: Path, trim_percent: float = 0.,
) -> tuple[np.ndarray, float, dict[str, object]]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError(f"expected non-empty [N,3] layout points from {source}, got {values.shape}")
    if not 0. <= float(trim_percent) < 50.:
        raise ValueError("trim_percent must be in [0, 50)")
    if trim_percent:
        lower, upper = np.percentile(values, (float(trim_percent), 100. - float(trim_percent)), axis=0)
    else:
        lower, upper = values.min(axis=0), values.max(axis=0)
    center = (lower + upper) * .5
    extent = upper - lower
    scale = float(extent.max())
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"degenerate scene-layout anchor: {source}")
    return center, scale, {
        "bbox_min": lower.tolist(),
        "bbox_max": upper.tolist(),
        "bbox_center": center.tolist(),
        "bbox_extent": extent.tolist(),
        "isotropic_extent": scale,
        "trim_percent": float(trim_percent),
    }


def scene_anchor_restore_transform(
    source_prior: np.ndarray,
    scene_anchor: np.ndarray,
    pixal_to_scene_bridge: np.ndarray,
    *,
    source_path: Path,
    anchor_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Restore mask-anchored scene position while preserving bridge Sim(3).

    ``scene_anchor`` is selected directly from the original scene-MoGe cloud
    by the instance mask and supplies the final object *position*.  The
    two-camera bridge supplies its full proper Sim(3) linear map (rotation and
    isotropic scale), because the two MoGe cameras jointly determine metric
    object size. Only its translation is replaced: a crop-normalised semantic
    image should not be allowed to shift a complete object away from the
    scene-MoGe mask that identified it.
    """
    source_center, _, source_bbox = _bbox_center_scale(source_prior, source=source_path)
    # RGB masks accurately identify the scene support, but a few depth rays at
    # their boundary can be arbitrarily wrong.  A fixed two-sided 2% trim is
    # global and mask-only; it prevents those rays from inflating an object's
    # restored scene scale while retaining its position.
    anchor_center, _, anchor_bbox = _bbox_center_scale(
        scene_anchor, source=anchor_path, trim_percent=2.,
    )
    bridge = _as_transform(pixal_to_scene_bridge, source=source_path)
    linear = bridge[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if np.linalg.det(linear) <= 0. or not np.allclose(singular_values, singular_values.mean(), rtol=2e-3, atol=2e-5):
        raise ValueError(f"bridge is not a proper isotropic Sim(3): {source_path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = anchor_center - linear @ source_center
    return transform, {
        "source_prior_bbox": source_bbox,
        "scene_mask_anchor_bbox": anchor_bbox,
        "bridge_linear_rotation_and_scale": linear.tolist(),
        "bridge_isotropic_scale": float(singular_values.mean()),
        "scene_mask_anchor_translation": anchor_center.tolist(),
        "pixal_to_scene": transform.tolist(),
    }


def bridge_pixal_to_scene_transform(registration_dir: Path) -> np.ndarray:
    """Return the camera-2 Pixal-to-camera-1 scene-partial Sim(3).

    Scene MoGe and Pixal-input MoGe are deliberately different cameras.  This
    bridge is the minimal required coordinate conversion: it composes the
    camera-2 native Pixal--MoGe alignment with a pixel-indexed native-MoGe to
    camera-1 scene-partial Sim(3).  Its output is already in the original
    shared scene-MoGe coordinates, including the object location and scale.
    """
    path = Path(registration_dir) / "bridge" / _BRIDGE_TRANSFORM
    if not path.is_file():
        raise FileNotFoundError(path)
    return _as_transform(np.load(path), source=path)


def cumulative_pixal_to_scene_transform(registration_dir: Path) -> np.ndarray:
    """Return the legacy final original-Pixal-to-partial Sim(3)."""
    registration_dir = Path(registration_dir)
    transform_path = registration_dir / "joint" / "two_camera_joint_pixal_to_partial.npy"
    if not transform_path.is_file():
        raise FileNotFoundError(transform_path)
    total = _as_transform(np.load(transform_path), source=transform_path)
    for stage in _REGISTRATION_STAGES:
        residual_path = registration_dir / stage / "camera1_amplified_residual.npy"
        if not residual_path.is_file():
            raise FileNotFoundError(residual_path)
        total = _as_transform(np.load(residual_path), source=residual_path) @ total
    return total


def materialize_scene_bridge_placements(
    manifest: SceneManifest,
    *,
    pixal_root: Path,
    partial_root: Path,
    registration_root: Path,
) -> dict[str, object]:
    """Write bridge-placed complete priors plus partial overlays for audit.

    This makes the global-coordinate contract explicit before GLB composition.
    No points are fused, filtered, recentered, or locally normalised.
    """
    pixal_root = Path(pixal_root)
    partial_root = Path(partial_root)
    registration_root = Path(registration_root)
    records: list[dict[str, object]] = []
    for item in manifest.instances:
        registration_dir = registration_root / item.instance_id
        bridge = bridge_pixal_to_scene_transform(registration_dir)
        source_prior = pixal_root / item.instance_id / "pixal3d_sampled_100k.ply"
        partial_path = partial_root / f"{item.instance_id}.ply"
        source_points, _ = load_colored_points(source_prior)
        partial_points, _ = load_colored_points(partial_path)
        anchor_path = partial_root.parent.parent / "instances" / item.instance_id / "layout_anchor_scene_frame.ply"
        # Existing runs made before the explicit anchor contract can still be
        # inspected, but new scene preparation always writes this file.
        if not anchor_path.is_file():
            anchor_path = partial_path
        anchor_points, _ = load_colored_points(anchor_path)
        transform, layout = scene_anchor_restore_transform(
            source_points, anchor_points, bridge, source_path=source_prior, anchor_path=anchor_path,
        )
        placed_points = _apply_transform(source_points, transform)
        placement_dir = registration_dir / _BRIDGE_PLACEMENT_DIR
        placement_dir.mkdir(parents=True, exist_ok=True)
        placed_path = placement_dir / "pixal_scene_placed_100k.ply"
        overlay_path = placement_dir / "partial_gray_pixal_red.ply"
        write_colored_points(placed_path, placed_points)
        write_colored_points(
            overlay_path,
            np.concatenate((partial_points, placed_points), axis=0),
            np.concatenate((
                np.full_like(partial_points, .62),
                np.tile(np.array((.92, .05, .05), dtype=np.float64), (len(placed_points), 1)),
            ), axis=0),
        )
        placed_lower, placed_upper = placed_points.min(axis=0), placed_points.max(axis=0)
        partial_lower, partial_upper = partial_points.min(axis=0), partial_points.max(axis=0)
        record = {
            "id": item.instance_id,
            "label": item.label,
            "method": "scene_moge_mask_anchor_translation_with_bridge_sim3",
            "coordinate_frame": "shared_pixal_moge_scene_camera",
            "source_prior": str(source_prior.resolve()),
            "scene_partial": str(partial_path.resolve()),
            "scene_mask_anchor": str(anchor_path.resolve()),
            "camera2_to_camera1_bridge": bridge.tolist(),
            "placed_prior": str(placed_path.resolve()),
            "overlay": str(overlay_path.resolve()),
            "placed_scene_bbox_center": ((placed_lower + placed_upper) * .5).tolist(),
            "partial_scene_bbox_center": ((partial_lower + partial_upper) * .5).tolist(),
            "complete_points": int(len(placed_points)),
            "partial_points": int(len(partial_points)),
            **layout,
        }
        (placement_dir / "scene_bridge_placement.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
    report = {
        "method": "scene_moge_mask_anchor_translation_with_bridge_sim3",
        "coordinate_frame": "shared_pixal_moge_scene_camera",
        "instances": records,
        "contract": (
            "Each source-scene MoGe mask supplies the final object centre. The camera-2 to camera-1 bridge supplies "
            "the full proper Sim(3) rotation and isotropic scale; only its translation is replaced by the scene-MoGe "
            "mask anchor. The two MoGe observations are never assumed to share a camera frame."
        ),
    }
    (registration_root / "scene_bridge_placement_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _scene_transform(registration_dir: Path) -> tuple[np.ndarray, str]:
    placement_path = Path(registration_dir) / _BRIDGE_PLACEMENT_DIR / "scene_bridge_placement.json"
    if placement_path.is_file():
        record = json.loads(placement_path.read_text(encoding="utf-8"))
        return _as_transform(record["pixal_to_scene"], source=placement_path), str(record["method"])
    bridge_path = Path(registration_dir) / "bridge" / _BRIDGE_TRANSFORM
    if bridge_path.is_file():
        return bridge_pixal_to_scene_transform(registration_dir), "camera2_to_camera1_bridge_direct"
    return cumulative_pixal_to_scene_transform(registration_dir), "legacy_cumulative_registration_sim3"


def _scene_moge_anchor_depth(registration_dir: Path) -> float | None:
    """Read an instance's source-scene MoGe depth from its saved mask anchor."""
    placement_path = Path(registration_dir) / _BRIDGE_PLACEMENT_DIR / "scene_bridge_placement.json"
    if not placement_path.is_file():
        return None
    record = json.loads(placement_path.read_text(encoding="utf-8"))
    anchor = np.asarray(record.get("scene_mask_anchor_translation"), dtype=np.float64)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        return None
    # This is the original scene-MoGe camera frame, where +Z is away from the
    # RGB camera. It is intentionally independent of the generated mesh.
    return float(anchor[2])


def _load_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(Path(path), force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError(f"expected a non-empty GLB scene: {path}")
    return loaded


def _bake_scene_meshes(scene: trimesh.Scene, *, source: Path) -> list[trimesh.Trimesh]:
    """Bake every Scene-graph transform into textured mesh vertices.

    The final asset remains a ``trimesh.Scene`` so it can hold the individual
    textured meshes, as in the reference scene implementation.  We do not,
    however, leave an instance Sim(3) in a nested GLTF node transform: some
    downstream viewers only inspect geometry-local coordinates and would then
    display all instances at their Pixal-local origins. ``Scene.dump`` applies
    every graph transform while preserving each mesh's texture visual.
    """
    baked = [mesh for mesh in scene.dump(concatenate=False)
             if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) and len(mesh.faces)]
    if not baked:
        raise ValueError(f"no triangle meshes after baking scene graph: {source}")
    return baked


def _collision_manager(mesh_parts: dict[str, list[trimesh.Trimesh]]) -> trimesh.collision.CollisionManager:
    """Build an exact FCL manager with stable per-instance component names."""
    try:
        manager = trimesh.collision.CollisionManager()
    except ValueError as error:
        raise RuntimeError(
            "Scene collision refinement requires python-fcl. Install the project dependencies, "
            "including python-fcl==0.7.0.11."
        ) from error
    for instance_id, meshes in mesh_parts.items():
        for index, mesh in enumerate(meshes):
            manager.add_object(f"{instance_id}:{index}", mesh)
    return manager


def _collision_pairs(mesh_parts: dict[str, list[trimesh.Trimesh]]) -> set[tuple[str, str]]:
    if len(mesh_parts) < 2:
        return set()
    manager = _collision_manager(mesh_parts)
    _, pairs = manager.in_collision_internal(return_names=True)
    result = set()
    for left, right in pairs:
        left_id, right_id = left.rsplit(":", 1)[0], right.rsplit(":", 1)[0]
        if left_id != right_id:
            result.add(tuple(sorted((left_id, right_id))))
    return result


def _object_center_and_scale(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, float, float]:
    vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes], axis=0)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    extent = upper - lower
    return (lower + upper) * .5, float(extent.max()), float(np.prod(extent))


def _camera_depth_mover(
    first: str,
    second: str,
    mesh_parts: dict[str, list[trimesh.Trimesh]],
    *,
    away_direction: np.ndarray,
    scene_moge_depths: dict[str, float] | None = None,
) -> str:
    """Choose the already-farther object; break near-depth ties by size.

    When available, visibility ordering comes directly from the original
    scene-MoGe mask anchors rather than from a generated Pixal mesh. This
    makes the decision invariant to prior-shape errors and object category.
    Sending the farther object farther along the same camera ray preserves
    source-image depth order without lifting it along a support-plane normal.
    """
    first_center, first_scale, first_volume = _object_center_and_scale(mesh_parts[first])
    second_center, second_scale, second_volume = _object_center_and_scale(mesh_parts[second])
    if scene_moge_depths is not None and first in scene_moge_depths and second in scene_moge_depths:
        first_depth = float(scene_moge_depths[first])
        second_depth = float(scene_moge_depths[second])
    else:
        # Legacy/debug fallback when a scene-mask anchor was not materialised.
        first_depth = float(first_center @ away_direction)
        second_depth = float(second_center @ away_direction)
    depth_tie = .02 * min(first_scale, second_scale)
    if abs(first_depth - second_depth) > depth_tie:
        return first if first_depth > second_depth else second
    # Axis-aligned bbox volume is only a deterministic support-size proxy; it
    # never changes a transform or routes by label/category.
    return min(((first_volume, first), (second_volume, second)))[1]


def _is_clear_after_camera_depth_shift(
    instance_id: str,
    mesh_parts: dict[str, list[trimesh.Trimesh]],
    *,
    away_direction: np.ndarray,
    distance: float,
    clearance: float,
    other_manager: trimesh.collision.CollisionManager | None = None,
) -> bool:
    if other_manager is None:
        others = {name: meshes for name, meshes in mesh_parts.items() if name != instance_id}
        if not others:
            return True
        other_manager = _collision_manager(others)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = away_direction * float(distance)
    for mesh in mesh_parts[instance_id]:
        if other_manager.in_collision_single(mesh, transform=transform):
            return False
    return True


def _minimum_camera_depth_shift(
    instance_id: str,
    mesh_parts: dict[str, list[trimesh.Trimesh]],
    *,
    away_direction: np.ndarray,
    clearance: float,
    max_expansions: int = 64,
) -> float | None:
    """Find the smallest positive scene-camera-depth shift clearing contacts.

    There is deliberately no object-scale trust-region cap here.  A generated
    prior can penetrate a much larger neighbouring mesh, in which case a
    half-extent correction is not enough to restore a physically valid scene.
    We instead expand a one-dimensional bracket along the source-camera depth
    ray and then bisect its first collision-free endpoint.  ``max_expansions``
    is only a finite-precision/termination safeguard, not a geometric cap.
    """
    if max_expansions < 1:
        raise ValueError("collision max_expansions must be positive")
    _, scale, _ = _object_center_and_scale(mesh_parts[instance_id])
    others = {name: meshes for name, meshes in mesh_parts.items() if name != instance_id}
    if not others:
        return 0.
    other_manager = _collision_manager(others)
    lower = 0.
    upper = max(float(clearance), scale * 1e-4)
    for _ in range(int(max_expansions)):
        if _is_clear_after_camera_depth_shift(
            instance_id, mesh_parts, away_direction=away_direction, distance=upper, clearance=clearance,
            other_manager=other_manager,
        ):
            break
        lower, upper = upper, upper * 2.
    else:
        return None
    # Sub-millimetre scene-MoGe precision is unnecessary once the configured
    # clearance is added; ten bisection steps keep this CPU-only assembly pass
    # bounded on high-face Pixal meshes.
    for _ in range(10):
        middle = (lower + upper) * .5
        if _is_clear_after_camera_depth_shift(
            instance_id, mesh_parts, away_direction=away_direction, distance=middle, clearance=clearance,
            other_manager=other_manager,
        ):
            upper = middle
        else:
            lower = middle
    # ``upper`` is the collision-free side of the binary bracket. Add a small
    # camera-depth margin to avoid re-contact after GLB serialisation without
    # changing the source-image lateral placement.
    guarded = upper + float(clearance)
    if _is_clear_after_camera_depth_shift(
        instance_id, mesh_parts, away_direction=away_direction, distance=guarded, clearance=clearance,
        other_manager=other_manager,
    ):
        return guarded
    return upper


def resolve_scene_camera_collisions(
    mesh_parts: dict[str, list[trimesh.Trimesh]],
    *,
    away_direction: np.ndarray,
    clearance: float,
    max_iterations: int = 64,
    scene_moge_depths: dict[str, float] | None = None,
) -> dict[str, object]:
    """Resolve mesh penetrations by minimal translation away from scene camera.

    This is a scene-assembly-only rigid adjustment. It preserves every Pixal
    mesh's scale and rotation and retains its source-image lateral location.
    The direction is the original scene-MoGe camera's positive optical axis,
    transformed into the final table-world frame.
    """
    if clearance < 0.:
        raise ValueError("collision clearance must be non-negative")
    direction = np.asarray(away_direction, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 1e-8:
        raise ValueError("away_direction must be a finite non-zero 3-vector")
    direction /= np.linalg.norm(direction)
    if max_iterations < 1:
        raise ValueError("collision max_iterations must be positive")
    if scene_moge_depths is not None:
        invalid = [name for name, depth in scene_moge_depths.items()
                   if name not in mesh_parts or not np.isfinite(float(depth))]
        if invalid:
            raise ValueError(f"invalid scene-MoGe anchor depths for: {invalid}")
    if clearance == 0. or len(mesh_parts) < 2:
        return {
            "enabled": False,
            "method": "disabled",
            "clearance": float(clearance),
            "camera_away_direction": direction.tolist(),
            "initial_collision_pairs": [],
            "remaining_collision_pairs": [],
            "per_instance_depth_shift": {name: 0. for name in mesh_parts},
        }

    # Exact FCL is intentionally queried only for pairs that actually collide.
    # This keeps the correction defined on the same textured geometry exported
    # by the scene assembler and avoids approximation-dependent false negatives.
    initial_pairs = _collision_pairs(mesh_parts)
    shifts = {name: 0. for name in mesh_parts}
    unresolved: set[tuple[str, str]] = set()
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        pairs = _collision_pairs(mesh_parts)
        if not pairs:
            break
        progress = False
        for first, second in sorted(pairs):
            mover = _camera_depth_mover(
                first, second, mesh_parts, away_direction=direction, scene_moge_depths=scene_moge_depths,
            )
            shift = _minimum_camera_depth_shift(
                mover, mesh_parts, away_direction=direction, clearance=float(clearance),
            )
            if shift is None:
                unresolved.add((first, second))
                continue
            for mesh in mesh_parts[mover]:
                mesh.apply_translation(direction * shift)
            shifts[mover] += float(shift)
            progress = True
        if not progress:
            break
    remaining_pairs = _collision_pairs(mesh_parts)
    return {
        "enabled": True,
        "method": "fcl_exact_collision_minimal_scene_camera_depth_translation",
        "clearance": float(clearance),
        "iterations": int(iterations),
        "camera_away_direction": direction.tolist(),
        "depth_order_source": "scene_moge_mask_anchor" if scene_moge_depths else "registered_mesh_center_fallback",
        "initial_collision_pairs": [list(pair) for pair in sorted(initial_pairs)],
        "unresolved_during_iteration": [list(pair) for pair in sorted(unresolved)],
        "remaining_collision_pairs": [list(pair) for pair in sorted(remaining_pairs)],
        "per_instance_depth_shift": shifts,
    }


def export_registered_scene_meshes(
    manifest: SceneManifest,
    *,
    pixal_root: Path,
    registration_root: Path,
    output_dir: Path,
    write_instance_meshes: bool = False,
    scene_context_path: Path | None = None,
    collision_clearance: float = 0.,
) -> dict[str, object]:
    """Apply shared-scene transforms to textured GLBs, then concatenate them.

    The input GLBs retain their Pixal decimation.  Final assembly does not
    re-tessellate, fuse, or prune geometry; by default it also avoids writing
    duplicate per-instance GLBs to keep the scene output compact.
    """
    pixal_root, registration_root, output_dir = map(Path, (pixal_root, registration_root, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if scene_context_path is None:
        scene_to_world = np.eye(4, dtype=np.float64)
        world_record = {"method": "identity_scene_camera_world", "scene_camera_to_table_world": scene_to_world.tolist()}
    else:
        scene_to_world, world_record = estimate_scene_table_world_transform(scene_context_path)
    # Keep a Scene of independent textured meshes instead of concatenating
    # their triangles: each Pixal instance can retain its own PBR material.
    # Their transformations are baked into vertices below, so all final Scene
    # nodes carry identity transforms just like the legacy scene assembler.
    mesh_parts: dict[str, list[trimesh.Trimesh]] = {}
    scene_moge_depths: dict[str, float] = {}
    pending_records: list[dict[str, object]] = []
    for item in manifest.instances:
        source = pixal_root / item.instance_id / "pixal3d.glb"
        camera_transform, placement_mode = _scene_transform(registration_root / item.instance_id)
        transform = scene_to_world @ camera_transform
        scene = _load_scene(source)
        scene.apply_transform(transform)
        baked_meshes = _bake_scene_meshes(scene, source=source)
        mesh_parts[item.instance_id] = baked_meshes
        anchor_depth = _scene_moge_anchor_depth(registration_root / item.instance_id)
        if anchor_depth is not None:
            scene_moge_depths[item.instance_id] = anchor_depth
        pending_records.append({
            "id": item.instance_id,
            "label": item.label,
            "source_mesh": str(source.resolve()),
            "transform": transform.tolist(),
            "placement_mode": placement_mode,
            "coordinate_frame": "table_aligned_scene_world",
            "pixal_to_scene_camera": camera_transform.tolist(),
        })
    # In the final table-world frame, positive scene-camera Z is the direction
    # away from the source RGB camera. Move only along it, retaining lateral
    # mask alignment and every registration rotation/scale.
    camera_away_direction = scene_to_world[:3, :3] @ np.array((0., 0., 1.))
    collision_report = resolve_scene_camera_collisions(
        mesh_parts,
        away_direction=camera_away_direction,
        clearance=float(collision_clearance),
        scene_moge_depths=scene_moge_depths or None,
    )
    shifts = collision_report["per_instance_depth_shift"]
    combined_meshes: list[trimesh.Trimesh] = []
    records: list[dict[str, object]] = []
    for item, record in zip(manifest.instances, pending_records):
        meshes = mesh_parts[item.instance_id]
        registered_path: Path | None = None
        if write_instance_meshes:
            registered_path = output_dir / f"{item.instance_id}_registered_mesh.glb"
            trimesh.Scene(meshes).export(registered_path, extension_webp=True)
        combined_meshes.extend(meshes)
        records.append({
            **record,
            "registered_mesh": str(registered_path.resolve()) if registered_path is not None else None,
            "scene_camera_depth_shift": float(shifts[item.instance_id]),
        })
    merged = output_dir / "completed_scene_registered_meshes.glb"
    combined = trimesh.Scene(combined_meshes)
    combined.export(merged, extension_webp=True)
    report = {
        "method": "direct_textured_pixal_mesh_scene_composition",
        "fusion_run": False,
        "coordinate_frame": "table_aligned_scene_world",
        "instance_meshes_written": bool(write_instance_meshes),
        "scene_world_alignment": world_record,
        "collision_refinement": collision_report,
        "instances": records,
        "merged_scene": str(merged.resolve()),
        "contract": (
            "Every complete textured Pixal mesh uses its camera-2-to-camera-1 bridge transform and one common "
            "scene-camera-to-table-world rigid transform. The final GLB is only a concatenation of already-decimated "
            "meshes; no points or mesh faces are removed."
        ),
    }
    (output_dir / "mesh_scene_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
