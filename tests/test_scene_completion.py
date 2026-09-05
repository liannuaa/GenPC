from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import trimesh

from src.scene_completion.contracts import load_scene_manifest
from src.scene_completion.instances import extract_scene_instances
from src.scene_completion.io import load_colored_points, write_colored_points
from src.scene_completion.meshes import (
    _camera_depth_mover,
    estimate_scene_table_world_transform,
    export_registered_scene_meshes,
    resolve_scene_camera_collisions,
    scene_anchor_restore_transform,
)
from src.scene_completion.orchestrator import pixal_command, registration_command, write_gpt_action_manifest
from src.scene_completion.scene_camera import SceneMoGeCamera, write_scene_camera_assets
from src.scene_completion.scene_moge import SceneMoGeObservation
from src.saved_camera import SavedCameraProjector


def _fixture_manifest(tmp_path: Path):
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    image_path = tmp_path / "scene.png"
    Image.fromarray(image).save(image_path)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:5, 1:5] = 255
    mask_path = tmp_path / "chair.png"
    Image.fromarray(mask).save(mask_path)
    payload = {
        "schema_version": 1,
        "source_image": "scene.png",
        "instances": [{"id": "chair_0", "label": "wooden chair", "mask": "chair.png", "layer": 2}],
    }
    manifest_path = tmp_path / "instances.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_path, manifest_path


def _observation(image_path: Path) -> SceneMoGeObservation:
    points = np.array(((-3., -3., 1.), (-2., -2., 1.), (-1., -1., 1.), (2., 2., 1.)))
    colors = np.array(((.8, .1, .1), (.7, .2, .2), (.6, .3, .3), (.1, .2, .9)))
    pixels = np.array(((1., 1.), (2., 2.), (3., 3.), (6., 6.)))
    return SceneMoGeObservation(
        image_path=image_path, points=points, colors=colors, pixel_xy=pixels,
        intrinsics=np.array(((1 / 8, 0., .5), (0., 1 / 8, .5), (0., 0., 1.))),
        image_hw=(8, 8), metadata={},
    )


def test_scene_instance_extraction_preserves_shared_moge_coordinates(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    observation = _observation(image)
    summary = extract_scene_instances(
        manifest, observation, output_root=tmp_path / "out", erosion_pixels=0,
        min_mask_pixels=3, min_moge_points=3,
    )
    partial, colors = load_colored_points(tmp_path / "out" / "inputs" / "partial" / "chair_0.ply")
    assert np.allclose(partial, observation.points[:3])
    assert np.allclose(colors, observation.colors[:3], atol=1 / 255)
    assert summary["instances"][0]["coordinate_frame"] == "shared_pixal_moge_scene_camera"
    assert (tmp_path / "out" / "instances" / "chair_0" / "masked_crop.png").is_file()


def test_scene_instance_extraction_accepts_monochrome_gpt_rgb_mask(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    mask_path = tmp_path / "chair.png"
    mono = np.asarray(Image.open(mask_path).convert("L"))
    Image.fromarray(np.repeat(mono[..., None], 3, axis=-1)).save(mask_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    extract_scene_instances(
        manifest, _observation(image), output_root=tmp_path / "out", erosion_pixels=0,
        min_mask_pixels=3, min_moge_points=3,
    )
    assert (tmp_path / "out" / "inputs" / "partial" / "chair_0.ply").is_file()


def test_scene_instance_extraction_assigns_overlapping_pixels_to_foreground_layer(tmp_path: Path):
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    image_path = tmp_path / "scene.png"
    Image.fromarray(image).save(image_path)
    back = np.zeros((8, 8), dtype=np.uint8)
    front = np.zeros((8, 8), dtype=np.uint8)
    back[1:5, 1:5] = 255
    front[3:7, 3:7] = 255
    Image.fromarray(back).save(tmp_path / "back.png")
    Image.fromarray(front).save(tmp_path / "front.png")
    manifest_path = tmp_path / "instances.json"
    manifest_path.write_text(json.dumps({
        "source_image": "scene.png",
        "instances": [
            {"id": "back", "label": "back object", "mask": "back.png", "layer": 0},
            {"id": "front", "label": "front object", "mask": "front.png", "layer": 1},
        ],
    }), encoding="utf-8")
    observation = SceneMoGeObservation(
        image_path=image_path,
        points=np.array(((1., 1., 1.), (3., 3., 1.), (4., 4., 1.), (6., 6., 1.))),
        colors=np.full((4, 3), .5),
        pixel_xy=np.array(((1., 1.), (3., 3.), (4., 4.), (6., 6.))),
        intrinsics=np.eye(3), image_hw=(8, 8), metadata={},
    )
    summary = extract_scene_instances(
        load_scene_manifest(manifest_path, expected_source_image=image_path), observation,
        output_root=tmp_path / "out", erosion_pixels=0, min_mask_pixels=1, min_moge_points=1,
    )
    back_points, _ = load_colored_points(tmp_path / "out" / "inputs" / "partial" / "back.ply")
    front_points, _ = load_colored_points(tmp_path / "out" / "inputs" / "partial" / "front.ply")
    assert len(back_points) == 1
    assert len(front_points) == 3
    assert summary["instances"][0]["occlusion_excluded_pixels"] == 4
    assert summary["unmasked_scene_context_points"] == 0
    assert summary["unmasked_scene_context"] is None
    assert not (tmp_path / "out" / "scene_context_unmasked_visible.ply").exists()


def test_scene_camera_assets_project_scene_moge_partial_without_depth_raster(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    observation = _observation(image)
    extract_scene_instances(
        manifest, observation, output_root=tmp_path / "out", erosion_pixels=0,
        min_mask_pixels=3, min_moge_points=3,
    )
    summary = write_scene_camera_assets(manifest, observation, output_root=tmp_path / "out")
    camera_path = tmp_path / "out" / "inputs" / "camera" / "chair_0" / "camera.pth"
    point_uv = np.load(tmp_path / "out" / "inputs" / "camera" / "chair_0" / "point_uv.npy")
    partial, _ = load_colored_points(tmp_path / "out" / "inputs" / "partial" / "chair_0.ply")
    camera = torch.load(camera_path, weights_only=False)
    assert isinstance(camera, SceneMoGeCamera)
    projector = SavedCameraProjector.from_partial(partial, camera_path, padding=.15, image_shape=(512, 512))
    pixels, _ = projector.project(partial)
    expected = np.c_[point_uv[:, 0], 1.0 - point_uv[:, 1]] * 511.0
    assert np.allclose(pixels, expected, atol=1e-5)
    assert summary["depth_to_semantic_used"] is False


def test_scene_gpt_actions_are_direct_and_instance_scoped(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    root = tmp_path / "out"
    (root / "instances" / "chair_0").mkdir(parents=True)
    action_path = write_gpt_action_manifest(root, manifest)
    action = json.loads(action_path.read_text(encoding="utf-8"))["instances"][0]
    assert action["completion_task"]["output"].endswith("gpt_outputs/chair_0.png")
    assert set(action["completion_task"]["inputs"]) == {"masked_scene_crop"}
    assert "wooden chair" in action["completion_task"]["prompt"]
    assert "POSE LOCK" in action["completion_task"]["prompt"]


def test_scene_registration_command_exposes_independent_worker_count(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    command, _ = registration_command(root=tmp_path / "out", manifest=manifest, sample_workers=2)
    worker_index = command.index("--sample-workers")
    assert command[worker_index + 1] == "2"


def test_scene_registration_command_uses_camera_bridge_only_route(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    command, _ = registration_command(root=tmp_path / "out", manifest=manifest, bridge_only=True)
    assert "--bridge-only" in command


def test_scene_pixal_command_has_explicit_face_budget(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    command, _ = pixal_command(root=tmp_path / "out", manifest=manifest, decimation_target=100_000)
    index = command.index("--decimation-target")
    assert command[index + 1] == "100000"


def test_scene_mesh_export_uses_cumulative_registered_sim3(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    pixal_root, registration_root = tmp_path / "pixal", tmp_path / "registration"
    source = pixal_root / "chair_0" / "pixal3d.glb"
    source.parent.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=(1., 2., 3.))
    trimesh.Scene(mesh).export(source)
    joint = np.eye(4)
    joint[:3, :3] *= 2.
    joint[:3, 3] = (1., 2., 3.)
    residual = np.eye(4)
    residual[:3, 3] = (-.5, .25, 1.)
    (registration_root / "chair_0" / "joint").mkdir(parents=True)
    np.save(registration_root / "chair_0" / "joint" / "two_camera_joint_pixal_to_partial.npy", joint)
    for stage in ("amplified", "wide_tilt", "final"):
        path = registration_root / "chair_0" / stage
        path.mkdir()
        np.save(path / "camera1_amplified_residual.npy", residual if stage == "final" else np.eye(4))
    report = export_registered_scene_meshes(
        manifest, pixal_root=pixal_root, registration_root=registration_root, output_dir=tmp_path / "scene_meshes",
        write_instance_meshes=True,
    )
    registered = tmp_path / "scene_meshes" / "chair_0_registered_mesh.glb"
    exported = trimesh.load(registered, force="scene", process=False)
    bounds = exported.bounds
    assert np.allclose(bounds[0], (-.5, .25, 1.))
    assert np.allclose(bounds[1], (1.5, 4.25, 7.))
    assert Path(report["merged_scene"]).is_file()
    # The final GLB is still a Scene for multi-material storage, but its
    # placement is baked into mesh vertices rather than nested node transforms.
    for node_name in exported.graph.nodes_geometry:
        node_transform, _ = exported.graph[node_name]
        assert np.allclose(node_transform, np.eye(4))


def test_scene_mesh_export_prefers_camera2_to_camera1_bridge(tmp_path: Path):
    image, manifest_path = _fixture_manifest(tmp_path)
    manifest = load_scene_manifest(manifest_path, expected_source_image=image)
    pixal_root, registration_root = tmp_path / "pixal", tmp_path / "registration"
    source = pixal_root / "chair_0" / "pixal3d.glb"
    source.parent.mkdir(parents=True)
    trimesh.Scene(trimesh.creation.box(extents=(1., 2., 3.))).export(source)
    bridge = np.eye(4)
    bridge[:3, :3] *= 2.
    bridge[:3, 3] = (4., -3., 2.)
    bridge_dir = registration_root / "chair_0" / "bridge"
    bridge_dir.mkdir(parents=True)
    np.save(bridge_dir / "two_camera_pixal_moge_pixal_to_partial.npy", bridge)
    report = export_registered_scene_meshes(
        manifest, pixal_root=pixal_root, registration_root=registration_root,
        output_dir=tmp_path / "scene_meshes", write_instance_meshes=True,
    )
    exported = trimesh.load(
        tmp_path / "scene_meshes" / "chair_0_registered_mesh.glb", force="scene", process=False,
    )
    assert np.allclose(exported.bounds[0], (3., -5., -1.))
    assert np.allclose(exported.bounds[1], (5., -1., 5.))
    assert report["instances"][0]["placement_mode"] == "camera2_to_camera1_bridge_direct"


def test_scene_anchor_restore_uses_mask_moge_position_and_bridge_scale():
    prior = np.array(((-2., -1., -3.), (2., -1., -3.), (-2., 1., 3.), (2., 1., 3.)))
    anchor = np.array(((8., 18., 28.), (12., 22., 32.), (8., 22., 28.), (12., 18., 32.)))
    bridge = np.eye(4)
    bridge[:3, :3] *= 7.  # Two-MoGe metric scale must be retained.
    bridge[:3, 3] = (-30., 40., 50.)
    transform, record = scene_anchor_restore_transform(
        prior, anchor, bridge, source_path=Path("prior.ply"), anchor_path=Path("anchor.ply"),
    )
    moved = prior @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose((moved.min(axis=0) + moved.max(axis=0)) * .5, (10., 20., 30.))
    assert np.isclose(np.ptp(moved, axis=0).max(), 7. * np.ptp(prior, axis=0).max())
    assert record["scene_mask_anchor_bbox"]["bbox_center"] == [10.0, 20.0, 30.0]


def test_scene_table_world_transform_levels_dominant_context_plane(tmp_path: Path):
    xs, zs = np.meshgrid(np.linspace(-2., 2., 32), np.linspace(1., 5., 32), indexing="ij")
    points = np.c_[xs.ravel(), np.full(xs.size, 3.), zs.ravel()]
    path = tmp_path / "context.ply"
    write_colored_points(path, points, np.full_like(points, .5))
    transform, record = estimate_scene_table_world_transform(path)
    world = points @ transform[:3, :3].T + transform[:3, 3]
    assert np.abs(world[:, 1]).max() < 1e-5
    assert record["inliers"] >= 512


def test_scene_collision_refinement_pushes_only_farther_mesh_along_camera_depth():
    near = trimesh.creation.box(extents=(1., 1., 1.))
    far = trimesh.creation.box(extents=(.5, .5, .5))
    far.apply_translation((0., 0., .6))
    original_near = near.vertices.copy()
    original_far = far.vertices.copy()
    parts = {"near": [near], "far": [far]}
    report = resolve_scene_camera_collisions(
        parts, away_direction=np.array((0., 0., 1.)), clearance=.01,
    )
    assert report["initial_collision_pairs"] == [["far", "near"]]
    assert report["remaining_collision_pairs"] == []
    assert report["per_instance_depth_shift"]["far"] > 0.
    assert report["per_instance_depth_shift"]["near"] == 0.
    assert np.allclose(near.vertices, original_near)
    assert np.allclose(far.vertices - original_far, (0., 0., report["per_instance_depth_shift"]["far"]))


def test_scene_collision_refinement_has_no_object_scale_shift_cap():
    near = trimesh.creation.box(extents=(2., 2., 2.))
    far = trimesh.creation.box(extents=(.1, .1, .1))
    parts = {"near": [near], "far": [far]}
    report = resolve_scene_camera_collisions(
        parts,
        away_direction=np.array((0., 0., 1.)),
        clearance=.01,
        scene_moge_depths={"near": 1., "far": 2.},
    )
    assert report["remaining_collision_pairs"] == []
    # Escaping the large box requires a translation over ten times the small
    # prior's extent, which used to be rejected by the .5-scale trust region.
    assert report["per_instance_depth_shift"]["far"] > 1.


def test_scene_collision_mover_uses_scene_moge_depth_not_prior_mesh_center():
    near = trimesh.creation.box(extents=(1., 1., 1.))
    far = trimesh.creation.box(extents=(.5, .5, .5))
    near.apply_translation((0., 0., .6))
    # Geometry centre alone would select `near`; the source scene-MoGe
    # observation explicitly says that `far` is behind it in the RGB camera.
    mover = _camera_depth_mover(
        "near", "far", {"near": [near], "far": [far]}, away_direction=np.array((0., 0., 1.)),
        scene_moge_depths={"near": 1., "far": 2.},
    )
    assert mover == "far"
