#!/usr/bin/env python3
"""Partial-core local editing of a registered complete Pixal surface.

Same-camera visible correspondences are positional handles.  A sparse graph of
translation-only nodes pulls nearby generated geometry toward the immutable
partial scan, while graph smoothness and compact support keep unobserved/far
geometry fixed.  GT is never loaded and no Pixal points are deleted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import trimesh

import scripts.run_pixal_pca_sim3_ttt_v2 as base
from scripts.run_registration_deformation_fusion_ablation import (
    build_deformation_graph,
    estimate_normals,
    evaluate_raw_partial_gate,
    influence_weights,
)


ROOT = base.ROOT
REGISTRATION_ROOT = (
    ROOT / "gpt_version/_pixal_guarded_unified_registration_v15_20260822")
OUTPUT_ROOT = ROOT / "gpt_version/_pixal_partial_core_local_edit_v16_20260822"


def aggregate_pairs(partial, generated, partial_ids, generated_ids):
    """Use one robust target per visible generated handle."""
    groups = {}
    for partial_id, generated_id in zip(partial_ids, generated_ids):
        groups.setdefault(int(generated_id), []).append(int(partial_id))
    source_ids = np.asarray(sorted(groups), dtype=np.int64)
    targets = np.stack([
        np.median(partial[np.asarray(groups[int(source_id)], dtype=np.int64)], axis=0)
        for source_id in source_ids
    ])
    return source_ids, targets


def deform_points(points, graph, translations, point_nodes=None,
                  point_weights=None):
    points = np.asarray(points, dtype=np.float64)
    normalized = (points - graph["center"]) / graph["diagonal"]
    if point_nodes is None or point_weights is None:
        point_nodes, point_weights = influence_weights(
            normalized, graph["nodes"], graph["point_weights"].shape[1])
    displacement = np.sum(
        point_weights[..., None] * translations[point_nodes], axis=1)
    return points + displacement * graph["diagonal"]


def optimize_local_edit(generated, partial, partial_normals, projector, args):
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    generated_normals = estimate_normals(generated, radius=.025 * diagonal)
    gate, partial_ids, generated_ids = evaluate_raw_partial_gate(
        partial, generated, projector, bbox_diagonal=diagonal,
        partial_normals=partial_normals, complete_normals=generated_normals,
        distance_ratio=args.correspondence_distance_ratio)
    source_ids, targets = aggregate_pairs(
        partial, generated, partial_ids, generated_ids)
    if len(source_ids) < args.min_handles:
        return generated.copy(), {
            "accepted": False, "reason": "insufficient_visible_handles",
            "visible_gate": gate, "handles": int(len(source_ids))}, None

    graph = build_deformation_graph(
        generated, node_spacing_ratio=args.node_spacing_ratio,
        min_nodes=args.min_nodes, max_nodes=args.max_nodes,
        graph_knn=args.graph_knn, influence_k=args.influence_k)
    normalized_targets = (targets - graph["center"]) / graph["diagonal"]
    anchor_points = graph["points"][source_ids]
    node_anchor_distance = cKDTree(anchor_points).query(graph["nodes"], k=1)[0]
    active = node_anchor_distance <= args.support_radius_ratio
    if int(active.sum()) < 4:
        return generated.copy(), {
            "accepted": False, "reason": "insufficient_active_nodes",
            "visible_gate": gate, "handles": int(len(source_ids)),
            "active_nodes": int(active.sum())}, None

    device = torch.device(args.device)
    nodes_t = torch.as_tensor(graph["nodes"], dtype=torch.float32, device=device)
    point_nodes_t = torch.as_tensor(
        graph["point_nodes"][source_ids], dtype=torch.long, device=device)
    point_weights_t = torch.as_tensor(
        graph["point_weights"][source_ids], dtype=torch.float32, device=device)
    source_t = torch.as_tensor(
        graph["points"][source_ids], dtype=torch.float32, device=device)
    target_t = torch.as_tensor(
        normalized_targets, dtype=torch.float32, device=device)
    normals_t = torch.as_tensor(
        partial_normals[partial_ids[:len(source_ids)]],
        dtype=torch.float32, device=device)
    edges_t = torch.as_tensor(graph["edges"], dtype=torch.long, device=device)
    active_t = torch.as_tensor(active, dtype=torch.bool, device=device)
    distance_t = torch.as_tensor(
        node_anchor_distance, dtype=torch.float32, device=device)
    identity_weight = torch.clamp(
        distance_t / max(args.support_radius_ratio, 1e-8), 0., 1.).square()
    translations = torch.nn.Parameter(torch.zeros(
        (len(graph["nodes"]), 3), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([translations], lr=args.learning_rate)
    history = []
    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        selected_translation = torch.sum(
            point_weights_t[..., None] * translations[point_nodes_t], dim=1)
        moved = source_t + selected_translation
        residual = moved - target_t
        distance = torch.linalg.norm(residual, dim=1)
        threshold = torch.quantile(distance.detach(), args.trim_quantile)
        keep = distance <= threshold
        data_point = F.smooth_l1_loss(
            residual[keep], torch.zeros_like(residual[keep]), beta=.006)
        if len(edges_t):
            edge_difference = (
                translations[edges_t[:, 0]] - translations[edges_t[:, 1]])
            edge_active = active_t[edges_t[:, 0]] | active_t[edges_t[:, 1]]
            smooth = edge_difference[edge_active].square().mean()
        else:
            smooth = torch.zeros((), dtype=torch.float32, device=device)
        identity = (identity_weight[:, None] * translations.square()).mean()
        loss = data_point + args.smooth_weight * smooth + args.identity_weight * identity
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            translations[~active_t].zero_()
            norms = torch.linalg.norm(translations, dim=1, keepdim=True).clamp_min(1e-9)
            translations.mul_(torch.clamp(
                args.max_displacement_ratio / norms, max=1.))
        if iteration % 20 == 0 or iteration + 1 == args.iterations:
            history.append({"iteration": iteration,
                            "loss": float(loss.detach().cpu()),
                            "data_point": float(data_point.detach().cpu()),
                            "smooth": float(smooth.detach().cpu()),
                            "identity": float(identity.detach().cpu())})

    translation_np = translations.detach().cpu().numpy()
    deformed = deform_points(
        generated, graph, translation_np, graph["point_nodes"],
        graph["point_weights"])
    before_distance = np.linalg.norm(generated[source_ids] - targets, axis=1)
    after_distance = np.linalg.norm(deformed[source_ids] - targets, axis=1)
    point_anchor_distance = cKDTree(generated[source_ids]).query(generated, k=1)[0]
    displacement = np.linalg.norm(deformed - generated, axis=1)
    far = point_anchor_distance > args.far_identity_radius_ratio * diagonal
    info = {
        "visible_gate": gate,
        "handles": int(len(source_ids)),
        "graph_nodes": int(len(graph["nodes"])),
        "active_nodes": int(active.sum()),
        "before_handle_mean": float(before_distance.mean()),
        "after_handle_mean": float(after_distance.mean()),
        "handle_improvement_ratio": float(
            1. - after_distance.mean() / max(before_distance.mean(), 1e-12)),
        "max_displacement": float(displacement.max()),
        "max_displacement_ratio": float(displacement.max() / diagonal),
        "moved_point_ratio": float((displacement > 1e-5 * diagonal).mean()),
        "far_point_count": int(far.sum()),
        "far_max_displacement_ratio": float(
            displacement[far].max() / diagonal) if far.any() else 0.,
        "history": history,
    }
    return deformed, info, {"graph": graph, "translations": translation_np}


def process(args, sample):
    started = time.perf_counter()
    registration_dir = args.registration_root / sample
    registered_path = registration_dir / f"{sample}_unified_registration_v14_registered_100k.ply"
    generated = base.load_points(registered_path)
    partial = base.load_points(ROOT / "data" / f"{sample}.ply")
    diagonal = max(float(np.linalg.norm(np.ptp(partial, axis=0))), 1e-8)
    partial_normals = estimate_normals(partial, radius=.03 * diagonal)
    camera_dir = args.camera_root / sample
    projector = base.SavedCameraProjector.from_partial(
        partial, camera_dir / "camera.pth", padding=args.padding,
        image_shape=(512, 512), device="cpu")
    before_projection = base.mask_metrics(
        base.render_mask(projector, partial, 512),
        base.render_mask(projector, generated, 512))
    deformed, edit_info, state = optimize_local_edit(
        generated, partial, partial_normals, projector, args)
    after_projection = base.mask_metrics(
        base.render_mask(projector, partial, 512),
        base.render_mask(projector, deformed, 512))
    accepted = bool(
        state is not None
        and edit_info["handle_improvement_ratio"] >= args.min_improvement_ratio
        and edit_info["far_max_displacement_ratio"] <= args.far_max_displacement_ratio
        and after_projection["iou"] >= before_projection["iou"] - args.max_iou_drop
        and after_projection["coverage"] >= before_projection["coverage"] - args.max_coverage_drop
        and after_projection["leakage"] <= before_projection["leakage"] + args.max_leakage_increase)
    if not accepted:
        deformed = generated.copy()
        after_projection = before_projection.copy()
    edit_info["accepted"] = accepted
    edit_info["reason"] = "local_edit_accepted" if accepted else "local_edit_rejected"

    fused = np.concatenate([deformed, partial], axis=0)
    output_dir = args.output_root / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{sample}_partial_core_local_edit_v16"
    paths = {
        "deformed_complete": Path(f"{stem}_deformed_complete_100k.ply"),
        "fused": Path(f"{stem}_fused.ply"),
        "compare": Path(f"{stem}_partial_gray_deformed_red.ply"),
        "mesh": Path(f"{stem}_deformed_mesh.glb"),
    }
    base.write_points(paths["deformed_complete"], deformed)
    base.write_points(paths["fused"], fused)
    base.write_compare(paths["compare"], partial, deformed)
    mesh = trimesh.load(
        args.gpt_root / sample / "pixal3d.glb", force="scene", process=False)
    registration_transform = np.load(
        registration_dir / f"{sample}_unified_registration_v14.npy")
    mesh.apply_transform(registration_transform)
    if accepted and state is not None:
        for geometry in mesh.geometry.values():
            geometry.vertices = deform_points(
                np.asarray(geometry.vertices), state["graph"],
                state["translations"])
    mesh.export(paths["mesh"])
    result = {
        "sample_id": sample,
        "method": "partial_core_compact_support_translation_graph_v16",
        "strict_zero_shot": True,
        "ground_truth_used_for_inference_or_selection": False,
        "fusion_run": True,
        "registered_complete_input": registered_path,
        "partial_input": ROOT / "data" / f"{sample}.ply",
        "before_projection": before_projection,
        "after_projection": after_projection,
        "local_edit": edit_info,
        "fusion_contract": {
            "deformed_complete_points": int(len(deformed)),
            "partial_core_points": int(len(partial)),
            "fused_points": int(len(fused)),
            "all_pixal_points_retained": len(deformed) == len(generated) == 100000,
            "partial_core_exact": bool(np.array_equal(fused[len(deformed):], partial)),
            "points_deleted": False,
            "local_scale_used": False,
            "far_geometry_identity_guard": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "shared_parameters": vars(args),
        "outputs": paths,
    }
    Path(f"{stem}_info.json").write_text(
        json.dumps(base.jsonable(result), indent=2))
    print(json.dumps(base.jsonable(result), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-root", type=Path, default=ROOT / "gpt_version")
    parser.add_argument("--camera-root", type=Path, default=base.CAMERA_ROOT)
    parser.add_argument("--registration-root", type=Path, default=REGISTRATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=["09639", "07136"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--padding", type=float, default=.15)
    parser.add_argument("--correspondence-distance-ratio", type=float, default=.08)
    parser.add_argument("--min-handles", type=int, default=256)
    parser.add_argument("--node-spacing-ratio", type=float, default=.045)
    parser.add_argument("--min-nodes", type=int, default=96)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--graph-knn", type=int, default=6)
    parser.add_argument("--influence-k", type=int, default=4)
    parser.add_argument("--support-radius-ratio", type=float, default=.10)
    parser.add_argument("--far-identity-radius-ratio", type=float, default=.14)
    parser.add_argument("--max-displacement-ratio", type=float, default=.04)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--trim-quantile", type=float, default=.75)
    parser.add_argument("--smooth-weight", type=float, default=30.)
    parser.add_argument("--identity-weight", type=float, default=10.)
    parser.add_argument("--min-improvement-ratio", type=float, default=.08)
    parser.add_argument("--far-max-displacement-ratio", type=float, default=.002)
    parser.add_argument("--max-iou-drop", type=float, default=.02)
    parser.add_argument("--max-coverage-drop", type=float, default=.02)
    parser.add_argument("--max-leakage-increase", type=float, default=.03)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample in args.samples:
        process(args, str(sample))


if __name__ == "__main__":
    main()
