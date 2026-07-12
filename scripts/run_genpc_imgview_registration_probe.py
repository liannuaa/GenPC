import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from munch import Munch
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import rotation_6d_to_matrix
from torch import optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optim_registration.diff_obj_pose import (  # noqa: E402
    Completionloss,
    ObjectPoseOptim,
    build_transform,
    compute_loss_function,
    compute_mask_from_rendering,
    get_init_rot,
    load_point_cloud,
)
from reg_xyz import icp_with_scaling  # noqa: E402
from utils.dataUtils import normalize_numpy, numpy2o3d  # noqa: E402


DEFAULT_SAMPLE_DIR = (
    PROJECT_ROOT
    / "workspace"
    / "scansalon_hunyuan3d_omni_car__132_geotransformer_only"
    / "car__132"
)


def render_reference_image_imgview(vert_pos, vert_col, radius, render_size, distance, device):
    eye = torch.tensor([[0.0, float(distance), 0.0]], dtype=torch.float32, device=device)
    at = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    R, T = look_at_view_transform(eye=eye, at=at, up=up, device=device)
    cameras = PerspectiveCameras(
        focal_length=(4.0,),
        R=R,
        T=T,
        image_size=((render_size, render_size),),
        device=device,
    )
    raster_settings = PointsRasterizationSettings(
        image_size=render_size,
        radius=torch.full((vert_pos.shape[0],), radius, dtype=torch.float32, device=device),
    )
    renderer = PulsarPointsRenderer(
        PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    ).to(device)
    result = renderer(
        Pointclouds(points=vert_pos[None], features=vert_col[None]),
        gamma=(1e-2,),
        zfar=(5.0,),
        znear=(1e-4,),
        radius_world=True,
        bg_col=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
    )[0]
    return result, compute_mask_from_rendering(result), R, T


def object_pose_optimization_imgview(
    glb_path,
    point_path,
    *,
    distance,
    radius=0.02,
    lr=0.01,
    iters=400,
    render_size=224,
    device="cuda:0",
    cam_bias_num=8,
    cd_loss_func="infocd",
    scale_init=0.75,
    scale_min=0.2,
    scale_max=0.9,
    scale_reg_weight=0.0,
    pose_mask_weight=0.3,
    pose_cd_weight=8.0,
    pose_cd_inv_weight=0.0,
    pose_cd_direction="complete_to_partial",
):
    device = torch.device(device)
    cdloss = Completionloss(loss_func=cd_loss_func)
    partial_xyz, partial_col = load_point_cloud(
        point_path=point_path, device=device, radius=radius, num_points=8000
    )
    complete_xyz, complete_col = load_point_cloud(
        point_path=glb_path, device=device, radius=radius, num_points=120000
    )
    print(
        "[2d-imgview] "
        f"complete={complete_xyz.shape[0]} partial={partial_xyz.shape[0]} "
        f"camera=+Y distance={float(distance):.4f}"
    )
    ref_img, ref_mask, R_cam, T_cam = render_reference_image_imgview(
        partial_xyz, partial_col, radius, render_size, distance, device
    )

    best_loss = float("inf")
    best_state = None
    for start in range(cam_bias_num):
        rot_init = get_init_rot("y", start * 360.0 / cam_bias_num, device)
        model = ObjectPoseOptim(
            complete_xyz,
            complete_col,
            radius,
            render_size,
            device,
            R_cam,
            T_cam,
            rot_init,
            scale_init=scale_init,
        ).to(device)
        optimizer = optim.Adam(
            [
                {"params": [model.rot_6d], "lr": lr},
                {"params": [model.trans], "lr": lr * 0.2},
                {"params": [model.log_scale], "lr": lr * 0.1},
            ]
        )
        local_best = float("inf")
        local_best_state = None
        for _ in range(iters + 1):
            optimizer.zero_grad()
            result, R_obj, scale, transformed_pts = model(return_pts=True)
            total_loss, *_ = compute_loss_function(
                ref_img,
                result,
                ref_mask,
                partial_xyz,
                transformed_pts,
                cdloss,
                mask_weight=pose_mask_weight,
                cd_weight=pose_cd_weight,
                cd_inv_weight=pose_cd_inv_weight,
                cd_direction=pose_cd_direction,
            )
            ortho_err = torch.norm(R_obj @ R_obj.T - torch.eye(3, device=device))
            loss = (
                total_loss
                + 0.001 * ortho_err
                + scale_reg_weight * (model.log_scale - math.log(scale_init)).pow(2).sum()
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.log_scale.clamp_(math.log(scale_min), math.log(scale_max))
            cur_loss = float(loss.detach().cpu())
            if cur_loss < local_best:
                local_best = cur_loss
                local_best_state = {
                    "rot_6d": model.rot_6d.detach().clone(),
                    "trans": model.trans.detach().clone(),
                    "log_scale": model.log_scale.detach().clone(),
                }
        print(f"[2d-imgview] start={start + 1}/{cam_bias_num} best_loss={local_best:.6f}")
        if local_best < best_loss:
            best_loss = local_best
            best_state = local_best_state

    rot_6d = best_state["rot_6d"]
    trans = best_state["trans"]
    scale = torch.exp(best_state["log_scale"])[0]
    R_obj = rotation_6d_to_matrix(rot_6d[None])[0]
    final_transform = build_transform(R_obj, trans, scale, complete_xyz.mean(0))
    return final_transform.detach().cpu().numpy(), best_loss


def normalize_pcd(pcd):
    xyz = np.asarray(pcd.points)
    colors = (
        np.asarray(pcd.colors)
        if pcd.has_colors()
        else np.ones((len(xyz), 3), dtype=np.float64) * 0.5
    )
    xyz_norm, _, _ = normalize_numpy(xyz, range=0.5)
    return numpy2o3d(xyz_norm, colors)


def nn_stats(source, target):
    d1 = np.asarray(source.compute_point_cloud_distance(target))
    d2 = np.asarray(target.compute_point_cloud_distance(source))
    return {
        "source_to_target_mean": float(np.mean(d1)),
        "source_to_target_p95": float(np.percentile(d1, 95)),
        "target_to_source_mean": float(np.mean(d2)),
        "target_to_source_p95": float(np.percentile(d2, 95)),
    }


def run(args):
    sample_dir = Path(args.sample_dir)
    flag = args.flag
    cfg = Munch.fromDict(yaml.safe_load((PROJECT_ROOT / "configs/config.yaml").read_text()))
    cfg.device = args.device

    partial_path = sample_dir / f"{flag}_hunyuan3d_omni_point_control_processed.ply"
    moge_full_path = sample_dir / f"{flag}_img_sam_cropped_mogev2_points.ply"
    moge_subset_path = sample_dir / f"{flag}_img_sam_cropped_mogev2_voxel1x_subset.ply"
    glb_path = sample_dir / f"{flag}_hunyuan3d_omni.glb"

    partial = o3d.io.read_point_cloud(str(partial_path))
    moge_full_raw = o3d.io.read_point_cloud(str(moge_full_path))
    moge_subset_raw = o3d.io.read_point_cloud(str(moge_subset_path))
    print(
        f"[inputs] partial={len(partial.points)} "
        f"moge_full={len(moge_full_raw.points)} moge_subset={len(moge_subset_raw.points)}"
    )

    pose_transform, pose_loss = object_pose_optimization_imgview(
        str(glb_path),
        str(partial_path),
        distance=float(getattr(cfg, "distance", 2.0)),
        radius=0.02,
        lr=0.01,
        iters=int(getattr(cfg, "reg_pose_iters", 400)),
        render_size=224,
        device=args.device,
        cam_bias_num=int(getattr(cfg, "reg_pose_cam_bias_num", 8)),
        cd_loss_func=getattr(cfg, "reg_cd_loss", "infocd"),
        scale_init=float(getattr(cfg, "reg_pose_scale_init", 0.75)),
        scale_min=float(getattr(cfg, "reg_pose_scale_min", 0.2)),
        scale_max=float(getattr(cfg, "reg_pose_scale_max", 0.9)),
        scale_reg_weight=float(getattr(cfg, "reg_pose_scale_reg_weight", 0.0)),
        pose_mask_weight=float(getattr(cfg, "reg_pose_mask_weight", 0.3)),
        pose_cd_weight=float(getattr(cfg, "reg_pose_cd_weight", 8.0)),
        pose_cd_inv_weight=float(getattr(cfg, "reg_pose_cd_inv_weight", 0.0)),
        pose_cd_direction=getattr(cfg, "reg_pose_cd_direction", "complete_to_partial"),
    )

    diff_transform = np.linalg.inv(pose_transform)
    source = deepcopy(partial)
    source.transform(diff_transform)
    target_full = normalize_pcd(moge_full_raw)
    target_subset = normalize_pcd(moge_subset_raw)

    source_down_base = source.voxel_down_sample(voxel_size=0.03)
    target_down_base = target_subset.voxel_down_sample(voxel_size=0.03)
    scales = np.linspace(
        float(cfg.reg_coarse_scale_range[0]),
        float(cfg.reg_coarse_scale_range[1]),
        int(cfg.reg_coarse_scale_steps),
    )
    best = None
    for scale in scales:
        result = icp_with_scaling(
            deepcopy(source_down_base),
            deepcopy(target_down_base),
            float(scale),
            max_correspondence_distance=0.075,
            init_transform=np.eye(4),
        )
        moved_target = deepcopy(target_down_base)
        moved_target.transform(np.linalg.inv(result.transformation))
        d1 = np.asarray(source_down_base.compute_point_cloud_distance(moved_target))
        d2 = np.asarray(moved_target.compute_point_cloud_distance(source_down_base))
        score = float(np.mean(d1) + 0.25 * np.mean(d2))
        print(
            f"[3d] scale={float(scale):.4f} fitness={result.fitness:.6f} "
            f"rmse={result.inlier_rmse:.6f} score={score:.6f} "
            f"p95={float(np.percentile(d1, 95)):.6f}"
        )
        if best is None or score < best["score"]:
            best = {
                "scale": float(scale),
                "score": score,
                "fitness": float(result.fitness),
                "rmse": float(result.inlier_rmse),
                "transformation": result.transformation,
            }

    inv_coarse = np.linalg.inv(best["transformation"])
    inv_diff = np.linalg.inv(diff_transform)
    final_full = deepcopy(target_full)
    final_subset = deepcopy(target_subset)
    final_full.transform(inv_coarse)
    final_subset.transform(inv_coarse)
    final_full.transform(inv_diff)
    final_subset.transform(inv_diff)

    prefix = sample_dir / f"{flag}_img_sam_cropped_mogev2_genpc2d3d_imgview"
    out_full = prefix.with_name(prefix.name + "_to_processed_partial.ply")
    out_subset = prefix.with_name(prefix.name + "_subset_to_processed_partial.ply")
    out_vis = prefix.with_name(prefix.name + "_red_vs_partial_gray.ply")
    out_info = prefix.with_name(prefix.name + "_info.json")
    out_transform = prefix.with_name(prefix.name + "_transform.npy")

    o3d.io.write_point_cloud(str(out_full), final_full)
    o3d.io.write_point_cloud(str(out_subset), final_subset)
    partial_vis = deepcopy(partial)
    partial_vis.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.55, 0.55, 0.55]]), (len(partial_vis.points), 1))
    )
    red = deepcopy(final_full)
    red.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]]), (len(red.points), 1))
    )
    o3d.io.write_point_cloud(str(out_vis), red + partial_vis)
    np.save(str(out_transform), inv_diff @ inv_coarse)

    info = {
        "method": "genpc_2d_imgview_pose_init_plus_3d_coarse_icp",
        "imgview_camera": {
            "eye": [0.0, float(getattr(cfg, "distance", 2.0)), 0.0],
            "at": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "projection": "xz_from_pos_y",
        },
        "note": "2D pose init uses the image/depth +Y camera and Hunyuan GLB as render proxy because MoGeV2 output is a point cloud.",
        "pose_loss": float(pose_loss),
        "best": {k: v for k, v in best.items() if k != "transformation"},
        "full_stats": nn_stats(partial, final_full),
        "subset_stats": nn_stats(partial, final_subset),
        "paths": {
            "partial": str(partial_path),
            "moge_full": str(moge_full_path),
            "moge_subset": str(moge_subset_path),
            "glb_proxy": str(glb_path),
            "full": str(out_full),
            "subset": str(out_subset),
            "vis": str(out_vis),
            "transform": str(out_transform),
        },
    }
    out_info.write_text(json.dumps(info, indent=2))
    print("[done]")
    print(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--flag", default="car__132")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
