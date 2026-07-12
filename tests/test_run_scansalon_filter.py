import csv
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_scansalon


def write_ply(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.zeros((count, 3), dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def test_build_cfg_excludes_denoised_partials_below_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        root.mkdir()
        workspace = Path(tmp) / "workspace"
        pcd_dir = root / "car"
        pcd_dir.mkdir()
        write_ply(pcd_dir / "bad.ply", 1200)
        write_ply(pcd_dir / "good.ply", 1200)

        with open(root / "metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["category", "pcd_filename", "mesh_filename"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "category": "car",
                    "pcd_filename": "car/bad.ply",
                    "mesh_filename": "car/bad.obj",
                }
            )
            writer.writerow(
                {
                    "category": "car",
                    "pcd_filename": "car/good.ply",
                    "mesh_filename": "car/good.obj",
                }
            )

        original_denoise = run_scansalon.denoise_point_cloud

        def fake_denoise(input_path, output_path, *args):
            clean_count = 900 if Path(input_path).stem == "bad" else 1001
            write_ply(output_path, clean_count)
            return 1200, clean_count

        run_scansalon.denoise_point_cloud = fake_denoise
        try:
            cfg = run_scansalon.build_cfg(
                Namespace(
                    config="configs/config.yaml",
                    dataset_root=str(root),
                    workspace=str(workspace),
                    gt_points=100,
                    metric_num_points=100,
                    min_partial_points=1000,
                    control_model="depth_passthrough",
                    qwen_cpu_offload=True,
                    reg_fine_xyz=False,
                    reg_pose_iters=1,
                    reg_pose_cam_bias_num=1,
                    reg_coarse_scale_steps=1,
                    max_samples=None,
                    pipeline_mode="per_sample",
                    save_intermediates=False,
                    skip_existing=False,
                    skip_gt_prepare=True,
                    no_denoise_partials=False,
                    denoise_nb_neighbors=20,
                    denoise_std_ratio=1.5,
                    denoise_radius=0.04,
                    denoise_min_neighbors=8,
                )
            )
        finally:
            run_scansalon.denoise_point_cloud = original_denoise

        assert cfg.sample_ids == ["car__good"]
        assert set(cfg.input_paths) == {"car__good"}


if __name__ == "__main__":
    test_build_cfg_excludes_denoised_partials_below_threshold()
