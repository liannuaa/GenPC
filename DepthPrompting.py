"""Frozen saved-view depth rasterisation and Qwen semantic completion.

The project previously carried several experimental view selectors, inpainters,
and image backends. The mainline uses only this deterministic reference view,
OpenCV depth-hole fill, and Qwen-Image-Edit.
"""

from __future__ import annotations

import math

import cv2
import fpsample
import kaolin as kal
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from torchvision.utils import save_image

from tools.qwen_image_edit import QwenImageEdit, resize_stage1_image_for_output
from src.mainline_data import prompt_label
from src.mainline_paths import model_path, sample_dir, sample_file


class DepthPrompting:
    """Generate one saved-view depth image and its Qwen semantic completion."""

    def __init__(self, cfg):
        if str(cfg.depth_projection) != "view_select":
            raise ValueError("The mainline requires depth_projection='view_select'.")
        if str(cfg.inpainter) != "cv2" or str(cfg.control_model) != "qwen_edit":
            raise ValueError("The mainline requires OpenCV inpainting and Qwen-Image-Edit.")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.depth2image: QwenImageEdit | None = None

    def _load_qwen(self) -> QwenImageEdit:
        if self.depth2image is None:
            self.depth2image = QwenImageEdit(
                device=self.device,
                transformer_path=str(model_path(
                    self.cfg, "qwen_edit_transformer_path",
                    "nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors",
                )),
                pipeline_path=str(model_path(self.cfg, "qwen_edit_pipeline_path", "Qwen-Image-Edit-2511")),
                step=int(self.cfg.qwen_edit_steps),
                generation_size=int(self.cfg.qwen_edit_generate_res),
                true_cfg_scale=float(self.cfg.qwen_edit_true_cfg_scale),
                negative_prompt=str(self.cfg.qwen_edit_negative_prompt),
                cpu_offload=True,
            )
        return self.depth2image

    def close(self) -> None:
        if self.depth2image is not None:
            self.depth2image.close()
        self.depth2image = None

    @staticmethod
    def _up_for_viewpoint(viewpoint: np.ndarray) -> np.ndarray:
        """Match the historical Fibonacci-camera convention exactly."""
        eye = np.asarray(viewpoint, dtype=np.float32)
        gaze = -eye
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if np.allclose(np.cross(gaze, world_up), 0):
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        side = np.cross(gaze, world_up)
        up = np.cross(side, gaze)
        return up / max(np.linalg.norm(up), 1e-8)

    def _camera_for_viewpoint(self, viewpoint: np.ndarray):
        return kal.render.camera.Camera.from_args(
            eye=torch.as_tensor(viewpoint, dtype=torch.float32, device=self.device),
            at=torch.zeros(3, dtype=torch.float32, device=self.device),
            up=torch.as_tensor(self._up_for_viewpoint(viewpoint), dtype=torch.float32, device=self.device),
            fov=math.pi * float(self.cfg.fovy) / 180.0,
            width=int(self.cfg.cam_res),
            height=int(self.cfg.cam_res),
            device=self.device,
        )

    def _fibonacci_viewpoints(self) -> np.ndarray:
        count = int(getattr(self.cfg, "view_num", 256))
        distance = float(self.cfg.distance)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        viewpoints = []
        for index in range(count):
            y = 1.0 - (index / float(count - 1)) * 2.0
            horizontal_radius = math.sqrt(max(1.0 - y * y, 0.0))
            theta = golden_angle * index
            viewpoints.append((
                math.cos(theta) * horizontal_radius * distance,
                y * distance,
                math.sin(theta) * horizontal_radius * distance,
            ))
        return np.asarray(viewpoints, dtype=np.float32)

    @staticmethod
    def _visible_indices(points: torch.Tensor, viewpoints: np.ndarray, radius: float) -> torch.Tensor:
        """Hidden-point removal is the only view scorer; it uses no labels or GT."""
        cloud = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        )
        masks = []
        for viewpoint in viewpoints:
            _, ids = cloud.hidden_point_removal(np.asarray(viewpoint), float(radius))
            mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
            mask[torch.as_tensor(np.asarray(ids), dtype=torch.long, device=points.device)] = True
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def _project_view(self, points: torch.Tensor, camera):
        transformed = camera.transform(points)
        xy = transformed[:, :2]
        center = (xy.min(dim=0).values + xy.max(dim=0).values) * .5
        scale = (xy.max(dim=0).values - xy.min(dim=0).values).max().clamp_min(1e-8)
        uv = (xy - center) / scale
        uv = uv * (1.0 - 2.0 * float(self.cfg.padding)) + .5
        return uv, transformed[:, 2]

    def _select_saved_view(self, points: torch.Tensor):
        """Choose the frozen zero-shot saved view used by the accepted mainline.

        This is deliberately small: coverage on a deterministic Fibonacci sphere,
        followed by the historical opposite-view depth tie break.  It replaces the
        accidental single hard-coded camera introduced during refactoring.
        """
        viewpoints = self._fibonacci_viewpoints()
        sample_count = min(int(getattr(self.cfg, "downsample_num", 10000)), int(points.shape[0]))
        sample_indices = fpsample.fps_sampling(points.detach().cpu().numpy(), sample_count).astype(np.int64)
        sample = points[torch.as_tensor(sample_indices, dtype=torch.long, device=points.device)]
        radius = float(getattr(self.cfg, "removal_radius", 10000.0))
        coverage = self._visible_indices(sample, viewpoints, radius).sum(dim=1)
        best_index = int(torch.argmax(coverage).item())
        chosen = viewpoints[best_index]

        # Preserve the old front/back decision.  It only resolves a view
        # ambiguity from the partial itself, before any image or prior is used.
        opposite = -chosen
        cameras = (self._camera_for_viewpoint(chosen), self._camera_for_viewpoint(opposite))
        depth_sums = []
        for camera, viewpoint in zip(cameras, (chosen, opposite)):
            _, depth = self._project_view(points, camera)
            visible = self._visible_indices(points, np.asarray([viewpoint]), radius)[0]
            depth_sums.append(depth[visible].sum())
        if depth_sums[1] > depth_sums[0]:
            chosen = opposite

        camera = self._camera_for_viewpoint(chosen)
        uv, depth = self._project_view(points, camera)
        visible = self._visible_indices(points, np.asarray([chosen]), radius)[0]
        return camera, chosen, uv, depth, visible

    @staticmethod
    def _paint(img: torch.Tensor, pixels: torch.Tensor, colors: torch.Tensor, radius: int) -> torch.Tensor:
        rows, cols = pixels[:, 0].long(), pixels[:, 1].long()
        offsets = torch.arange(-radius + 1, radius, device=img.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        rows = rows[:, None, None] + dy
        cols = cols[:, None, None] + dx
        valid = (rows >= 0) & (rows < img.shape[1]) & (cols >= 0) & (cols < img.shape[2])
        rows, cols = rows[valid], cols[valid]
        values = colors[:, None, None, :].expand(-1, dy.shape[0], dy.shape[1], -1)[valid]
        img[:, rows, cols] = values.T
        return torch.flip(img, dims=[1])

    def _rasterise_depth(self, pixels: torch.Tensor, depths: torch.Tensor, colors: torch.Tensor):
        resolution = int(self.cfg.res)
        sparse = torch.zeros((3, resolution, resolution), device=self.device)
        temporary = torch.zeros_like(sparse)
        normalized_depth = .1 + .8 * (1.0 - (depths - depths.min()) / (depths.max() - depths.min()).clamp_min(1e-8))
        # Keep the RGB sparse image only for visibility/mask construction.  The
        # semantic model must receive the normalized depth raster below; using
        # RGB here turns the intended depth conditioning into a coloured point
        # cloud and destroys the saved-view geometry.
        sparse = self._paint(sparse, pixels, colors, int(self.cfg.point_size))
        sparse_depth = self._paint(
            torch.zeros_like(sparse),
            pixels,
            normalized_depth[:, None].expand(-1, 3),
            int(self.cfg.point_size),
        )
        front = self._paint(temporary, pixels, colors, int(self.cfg.point_size) * int(self.cfg.mask_pixel_rate)) != 0
        occupied = sparse != 0
        hole_mask = ((~front).to(torch.int32) * 255 ^ (~occupied).to(torch.int32) * 255).float() / 255.0
        return sparse_depth, hole_mask

    def _save_depth(self, xyz: torch.Tensor, rgb: torch.Tensor, flag: str):
        camera, viewpoint, uv, depths, visible = self._select_saved_view(xyz)
        pixels = (uv * int(self.cfg.res)).long().clamp(0, int(self.cfg.res) - 1)
        pixels = torch.stack([pixels[:, 1], pixels[:, 0]], dim=1)
        sparse_depth, hole_mask = self._rasterise_depth(pixels[visible], depths[visible], rgb[visible])
        directory = sample_dir(self.cfg, flag)
        directory.mkdir(parents=True, exist_ok=True)
        raw_depth = sample_file(self.cfg, flag, "raw_depth.png")
        depth = sample_file(self.cfg, flag, "depth.png")
        save_image(sparse_depth, raw_depth)
        depth_np = (sparse_depth.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        mask_np = (hole_mask.permute(1, 2, 0).detach().cpu().numpy()[..., 0] * 255).astype(np.uint8)
        inpainted = cv2.inpaint(depth_np, mask_np, 2, cv2.INPAINT_NS)
        save_image(torch.from_numpy(inpainted).permute(2, 0, 1).float() / 255.0, depth)
        save_image(hole_mask, sample_file(self.cfg, flag, "mask.png"))
        np.save(sample_file(self.cfg, flag, "point_uv.npy"), uv.detach().cpu().numpy())
        np.save(sample_file(self.cfg, flag, "viewpoint.npy"), viewpoint)
        torch.save(camera, sample_file(self.cfg, flag, "camera.pth"))
        return raw_depth

    def getImage(self, xyz: torch.Tensor, flag: str, rgb: torch.Tensor, *, depth_gen: bool = True, img_gen: bool = True) -> None:
        if not depth_gen or not img_gen:
            raise ValueError("The mainline semantic stage always produces both depth and semantic images.")
        raw_depth = self._save_depth(xyz, rgb, flag)
        editor = self._load_qwen()
        input_size = int(self.cfg.depth_image_input_res)
        image = Image.open(raw_depth).convert("RGB").resize((input_size, input_size), Image.Resampling.LANCZOS)
        semantic = editor.generate(image, prompt_label(flag, self.cfg), size=int(self.cfg.generate_res))
        semantic.save(sample_file(self.cfg, flag, "img.png"))
        resize_stage1_image_for_output(editor.last_stage1_image, int(self.cfg.generate_res)).save(
            sample_file(self.cfg, flag, "qwen_edit_stage1.png")
        )
        sample_file(self.cfg, flag, "qwen_edit_prompt.txt").write_text(
            "\n".join([
                "input_image: raw_depth.png",
                f"prompt: {editor.last_prompt}",
                f"negative_prompt: {editor.negative_prompt!r}",
                f"true_cfg_scale: {editor.true_cfg_scale}",
                f"num_inference_steps: {editor.step}",
            ]) + "\n",
            encoding="utf-8",
        )
