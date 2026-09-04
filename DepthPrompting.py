"""Frozen saved-view depth rasterisation and Qwen semantic completion.

The project previously carried several experimental view selectors, inpainters,
and image backends. The mainline uses only this deterministic reference view,
OpenCV depth-hole fill, and Qwen-Image-Edit.
"""

from __future__ import annotations

import math

import cv2
import kaolin as kal
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

from tools.qwen_image_edit import QwenImageEdit, resize_stage1_image_for_output
from src.mainline_data import prompt_label
from src.mainline_paths import model_path, sample_dir, sample_file


class DepthPrompting:
    """Generate one saved-view depth image and its Qwen semantic completion."""

    def __init__(self, cfg):
        if str(cfg.depth_projection) != "reference_view":
            raise ValueError("The mainline requires depth_projection='reference_view'.")
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

    def _reference_camera(self):
        eye_direction = np.asarray(self.cfg.reference_eye_direction, dtype=np.float32)
        eye_direction /= max(np.linalg.norm(eye_direction), 1e-8)
        camera_up = np.asarray(self.cfg.reference_camera_up, dtype=np.float32)
        camera_up -= np.dot(camera_up, eye_direction) * eye_direction
        camera_up /= max(np.linalg.norm(camera_up), 1e-8)
        viewpoint = eye_direction * float(self.cfg.distance)
        camera = kal.render.camera.Camera.from_args(
            eye=torch.tensor(viewpoint, dtype=torch.float32),
            at=torch.zeros(3, dtype=torch.float32),
            up=torch.tensor(camera_up, dtype=torch.float32),
            fov=math.pi * float(self.cfg.fovy) / 180.0,
            width=int(self.cfg.cam_res),
            height=int(self.cfg.cam_res),
            device=self.device,
        )
        return camera, viewpoint

    def _project_reference_view(self, points: torch.Tensor, camera):
        transformed = camera.transform(points)
        xy = transformed[:, :2]
        center = (xy.min(dim=0).values + xy.max(dim=0).values) * .5
        scale = (xy.max(dim=0).values - xy.min(dim=0).values).max().clamp_min(1e-8)
        uv = (xy - center) / scale
        uv = uv * (1.0 - 2.0 * float(self.cfg.padding)) + .5
        return uv, transformed[:, 2]

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
        sparse = self._paint(sparse, pixels, colors, int(self.cfg.point_size))
        _ = self._paint(torch.zeros_like(sparse), pixels, normalized_depth[:, None].expand(-1, 3), int(self.cfg.point_size))
        front = self._paint(temporary, pixels, colors, int(self.cfg.point_size) * int(self.cfg.mask_pixel_rate)) != 0
        occupied = sparse != 0
        hole_mask = ((~front).to(torch.int32) * 255 ^ (~occupied).to(torch.int32) * 255).float() / 255.0
        return sparse, hole_mask

    def _save_depth(self, xyz: torch.Tensor, rgb: torch.Tensor, flag: str):
        camera, viewpoint = self._reference_camera()
        uv, depths = self._project_reference_view(xyz, camera)
        pixels = (uv * int(self.cfg.res)).long().clamp(0, int(self.cfg.res) - 1)
        pixels = torch.stack([pixels[:, 1], pixels[:, 0]], dim=1)
        sparse, hole_mask = self._rasterise_depth(pixels, depths, rgb)
        directory = sample_dir(self.cfg, flag)
        directory.mkdir(parents=True, exist_ok=True)
        raw_depth = sample_file(self.cfg, flag, "raw_depth.png")
        depth = sample_file(self.cfg, flag, "depth.png")
        save_image(sparse, raw_depth)
        depth_np = (sparse.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
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
