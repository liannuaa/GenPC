import os
import time
import math
import json
import cv2
import numpy as np
import torch
import kaolin as kal
import open3d as o3d
from torchvision.utils import save_image
from PIL import Image
from PIL import ImageDraw
import warnings
from utils.dataUtils import getRandomColor, resolve_prompt_label, save_ply_xyzrgb
from utils.camera_utils import calculate_up_vector, create_cameras
from utils.runtime import model_path, sample_dir, sample_file
import fpsample
from diffusers.utils import load_image
warnings.filterwarnings("ignore")


class DepthPrompting:
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)

        if self.cfg.inpainter == "flux":
            from tools.painting_flux1dev import Painting_Flux

            self.inpainter = Painting_Flux(self.device)
        elif self.cfg.inpainter == "DDNM":
            from models.DDNM.ddnm_inpainting import Inpainter

            self.inpainter = Inpainter(self.device)
        elif self.cfg.inpainter == "cv2":
            self.inpainter = cv2.inpaint
        else:
            raise NotImplementedError(
                f"Inpainter {self.cfg.inpainter} not implemented."
            )

        self.cameras, self.viewpoints = create_cameras(
            num_views=self.cfg.view_num,
            distribution=self.cfg.camera_distribution,
            distance=self.cfg.distance,
            fovy=self.cfg.fovy,
            res=self.cfg.cam_res,
            device=self.device,
        )
        # Load image-generation models only after depth/view selection. Semantic
        # view ranking can then temporarily use the GPU without co-residency.
        self.depth2Image = None
        self.semantic_selected_image = None

    def _load_depth2image(self):
        if self.depth2Image is not None or self.cfg.control_model == "depth_passthrough":
            return
        if self.cfg.control_model == "controlnet":
            from tools.controlnet_depth import ControlNet_Depth

            self.depth2Image = ControlNet_Depth(self.device)
        elif self.cfg.control_model == "adapter":
            from tools.adapter_depth import Adapter_Depth

            self.depth2Image = Adapter_Depth(self.device)
        elif self.cfg.control_model == "flux":
            from tools.flux_depth import Flux_depth

            self.depth2Image = Flux_depth(self.device)
        elif self.cfg.control_model == "qwen":
            from tools.qwen_depth import Qwen_depth
            self.depth2Image = Qwen_depth(
                device=self.device,
                transformer_path=str(
                    model_path(
                        self.cfg,
                        "qwen_transformer_path",
                        "nunchaku-qwen-image/svdq-int4_r128-qwen-image-lightningv1.0-4steps.safetensors",
                    )
                ),
                pipeline_path=str(
                    model_path(
                        self.cfg,
                        "qwen_pipeline_path",
                        "Qwen-Image",
                    )
                ),
                controlnet_path=str(
                    model_path(
                        self.cfg,
                        "qwen_controlnet_path",
                        "Qwen-Image-ControlNet-Union",
                    )
                ),
                cpu_offload=bool(getattr(self.cfg, "qwen_cpu_offload", True)),
                cpu_text_encoder=bool(getattr(self.cfg, "qwen_cpu_text_encoder", False)),
            )
        elif self.cfg.control_model == "qwen_edit":
            from tools.qwen_image_edit import QwenImageEdit

            self.depth2Image = QwenImageEdit(
                device=self.device,
                transformer_path=str(
                    model_path(
                        self.cfg,
                        "qwen_edit_transformer_path",
                        "nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors",
                    )
                ),
                pipeline_path=str(
                    model_path(self.cfg, "qwen_edit_pipeline_path", "Qwen-Image-Edit-2511")
                ),
                step=int(getattr(self.cfg, "qwen_edit_steps", 40)),
                true_cfg_scale=float(getattr(self.cfg, "qwen_edit_true_cfg_scale", 4.0)),
                generation_size=int(getattr(self.cfg, "qwen_edit_generate_res", 1024)),
                cpu_offload=bool(getattr(self.cfg, "qwen_cpu_offload", True)),
            )
        else:
            raise NotImplementedError(
                f"Control model {self.cfg.control_model} not implemented."
            )

    def close(self):
        if hasattr(self, "depth2Image") and hasattr(self.depth2Image, "close"):
            self.depth2Image.close()
        self.depth2Image = None

    def save_depth_view_point_cloud(self, flag, xyz, rgb, viewpoint):
        if not bool(getattr(self.cfg, "save_depth_view_point_cloud", True)):
            return

        view_np = np.asarray(viewpoint, dtype=np.float32)
        z_axis = torch.from_numpy(view_np).to(device=xyz.device, dtype=xyz.dtype)
        z_axis = z_axis / z_axis.norm().clamp_min(1e-8)

        up_np = calculate_up_vector(view_np, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        y_axis = torch.from_numpy(up_np).to(device=xyz.device, dtype=xyz.dtype)
        y_axis = y_axis / y_axis.norm().clamp_min(1e-8)

        x_axis = torch.cross(y_axis, z_axis, dim=0)
        x_axis = x_axis / x_axis.norm().clamp_min(1e-8)
        y_axis = torch.cross(z_axis, x_axis, dim=0)
        y_axis = y_axis / y_axis.norm().clamp_min(1e-8)

        screen_points = torch.stack(
            (
                torch.matmul(xyz, x_axis),
                torch.matmul(xyz, y_axis),
                torch.matmul(xyz, z_axis),
            ),
            dim=-1,
        )
        save_ply_xyzrgb(
            screen_points.detach().cpu().numpy(),
            rgb.detach().cpu().numpy(),
            str(sample_file(self.cfg, flag, "depth_view_point_cloud.ply")),
        )

    def getImage(self, xyz, flag, rgb=None, depth_gen=True, img_gen=True):
        print("Stage 1 : Depth Prompting.....")
        start = time.time()
        if depth_gen and rgb is None:
            rgb = torch.tensor(getRandomColor(xyz.shape[0])).float().to(self.device)
        if depth_gen:
            self.getDepth(xyz, flag, rgb)
        depth_input_res = int(getattr(self.cfg, "qwen_depth_input_res", 512))
        self.depth = load_image(str(sample_file(self.cfg, flag, "depth.png"))).resize(
            (depth_input_res, depth_input_res)
        )
        if img_gen:
            print(" Image Generation.....")
            if self.cfg.control_model == "depth_passthrough":
                self.image = self.depth_to_white_bg_image(self.depth, self.cfg.generate_res)
            elif self.semantic_selected_image is not None:
                # The semantic selector already generated this exact candidate at
                # final resolution. Reuse it instead of running Qwen-Image twice.
                self.image = self.semantic_selected_image
            else:
                self._load_depth2image()
                prompt_label = resolve_prompt_label(flag, self.cfg)
                scale_overrides = getattr(self.cfg, "qwen_controlnet_conditioning_scale_overrides", {}) or {}
                controlnet_conditioning_scale = float(
                    scale_overrides.get(
                        str(flag),
                        getattr(self.cfg, "qwen_controlnet_conditioning_scale", 1.0),
                    )
                )
                self.image = self.depth2Image.generate(
                    self.depth,
                    prompt_label,
                    size=self.cfg.generate_res,
                    input_size=depth_input_res,
                    mode="depth",
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                )
            self.image.save(sample_file(self.cfg, flag, "img.png"))
        end = time.time()
        print(f" Take {int(end-start)} seconds")

    def depth_to_white_bg_image(self, depth, size, threshold=4):
        depth_l = depth.convert("L")
        if depth_l.size != (size, size):
            depth_l = depth_l.resize((size, size))
        depth_np = np.asarray(depth_l, dtype=np.uint8)
        mask = depth_np > threshold
        rgb = np.full((size, size, 3), 255, dtype=np.uint8)
        rgb[mask] = np.repeat(depth_np[mask, None], 3, axis=1)
        return Image.fromarray(rgb)

    def viewpoint_select(self, xyz):
        """选择最佳视角，包含启发式防止视角翻转逻辑"""
        # 1. 初始视角选择
        if self.cfg.view_num == 6:
            best_view_idx = 1
        else:
            sample_num = min(int(self.cfg.downsample_num), int(xyz.shape[0]))
            xyz_fps_idx = fpsample.fps_sampling(
                xyz.cpu().numpy(), sample_num
            ).astype(np.int64)
            xyz_fps_idx = torch.from_numpy(xyz_fps_idx).long().to(xyz.device)
            xyz_fps = xyz[xyz_fps_idx]
            print(" Finding best viewpoint...")
            visible_points = self.getVisiblePoints(
                xyz_fps, self.viewpoints, self.cfg.removal_radius
            )
            best_view_idx = torch.argmax(visible_points.sum(dim=1)).item()

        # 2. 启发式防止视角翻转 (Heuristic to prevent viewpoint flip)
        original_viewpoint = self.viewpoints[best_view_idx]
        opposite_viewpoint = -original_viewpoint  # 以(0,0,0)为中心反转

        # 计算原视角和相反视角的深度和可见性
        up_vector = calculate_up_vector(opposite_viewpoint, np.array([0.0, 0.0, 0.0]))
        opposite_camera = kal.render.camera.Camera.from_args(
            eye=torch.tensor(opposite_viewpoint).float(),
            at=torch.tensor([0.0, 0.0, 0.0]).float(),
            up=torch.tensor(up_vector).float(),
            fov=math.pi * self.cfg.fovy / 180,
            width=self.cfg.cam_res,
            height=self.cfg.cam_res,
            device=self.device,
        )

        # 获取两个视角的 UV 和深度数据用于对比
        # 注意：这里只选取对比所需的两个相机
        test_cams = [self.cameras[best_view_idx], opposite_camera]
        _, test_depths, _ = self.getUvs(test_cams, xyz, rescale=self.cfg.rescale, padding=self.cfg.padding)

        # 获取可见点索引进行求和
        vis_mask = self.getVisiblePoints(xyz, [original_viewpoint, opposite_viewpoint], radius=1)
        depth_sum_1 = test_depths[0][vis_mask[0]].sum().item()
        depth_sum_2 = test_depths[1][vis_mask[1]].sum().item()

        # print(f' Original view depth sum: {depth_sum_1:.4f}')
        # print(f' Opposite view depth sum: {depth_sum_2:.4f}')

        if depth_sum_2 > depth_sum_1:
            # print(" Using opposite view (larger depth sum)")
            # 将相反视角相机追加到列表中以维持索引引用
            if isinstance(self.viewpoints, torch.Tensor):
                self.viewpoints = torch.cat([self.viewpoints, torch.tensor(opposite_viewpoint).to(self.viewpoints).unsqueeze(0)], dim=0)
            else:
                self.viewpoints = np.vstack([self.viewpoints, opposite_viewpoint])
            self.cameras.append(opposite_camera)
            best_view_idx = len(self.cameras) - 1

        return best_view_idx

    def _canonical_axes(self, xyz):
        """Estimate a stable horizontal object frame while keeping a known up axis."""
        axis_name = str(getattr(self.cfg, "canonical_up_axis", "y")).lower()
        axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_name, 1)
        up = np.zeros(3, dtype=np.float32)
        up[axis_index] = 1.0

        points = xyz.detach().cpu().numpy().astype(np.float64)
        centered = points - np.median(points, axis=0, keepdims=True)
        horizontal_indices = [idx for idx in range(3) if idx != axis_index]
        horizontal = centered[:, horizontal_indices]

        # Trim distant points before PCA so a few scan outliers cannot rotate the car.
        radius = np.linalg.norm(horizontal, axis=1)
        keep = radius <= np.quantile(radius, 0.98)
        covariance = np.cov(horizontal[keep].T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_2d = eigenvectors[:, int(np.argmax(eigenvalues))]
        length_axis = np.zeros(3, dtype=np.float32)
        length_axis[horizontal_indices] = principal_2d.astype(np.float32)
        length_axis /= max(np.linalg.norm(length_axis), 1e-8)

        # Remove PCA's arbitrary sign to keep runs reproducible.
        dominant = int(np.argmax(np.abs(length_axis)))
        if length_axis[dominant] < 0:
            length_axis = -length_axis
        side_axis = np.cross(up, length_axis)
        side_axis /= max(np.linalg.norm(side_axis), 1e-8)
        return up, length_axis, side_axis

    def _canonical_candidate_cameras(self, xyz):
        up, length_axis, side_axis = self._canonical_axes(xyz)
        elevations = getattr(self.cfg, "canonical_elevations", [12, 18, 24])
        azimuths = getattr(
            self.cfg,
            "canonical_azimuths",
            [30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330],
        )
        distance = float(self.cfg.distance)
        cameras = []
        candidates = []
        for elevation_deg in elevations:
            elevation = math.radians(float(elevation_deg))
            for azimuth_deg in azimuths:
                azimuth = math.radians(float(azimuth_deg))
                horizontal = (
                    math.cos(azimuth) * length_axis
                    + math.sin(azimuth) * side_axis
                )
                direction = (
                    math.cos(elevation) * horizontal + math.sin(elevation) * up
                )
                direction /= max(np.linalg.norm(direction), 1e-8)
                eye = direction * distance

                # Project object-up onto the image plane. This removes camera roll.
                forward = -direction
                camera_up = up - np.dot(up, forward) * forward
                camera_up /= max(np.linalg.norm(camera_up), 1e-8)
                camera = kal.render.camera.Camera.from_args(
                    eye=torch.tensor(eye, dtype=torch.float32),
                    at=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                    up=torch.tensor(camera_up, dtype=torch.float32),
                    fov=math.pi * float(self.cfg.fovy) / 180.0,
                    width=int(self.cfg.cam_res),
                    height=int(self.cfg.cam_res),
                    device=self.device,
                )
                cameras.append(camera)
                candidates.append(
                    {
                        "viewpoint": eye.astype(np.float32),
                        "elevation": float(elevation_deg),
                        "azimuth": float(azimuth_deg),
                    }
                )
        return cameras, candidates

    def _score_canonical_projection(self, point_uv, visible):
        """Score how complete, coherent and conventionally framed a projection is."""
        score_res = int(getattr(self.cfg, "canonical_score_res", 160))
        uv = point_uv[visible].detach().cpu().numpy()
        if len(uv) == 0:
            return {"score": -1e9}

        pixels = np.clip((uv * score_res).astype(np.int32), 0, score_res - 1)
        mask = np.zeros((score_res, score_res), dtype=np.uint8)
        mask[pixels[:, 1], pixels[:, 0]] = 255
        kernel_size = int(getattr(self.cfg, "canonical_score_point_size", 5))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dense = cv2.dilate(mask, kernel, iterations=1)
        dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, kernel, iterations=2)

        ys, xs = np.where(dense > 0)
        if len(xs) == 0:
            return {"score": -1e9}
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        bbox_area = max(width * height, 1)
        silhouette_area = int((dense > 0).sum())

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(dense)
        largest_component = (
            int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
        )
        connectedness = largest_component / max(silhouette_area, 1)

        contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros_like(dense)
        for contour in contours:
            cv2.drawContours(hull_mask, [cv2.convexHull(contour)], -1, 255, -1)
        hull_area = max(int((hull_mask > 0).sum()), 1)
        solidity = silhouette_area / hull_area

        coarse = np.unique(pixels // 4, axis=0).shape[0]
        max_coarse = max((score_res // 4) ** 2, 1)
        grid_coverage = coarse / max_coarse
        visible_ratio = float(visible.float().mean().item())
        bbox_fill = silhouette_area / bbox_area
        aspect = width / max(height, 1)
        aspect_score = math.exp(-abs(math.log(max(aspect, 1e-4) / 1.65)))

        weights = getattr(self.cfg, "canonical_score_weights", {}) or {}
        score = (
            float(weights.get("visible", 0.28)) * visible_ratio
            + float(weights.get("grid", 0.18)) * grid_coverage
            + float(weights.get("connected", 0.18)) * connectedness
            + float(weights.get("solidity", 0.16)) * solidity
            + float(weights.get("fill", 0.08)) * bbox_fill
            + float(weights.get("aspect", 0.12)) * aspect_score
        )
        return {
            "score": float(score),
            "visible_ratio": visible_ratio,
            "grid_coverage": float(grid_coverage),
            "connectedness": float(connectedness),
            "solidity": float(solidity),
            "bbox_fill": float(bbox_fill),
            "aspect": float(aspect),
            "aspect_score": float(aspect_score),
        }

    def _candidate_depth_image(self, point_uv, point_depth, visible, size=224):
        selected_uv = point_uv[visible]
        selected_depth = point_depth[visible]
        if selected_uv.shape[0] == 0:
            return Image.new("RGB", (size, size), "black")
        point_pixels = (selected_uv * int(self.cfg.res)).long()
        point_pixels = torch.stack(
            (point_pixels[:, 1], point_pixels[:, 0]), dim=-1
        ).clip(0, int(self.cfg.res) - 1)
        colors = torch.ones(
            (point_pixels.shape[0], 3), device=point_pixels.device
        )
        _, raw_depth, hole_mask, _ = self.getRawDepth(
            point_pixels,
            selected_depth,
            colors=colors,
            dataset=self.cfg.dataset,
            res=int(self.cfg.res),
            point_size=int(self.cfg.point_size),
            mask_pixel_rate=int(self.cfg.mask_pixel_rate),
        )
        depth_np = (
            raw_depth.permute(1, 2, 0).detach().cpu().numpy() * 255
        ).astype(np.uint8)
        mask_np = (
            hole_mask[0].detach().cpu().numpy() * 255
        ).astype(np.uint8)
        inpainted = cv2.inpaint(depth_np, mask_np, 2, cv2.INPAINT_NS)
        image = Image.fromarray(inpainted).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return image

    def _semantic_rerank_candidates(
        self, flag, candidates, point_uvs, point_depths, visibility
    ):
        sample_dir(self.cfg, flag).mkdir(parents=True, exist_ok=True)
        top_k = min(int(getattr(self.cfg, "semantic_view_top_k", 6)), len(candidates))
        geometry_order = sorted(
            range(len(candidates)),
            key=lambda index: candidates[index]["score"],
            reverse=True,
        )
        # Repeated elevations of one azimuth provide little semantic diversity.
        # Keep only the best geometry candidate per azimuth before VLM ranking.
        ranked_indices = []
        used_azimuths = set()
        for index in geometry_order:
            azimuth = float(candidates[index]["azimuth"])
            if azimuth in used_azimuths:
                continue
            ranked_indices.append(index)
            used_azimuths.add(azimuth)
            if len(ranked_indices) == top_k:
                break

        use_generated = bool(
            getattr(self.cfg, "semantic_view_use_generated_previews", True)
        )
        generated_images = {}
        category = resolve_prompt_label(flag, self.cfg)
        if use_generated:
            if self.cfg.control_model not in {"qwen", "qwen_edit"}:
                raise ValueError(
                    "semantic generated-preview ranking requires qwen or qwen_edit"
                )
            self._load_depth2image()
            preview_size = int(
                getattr(self.cfg, "semantic_view_preview_size", self.cfg.generate_res)
            )
            input_size = int(getattr(self.cfg, "qwen_depth_input_res", 512))
            scale = float(
                getattr(self.cfg, "qwen_controlnet_conditioning_scale", 1.0)
            )
            seed = int(getattr(self.cfg, "semantic_view_preview_seed", 12345))
            print(
                f" Generating {len(ranked_indices)} Qwen view previews "
                f"(ControlNet scale={scale:.2f})..."
            )
            for display_id, candidate_index in enumerate(ranked_indices):
                depth_image = self._candidate_depth_image(
                    point_uvs[candidate_index],
                    point_depths[candidate_index],
                    visibility[candidate_index],
                    size=input_size,
                )
                depth_image.save(
                    sample_file(
                        self.cfg, flag, f"view_candidate_{display_id}_depth.png"
                    )
                )
                generated = self.depth2Image.generate(
                    depth_image,
                    category,
                    size=preview_size,
                    input_size=input_size,
                    mode="depth",
                    controlnet_conditioning_scale=scale,
                    seed=seed,
                )
                generated.save(
                    sample_file(self.cfg, flag, f"view_candidate_{display_id}.png")
                )
                generated_images[display_id] = generated.copy()
            # Qwen3-VL needs the GPU next. The selected preview is retained as PIL.
            self.close()

        tile_size = int(getattr(self.cfg, "semantic_view_tile_size", 224))
        columns = 4
        rows = math.ceil(top_k / columns)
        label_height = 28
        sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        metadata = []
        for display_id, candidate_index in enumerate(ranked_indices):
            if use_generated:
                tile = generated_images[display_id].resize((tile_size, tile_size))
            else:
                tile = self._candidate_depth_image(
                    point_uvs[candidate_index],
                    point_depths[candidate_index],
                    visibility[candidate_index],
                    size=tile_size,
                )
            x = (display_id % columns) * tile_size
            y = (display_id // columns) * (tile_size + label_height)
            sheet.paste(tile, (x, y + label_height))
            draw.rectangle((x, y, x + tile_size, y + label_height), fill="white")
            draw.text((x + 8, y + 5), f"ID {display_id}", fill="black")
            metadata.append(
                {
                    "id": display_id,
                    "candidate_index": candidate_index,
                    "azimuth": candidates[candidate_index]["azimuth"],
                    "elevation": candidates[candidate_index]["elevation"],
                }
            )
        sheet_path = sample_file(self.cfg, flag, "view_candidates.png")
        sheet.save(sheet_path)

        from tools.qwen3_vl_view_selector import select_canonical_view

        selector_path = getattr(
            self.cfg,
            "semantic_view_model_path",
            "/opt/data/private/cr/resources/Qwen3-VL-8B-Instruct",
        )
        selected_id, response = select_canonical_view(
            sheet,
            metadata,
            selector_path,
            category,
            candidate_kind="generated reconstructions" if use_generated else "depth maps",
        )
        response_path = sample_file(self.cfg, flag, "view_semantic_selection.txt")
        with open(response_path, "w") as handle:
            handle.write(response)
        if use_generated:
            self.semantic_selected_image = generated_images[selected_id]
        return metadata[selected_id]["candidate_index"], response

    def canonical_viewpoint_select(self, xyz, flag):
        cameras, candidates = self._canonical_candidate_cameras(xyz)
        print(f" Scoring {len(cameras)} upright canonical viewpoints...")
        point_uvs, point_depths, _ = self.getUvs(
            cameras, xyz, rescale=self.cfg.rescale, padding=self.cfg.padding
        )
        viewpoints = [candidate["viewpoint"] for candidate in candidates]
        if str(getattr(self.cfg, "canonical_visibility", "hpr")).lower() == "all":
            visibility = torch.ones(
                (len(viewpoints), xyz.shape[0]),
                dtype=torch.bool,
                device=xyz.device,
            )
        else:
            visibility = self.getVisiblePoints(
                xyz, viewpoints, self.cfg.removal_radius
            )

        for index, candidate in enumerate(candidates):
            candidate.update(
                self._score_canonical_projection(point_uvs[index], visibility[index])
            )
        geometry_best_index = max(
            range(len(candidates)), key=lambda idx: candidates[idx]["score"]
        )
        best_index = geometry_best_index
        if bool(getattr(self.cfg, "semantic_view_selector", False)):
            try:
                best_index, semantic_response = self._semantic_rerank_candidates(
                    flag, candidates, point_uvs, point_depths, visibility
                )
                print(f" Qwen3-VL semantic view selection: {semantic_response.strip()}")
            except Exception as error:
                print(
                    " Qwen3-VL view selection failed; using geometry fallback: "
                    f"{error}"
                )
        selected = candidates[best_index]
        print(
            " Selected canonical view: "
            f"azimuth={selected['azimuth']:.0f}, "
            f"elevation={selected['elevation']:.0f}, score={selected['score']:.4f}"
        )

        output = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            serialized = dict(candidate)
            serialized["viewpoint"] = candidate["viewpoint"].tolist()
            output.append(serialized)
        sample_dir(self.cfg, flag).mkdir(parents=True, exist_ok=True)
        with open(sample_file(self.cfg, flag, "view_scores.json"), "w") as handle:
            json.dump(output, handle, indent=2)
        return (
            cameras[best_index],
            selected["viewpoint"],
            point_uvs[best_index],
            point_depths[best_index],
            visibility[best_index],
        )

    def getDepth(self, xyz, flag, rgb):
        with torch.no_grad():
            projection = getattr(self.cfg, "depth_projection", "view_select")
            if projection == "xz_from_pos_y":
                selected_point_uvs, selected_point_depths, visible_point_idx = (
                    self.project_xz_from_pos_y(xyz)
                )
                self.view = np.array([0.0, float(self.cfg.distance), 0.0], dtype=np.float32)
                self.cam = self.create_pos_y_camera()
            elif projection == "reference_view":
                (
                    self.cam,
                    self.view,
                    selected_point_uvs,
                    selected_point_depths,
                    visible_point_idx,
                ) = self.project_reference_view(xyz)
            elif projection == "canonical_view":
                (
                    self.cam,
                    self.view,
                    selected_point_uvs,
                    selected_point_depths,
                    visible_point_idx,
                ) = self.canonical_viewpoint_select(xyz, flag)
            else:
                best_view_idx = self.viewpoint_select(xyz)

                self.view = self.viewpoints[best_view_idx]
                self.cam = self.cameras[best_view_idx]

                # 渲染选中的视角
                point_uvs, point_depths, _ = self.getUvs([self.cam], xyz, rescale=self.cfg.rescale, padding=self.cfg.padding)
                selected_point_uvs = point_uvs[0]
                selected_point_depths = point_depths[0]

                visible_point_idx = self.getVisiblePoints(xyz, [self.view], self.cfg.removal_radius)[0]

            # 渲染选中的视角
            point_pixels = (selected_point_uvs * self.cfg.res).long()
            point_pixels = torch.cat(
                (point_pixels[:, 1].unsqueeze(-1), point_pixels[:, 0].unsqueeze(-1)),
                dim=-1,
            )
            point_pixels = point_pixels.clip(0, self.cfg.res - 1)

            # depth and mask
            sparse_img, raw_depth, hole_mask1, hole_mask2 = self.getRawDepth(
                point_pixels[visible_point_idx],
                selected_point_depths[visible_point_idx],
                colors=rgb[visible_point_idx],
                dataset=self.cfg.dataset,
                res=self.cfg.res,
                point_size=self.cfg.point_size,
                mask_pixel_rate=self.cfg.mask_pixel_rate,
            )

            # inpainting [3,h,w]
            sample_dir(self.cfg, flag).mkdir(parents=True, exist_ok=True)
            raw_depth_path = sample_file(self.cfg, flag, "raw_depth.png")
            mask_path = sample_file(self.cfg, flag, "mask.png")
            depth_path = sample_file(self.cfg, flag, "depth.png")
            save_image(raw_depth, raw_depth_path)
            print(" Inpainting depth...")
            if self.cfg.inpainter == "flux":
                save_image(hole_mask1, mask_path)
                depth = self.inpainter.paint(
                    load_image(str(raw_depth_path)),
                    load_image(str(mask_path)),
                    prompt="complete the depth map. ",
                    size=self.cfg.res,
                )
                depth.save(depth_path)
            elif self.cfg.inpainter == "DDNM":
                save_image(hole_mask2, mask_path)
                depth = self.inpainter.inpaint(
                    masked_imgs=raw_depth.permute(1, 2, 0).unsqueeze(0),
                    masks=hole_mask2.permute(1, 2, 0).unsqueeze(0),
                )[0]
                save_image(depth, depth_path)
            elif self.cfg.inpainter == "cv2":
                depth_np = (raw_depth.permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
                mask_np = (
                    hole_mask1.permute(1, 2, 0).cpu().numpy()[:, :, 0] * 255
                ).astype(np.uint8)
                inpainted_depth = self.inpainter(depth_np, mask_np, 2, cv2.INPAINT_NS)
                inpainted_depth = (
                    torch.from_numpy(inpainted_depth).permute(2, 0, 1).float() / 255.0
                )
                save_image(inpainted_depth, depth_path)
                save_image(hole_mask1, mask_path)

            self.point_uv = selected_point_uvs
            np.save(
                sample_file(self.cfg, flag, "point_uv.npy"),
                self.point_uv.detach().cpu().numpy(),
            )
            np.save(sample_file(self.cfg, flag, "viewpoint.npy"), self.view)
            torch.save(self.cam, sample_file(self.cfg, flag, "camera.pth"))
            self.save_depth_view_point_cloud(flag, xyz, rgb, self.view)


    def create_pos_y_camera(self):
        return kal.render.camera.Camera.from_args(
            eye=torch.tensor([0.0, float(self.cfg.distance), 0.0]).float(),
            at=torch.tensor([0.0, 0.0, 0.0]).float(),
            up=torch.tensor([0.0, 0.0, 1.0]).float(),
            fov=math.pi * self.cfg.fovy / 180,
            width=self.cfg.cam_res,
            height=self.cfg.cam_res,
            device=self.device,
        )

    def project_reference_view(self, points):
        eye_direction = np.asarray(
            getattr(
                self.cfg,
                "reference_eye_direction",
                [-0.14270581, -0.75454755, -0.64054122],
            ),
            dtype=np.float32,
        )
        eye_direction /= max(np.linalg.norm(eye_direction), 1e-8)
        camera_up = np.asarray(
            getattr(
                self.cfg,
                "reference_camera_up",
                [0.39445910, -0.63690607, 0.66238408],
            ),
            dtype=np.float32,
        )
        camera_up -= np.dot(camera_up, eye_direction) * eye_direction
        camera_up /= max(np.linalg.norm(camera_up), 1e-8)
        viewpoint = eye_direction * float(self.cfg.distance)
        camera = kal.render.camera.Camera.from_args(
            eye=torch.tensor(viewpoint, dtype=torch.float32),
            at=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            up=torch.tensor(camera_up, dtype=torch.float32),
            fov=math.pi * float(self.cfg.fovy) / 180.0,
            width=int(self.cfg.cam_res),
            height=int(self.cfg.cam_res),
            device=self.device,
        )
        point_uvs, point_depths, _ = self.getUvs(
            [camera], points, rescale=self.cfg.rescale, padding=self.cfg.padding
        )
        visible = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        return camera, viewpoint, point_uvs[0], point_depths[0], visible

    def project_xz_from_pos_y(self, points):
        padding = float(self.cfg.padding)
        xz = points[:, [0, 2]]
        xz_min = xz.min(dim=0).values
        xz_max = xz.max(dim=0).values
        xz_center = (xz_min + xz_max) * 0.5
        xz_scale = (xz_max - xz_min).max().clamp_min(1e-8)
        point_uvs = (xz - xz_center) / xz_scale
        point_uvs = point_uvs * (1 - 2 * padding) + 0.5
        point_depths = -points[:, 1]
        viewpoint = np.array([0.0, float(self.cfg.distance), 0.0], dtype=np.float32)
        visible_point_idx = self.getVisiblePoints(
            points, [viewpoint], self.cfg.removal_radius
        )[0]
        return point_uvs, point_depths, visible_point_idx

    def getUvs(self, cams, points, rescale=True, padding=0.15):
        transformed_points = torch.zeros(
            (len(cams), points.shape[0], 3), device=self.device
        )
        # 点云压缩成某个视角下的一个平面
        for i, cam in enumerate(cams):
            transformed_points[i] = cam.transform(points)  # [point_num,3]
        if rescale:
            vertice_uvs = transformed_points[:, :, :2]
            ori_vertice_uvs_min = vertice_uvs.min(1)[0]  # cam_num,2
            ori_vertice_uvs_max = vertice_uvs.max(1)[0]  # cam_num,2
            ori_vertice_uvs_min = ori_vertice_uvs_min.unsqueeze(1)  # cam_num,1,2
            ori_vertice_uvs_max = ori_vertice_uvs_max.unsqueeze(1)  # cam_num,1,2
            uv_centers = (ori_vertice_uvs_min + ori_vertice_uvs_max) / 2  # cam_num,1,2
            uv_scales = (
                (ori_vertice_uvs_max - ori_vertice_uvs_min).max(2)[0].unsqueeze(2)
            )  # cam_num,1,2
            point_uvs = transformed_points[..., :2]
            point_uvs = (
                point_uvs - uv_centers
            ) / uv_scales  # now all between -0.5, 0.5
            point_uvs = point_uvs * (1 - 2 * padding)  # now all between -0.45, 0.45
            point_uvs = point_uvs + 0.5  # now all between 0.05, 0.95
            point_depths = transformed_points[:, :, 2]  # # [num_cameras,point_num]
        else:
            point_uvs = transformed_points[..., :2]
            point_uvs = (point_uvs + 1) * 0.5  #
            point_depths = transformed_points[:, :, 2]
        return (
            point_uvs,
            point_depths,
            transformed_points,
        )

    def getVisiblePoints(
        self,
        points,
        viewpoints=None,
        radius=None,
    ):
        point_visibility = torch.zeros(
            (len(viewpoints), points.shape[0]), device=points.device
        ).bool()
        for i_cam in range(len(viewpoints)):
            pcd = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(points.cpu().numpy())
            )
            o3d_camera = np.array(viewpoints[i_cam])
            _, pt_map = pcd.hidden_point_removal(o3d_camera, radius)
            visible_point_ids = np.array(pt_map)
            point_visibility[i_cam, visible_point_ids] = True
        return point_visibility

    def paintPixels(self, img, pixel_coords, pixel_colors, point_size):
        """
        :param img: torch tensor of shape [3,res,res]
        :param pixel_coords: [N,2]
        :param pixel_colors: [N,3]
        :param point_size: paint not only the given pixels, but for each pixel, paint its neighbors whose distance to it is smaller than (point_size-1).
        :return:
        """
        N = pixel_coords.shape[0]
        C = img.shape[0]
        if not torch.is_tensor(pixel_colors):
            pixel_colors = pixel_colors * torch.ones((N, C), device=img.device).float()
        if point_size == 1:
            img[:, pixel_coords[:, 0], pixel_coords[:, 1]] = pixel_colors.permute(1, 0)
        else:
            pixel_coords = pixel_coords.long()
            if point_size > 1:
                xx, yy = torch.meshgrid(
                    torch.arange(-point_size + 1, point_size, 1),
                    torch.arange(-point_size + 1, point_size, 1),
                )
                grid = (
                    torch.stack((xx, yy), 2)
                    .view(point_size * 2 - 1, point_size * 2 - 1, 2)
                    .to(img.device)
                )  # grid_res,grid_res,2
                grid_res = grid.shape[0]
                grid = grid + pixel_coords.unsqueeze(1).unsqueeze(
                    1
                )  # [N,grid_res,grid_res,2]
                pixel_colors = (
                    pixel_colors.unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, grid_res, grid_res, 1)
                )  # [N,3] -> [N,grid_res,grid_res,3]
                mask = (
                    (grid[:, :, :, 0] >= 0)
                    & (grid[:, :, :, 0] < img.shape[1])
                    & (grid[:, :, :, 1] >= 0)
                    & (grid[:, :, :, 1] < img.shape[2])
                )  # [N,grid_res,grid_res],
                grid = grid[mask]  # [final_pixel_num,2】
                pixel_colors = pixel_colors[mask]  # [final_pixel_num,3】
                indices = grid.long()
                img[:, indices[:, 0], indices[:, 1]] = pixel_colors.permute(
                    1, 0
                )  # .unsqueeze(1).repeat(1, grid.shape[0], 1)
        return torch.flip(img, dims=[1])

    def getRawDepth(
        self,
        point_pixels,
        point_depth,
        dataset,
        colors=None,
        res=512,
        point_size=1,
        mask_pixel_rate=3,
    ):
        """
        :param point_pixels: [point_num,2]
        :param point_depth: [point_num]
        :param colors: [3, point_num]
        :param visible_point: [num_cameras,point_num]
        """
        sparse_img, all_img, sparse_depth, all_depth, all_temp = [
            torch.zeros((3, self.cfg.res, self.cfg.res), device=self.device)
            for _ in range(5)
        ]
        # depth
        visible_point_depth = 0.1 + 0.8 * (
            1
            - (point_depth - point_depth.min())
            / (point_depth.max() - point_depth.min())
        ).unsqueeze(1).expand(-1, 3)
        sparse_img = self.paintPixels(
            sparse_img, point_pixels, colors, point_size=point_size
        )
        sparse_depth = self.paintPixels(
            sparse_depth, point_pixels, visible_point_depth, point_size=point_size
        )
        # all_depth = self.paint_pixels(all_depth, point_pixels, depth_normalized.unsqueeze(1).expand(-1, 3),point_size=point_size)

        # mask
        all_front_mask = (
            self.paintPixels(
                all_temp, point_pixels, colors, point_size=point_size * mask_pixel_rate
            )
            != 0
        ).float()
        all_back_mask = 1 - all_front_mask
        front_mask = (sparse_img != 0).float()
        back_mask = 1 - front_mask
        hole_mask1 = (
            (all_back_mask * 255).int() ^ (back_mask * 255).int()
        ).float() / 255
        hole_mask2 = (
            (all_front_mask * 255).int() ^ (back_mask * 255).int()
        ).float() / 255
        return sparse_img, sparse_depth, hole_mask1, hole_mask2
