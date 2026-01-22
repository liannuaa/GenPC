import time
from torchvision.utils import save_image
from PIL import Image
import warnings
from munch import Munch
import yaml
from utils.dataUtils import *
from utils.camera_utils import *
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
                self.device,
                transformer_path="/root/shared-nvme/genpc_open/models/nunchaku-qwen-image-edit-2509/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors",
                pipeline_path="/root/shared-nvme/genpc_open/models/qwen-image-edit-2509",
            )
        else:
            raise NotImplementedError(
                f"Control model {self.cfg.control_model} not implemented."
            )

    def getImage(self, xyz, flag, rgb=None, depth_gen=True, img_gen=True):
        print("Stage 1 : Depth Prompting.....")
        start = time.time()
        if rgb is None:
            rgb = torch.tensor(getRandomColor(xyz.shape[0])).float().to(self.device)
        if depth_gen:
            self.getDepth(xyz, flag, rgb)
        self.depth = load_image(f"{self.cfg.output_path}/{flag}/depth.png")
        self.depth.resize((self.cfg.generate_res, self.cfg.generate_res))
        if img_gen:
            print(" Image Generation.....")
            self.image = self.depth2Image.generate(
                self.depth, getCategory(flag), size=self.cfg.generate_res
            )
            self.image.save(f"{self.cfg.output_path}/{flag}/img.png")
        end = time.time()
        print(f" Take {int(end-start)} seconds")

    def viewpoint_select(self, xyz):
        xyz_fps_idx = fpsample.fps_sampling(
            xyz.cpu().numpy(), self.cfg.downsample_num
        ).astype(np.int64)
        xyz_fps_idx = torch.from_numpy(xyz_fps_idx).long().to(xyz.device)
        xyz_fps = xyz[xyz_fps_idx]
        print(" Finding best viewpoint...")
        visible_points = self.getVisiblePoints(
            xyz_fps, self.viewpoints, self.cfg.removal_radius
        )
        best_view_idx = torch.argmax(visible_points.sum(dim=1))
        return best_view_idx

    def getDepth(self, xyz, flag, rgb):
        with torch.no_grad():
            point_uvs, point_depths, transformed_points = self.getUvs(
                self.cameras, xyz, rescale=self.cfg.rescale, padding=self.cfg.padding
            )
            if self.cfg.view_num == 6:
                best_view_idx = 1
            else:
                best_view_idx = self.viewpoint_select(xyz)

            # 创建相反视角的相机
            original_viewpoint = self.viewpoints[best_view_idx]
            opposite_viewpoint = -original_viewpoint  # 以(0,0,0)为中心反转

            # 计算up向量
            from utils.camera_utils import calculate_up_vector

            up_vector = calculate_up_vector(
                opposite_viewpoint, np.array([0.0, 0.0, 0.0])
            )

            # 创建相反视角的相机
            import math

            opposite_camera = kal.render.camera.Camera.from_args(
                eye=torch.tensor(opposite_viewpoint).float(),
                at=torch.tensor([0.0, 0.0, 0.0]).float(),
                up=torch.tensor(up_vector).float(),
                fov=math.pi * self.cfg.fovy / 180,
                width=self.cfg.cam_res,
                height=self.cfg.cam_res,
                device=self.device,
            )

            # 计算相反视角的UV和深度
            opposite_cameras = [opposite_camera]
            opposite_point_uvs, opposite_point_depths, _ = self.getUvs(
                opposite_cameras,
                xyz,
                rescale=self.cfg.rescale,
                padding=self.cfg.padding,
            )

            # 获取两个视角的可见点
            visible_point_idx_1 = self.getVisiblePoints(
                xyz,
                self.viewpoints[best_view_idx : best_view_idx + 1],
                self.cfg.removal_radius,
            )[0]
            visible_point_idx_2 = self.getVisiblePoints(
                xyz, [opposite_viewpoint], self.cfg.removal_radius
            )[0]

            # 计算两个视角的深度总和（使用启发式方法）
            depth_sum_1 = point_depths[best_view_idx][visible_point_idx_1].sum().item()
            depth_sum_2 = opposite_point_depths[0][visible_point_idx_2].sum().item()

            # print(f' Original view depth sum: {depth_sum_1:.4f}')
            # print(f' Opposite view depth sum: {depth_sum_2:.4f}')

            # 选择深度总和更大的视角
            if depth_sum_1 >= depth_sum_2:
                selected_idx = 0
                visible_point_idx = visible_point_idx_1
                selected_point_uvs = point_uvs[best_view_idx]
                selected_point_depths = point_depths[best_view_idx]
                self.view = self.viewpoints[best_view_idx]
                self.cam = self.cameras[best_view_idx]
                print(" Using original view (larger depth sum)")
            else:
                selected_idx = 1
                visible_point_idx = visible_point_idx_2
                selected_point_uvs = opposite_point_uvs[0]
                selected_point_depths = opposite_point_depths[0]
                self.view = opposite_viewpoint
                self.cam = opposite_camera
                print(" Using opposite view (larger depth sum)")

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
            os.makedirs(f"{self.cfg.output_path}/{flag}", exist_ok=True)
            save_image(raw_depth, f"{self.cfg.output_path}/{flag}/raw_depth.png")
            print(" Inpainting depth...")
            if self.cfg.inpainter == "flux":
                save_image(hole_mask1, f"{self.cfg.output_path}/{flag}/mask.png")
                depth = self.inpainter.paint(
                    load_image(f"{self.cfg.output_path}/{flag}/raw_depth.png"),
                    load_image(f"{self.cfg.output_path}/{flag}/mask.png"),
                    prompt="complete the depth map. ",
                    size=self.cfg.res,
                )
                depth.save(f"{self.cfg.output_path}/{flag}/depth.png")
            elif self.cfg.inpainter == "DDNM":
                save_image(hole_mask2, f"{self.cfg.output_path}/{flag}/mask.png")
                depth = self.inpainter.inpaint(
                    masked_imgs=raw_depth.permute(1, 2, 0).unsqueeze(0),
                    masks=hole_mask2.permute(1, 2, 0).unsqueeze(0),
                )[0]
                save_image(depth, f"{self.cfg.output_path}/{flag}/depth.png")
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
                save_image(inpainted_depth, f"{self.cfg.output_path}/{flag}/depth.png")
                save_image(hole_mask1, f"{self.cfg.output_path}/{flag}/mask.png")

            self.point_uv = selected_point_uvs
            np.save(
                f"{self.cfg.output_path}/{flag}/point_uv.npy",
                self.point_uv.detach().cpu().numpy(),
            )
            np.save(f"{self.cfg.output_path}/{flag}/viewpoint.npy", self.view)
            torch.save(self.cam, f"{self.cfg.output_path}/{flag}/camera.pth")

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


if __name__ == "__main__":
    cfg_txt = open("./configs/config.yaml", "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    xyz = torch.tensor(load_xyz("./data/09639.ply")).to(cfg.device)

    dp = DepthPrompting(cfg)
    flag = "06188"
    dp.getImage(
        xyz,
        flag,
    )
    # dp.getRawDepth(xyz, flag)
