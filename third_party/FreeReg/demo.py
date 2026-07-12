import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
from Utils.pose import kabschor
from tools.DepthPro.command_pro_dpt import apple_pro_depth
from tools.YOHO.yoho_extract import yoho_extractor,nn_match
from Utils.utils import dpt_3d_convert,edge_filter,visual_pcd,visual_pcds,transform_points

FREEREG_ROOT = Path(__file__).resolve().parent
LEGACY_FREEREG_ROOT = Path("/opt/data/private/cr/lab/FreeReg")


def _resolve_weight(env_name, local_relative):
    env_path = os.environ.get(env_name)
    if env_path:
        return env_path

    local_path = FREEREG_ROOT / local_relative
    if local_path.exists():
        return str(local_path)

    legacy_path = LEGACY_FREEREG_ROOT / local_relative
    if legacy_path.exists():
        return str(legacy_path)

    return str(local_path)

class Pipe():
    def __init__(self,nkpts=3000,vs=None,w_2d=.5,ir_2d=None,ir_3d=None):
        self.vs = vs
        self.nkpts = nkpts
        self.w_2d = w_2d
        self.ir_2d = ir_2d
        self.ir_3d = ir_3d
        self._load_model()
        self._load_solver()

    def _load_model(self):
        self.depthpro = apple_pro_depth(
            ckpt=_resolve_weight("FREEREG_DEPTHPRO_CKPT", "tools/DepthPro/checkpoints/depth_pro.pt"),
            device='cuda',
        )
        self.yoho = yoho_extractor(
            fcgf_ckpt=_resolve_weight("FREEREG_FCGF_CKPT", "tools/YOHO/model/Backbone/best_val_checkpoint.pth"),
            yoho_ckpt=_resolve_weight("FREEREG_YOHO_CKPT", "tools/YOHO/model/PartI_train/model_best.pth"),
            device='cuda',
        )

    def _load_solver(self):
        self.projector = dpt_3d_convert()
        self.matcher = nn_match()
        self.solver = kabschor(w_2d=self.w_2d)

    def _load_img(self,img_fn):
        return np.array(Image.open(img_fn))/255.

    def _load_pcd(self,pcd_fn):
        if pcd_fn[-3:] == 'npy':
            return np.load(pcd_fn)
        else:
            pcd = o3d.io.read_point_cloud(pcd_fn)
            pcd = np.array(pcd.points)[:,0:3]
            return pcd

    @torch.no_grad()
    def _image_to_3d(self, image):
        self.H,self.W = image.shape[0:2]
        depth,_,_,intrinsic = self.depthpro(image)
        sky = depth > 199.
        xyz = self.projector.proj_depth(depth,intrinsic,depth_unit=1.)
        edg = edge_filter(depth,sky,times=0.05).reshape(-1)
        xyz = xyz[(~edg)&(~sky.reshape(-1))]
        xyz = np.random.permutation(xyz)[:50000]
        return xyz, intrinsic

    def _coarse_align(self, image_pc, pc):
        # coarsely align the generated and scanned point cloud
        range_pc = np.percentile(pc,60,axis=0) - np.percentile(pc,40,axis=0)
        range_ipc = np.percentile(image_pc,60,axis=0) - np.percentile(image_pc,40,axis=0)
        range_ratio = range_pc/(range_ipc+0.1)
        range_ratio = np.mean(range_ratio)
        image_pc = image_pc * range_ratio
        return image_pc,pc

    def _determine_vs(self, image_pc, pc):
        if self.vs is not None: return 0
        delta = 0
        for pc in [image_pc,pc]:
            pc_delta = np.max(pc,axis=0) - np.min(pc,axis=0)
            delta += np.mean(pc_delta)
        delta /= 2
        self.vs = delta / 3 * 0.025

    @torch.no_grad()
    def _extract_yoho(self, pc, nkpts=5000):
        '''
        pc numpy array n*3
        '''
        kpts, yoho_desc, yoho_eqv = self.yoho.run(pc, self.vs, nkpts)
        return kpts, yoho_desc

    def _match(self, sf, tf):
        # des0 --> n0*32
        # des1 --> n1*32
        msid,mtid = self.matcher.find_corr(sf,tf)
        pps = np.concatenate([msid[:,None],mtid[:,None]],axis=-1)
        return pps

    def _pose(self, suvs, skpts, tkpts, pps, intrinsic):
        # for inlier determination
        self.solver.ird_3d = self.vs*5 if self.ir_3d is None else self.ir_3d
        self.solver.ird_2d = max(10,(self.H+self.W)/200.) if self.ir_2d is None else self.ir_2d
        self.solver.set_intrinsic(intrinsic)
        T = self.solver.run(suvs,skpts,tkpts,pps)
        return T

    def _check(self, image, pc, intrinsic, T):
        H,W = image.shape[0:2]
        uvs,_ = self.projector.proj_3to2(pc,intrinsic,T)
        msk = (uvs[:,0]>0) & (uvs[:,1]>0) & (uvs[:,0]<W) & (uvs[:,1]<H)
        rgb = np.ones_like(pc)*0.5
        valid_uvs = uvs[msk].astype(np.int32)
        rgb[msk] = image[valid_uvs[:,1],valid_uvs[:,0]]
        visual_pcd(pc,rgb,normal=True)

    def __call__(self, img, pc):
        print('[Mono Depth Estimation] Project image to 3D with DepthPro...')
        img_pc,intrinsic = self._image_to_3d(img)
        img_pc,pc = self._coarse_align(img_pc,pc)
        self._determine_vs(img_pc,pc)
        print('[YOHO Features] Extract YOHO features on recon point cloud...')
        img_kpts, img_feats = self._extract_yoho(img_pc,self.nkpts)
        img_uvs,_ = self.projector.proj_3to2(img_pc,intrinsic,np.eye(4))
        print('[YOHO Features] Extract YOHO features on given point cloud...')
        pcd_kpts, pcd_feats = self._extract_yoho(pc,self.nkpts)
        print('[Match Corresponses] Match the constructed YOHO features to matches...')
        pps = self._match(img_feats,pcd_feats).astype(np.int16)
        print('[Kabsch] Solve the final pose...')
        T = self._pose(img_uvs,img_kpts,pcd_kpts,pps,intrinsic)
        visual_pcds([transform_points(img_pc,T),pc])
        self._check(img,pc,intrinsic,T)
        print("[Result] Estimated intrinsic of image:\n", intrinsic)
        print("[Result] Estimated extrinsic of image:\n", np.linalg.inv(T))

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',default="demo/outdoor/000183.png",type=str,help='dir of image file')
    parser.add_argument('--pcd',default="demo/outdoor/000189.npy",type=str,help='dir of point cloud file, pcd/ply/npy')
    # The following parameters do not need setting if you just want to use F-FreeReg on-the-fly.
    parser.add_argument('--nkpts',default=5000,type=int,help='extract 5k kpts on each pc')
    parser.add_argument('--vs',default=None,type=float,help='voxel size for pc voxelization, referring https://github.com/HpWang-whu/YOHO/tree/master/others,\
                                                             if set to None, we will auto-set it.')
    parser.add_argument('--w_2d',default=0.5,type=int,help='determine the weight of pnp in ransac')
    parser.add_argument('--ir_2d',default=None,type=int,help='corr-d threshold in pnp ransac, extract 5k kpts on each pc, suggest: 10-30')
    parser.add_argument('--ir_3d',default=None,type=int,help='corr-d threshold in kabsch ransac, set None if you donnot know the scale of the point cloud, suggest: point cloud scale * 0.03')
    args = parser.parse_args()

    pipe = Pipe(args.nkpts,args.vs,args.w_2d,args.ir_2d,args.ir_3d)
    img = pipe._load_img(args.img)
    pcd = pipe._load_pcd(args.pcd)
    pipe(img,pcd)
