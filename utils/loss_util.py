import sys
import torch
sys.path.append('..')

# from loss_functions import chamfer_l1, chamfer_l2, chamfer_partial_l1, chamfer_partial_l2, emd_loss
from loss_functions import chamfer_3DDist, emdModule

class Completionloss:
    def __init__(self, loss_func='cd_l1'):
        self.loss_func = loss_func
        self.chamfer_dist = chamfer_3DDist()
        self.EMD = torch.nn.DataParallel(emdModule().cuda()).cuda()

        if loss_func == 'cd_l1':
            self.metric = self.chamfer_l1
            self.partial_matching = self.chamfer_partial_l1
        elif loss_func == 'cd_l2':
            self.metric = self.chamfer_l2
            self.partial_matching = self.chamfer_partial_l2
        elif loss_func == 'infocd':
            self.metric = self.info_cd
            self.partial_matching = self.info_cd_partial
        elif loss_func == 'emd':
            self.metric = self.emd_loss
        else:
            raise Exception('loss function {} not supported yet!'.format(loss_func))

    def chamfer_l1(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        d1 = torch.mean(torch.sqrt(d1))
        d2 = torch.mean(torch.sqrt(d2))
        return (d1 + d2) / 2

    def chamfer_l2(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        return torch.mean(d1) + torch.mean(d2)

    def chamfer_partial_l1(self, pcd1, pcd2):
        d1, d2, _, _ = self.chamfer_dist(pcd1, pcd2)
        d1 = torch.mean(torch.sqrt(d1))
        return d1

    def chamfer_partial_l2(self, pcd1, pcd2):
        d1, d2, _, _ = self.chamfer_dist(pcd1, pcd2)
        d1 = torch.mean(d1)
        return d1

    def _info_distance(self, dist):
        # Adapted from PGD InfoCD: https://github.com/git-guocc/PGD/blob/main/models/InfoCD.py
        dist = torch.clamp(dist, min=1e-9)
        d = torch.sqrt(dist)
        exp_d = torch.exp(-0.5 * d)
        normalizer = torch.sum(exp_d + 1e-7, dim=-1).unsqueeze(-1)
        return -torch.log(exp_d / (normalizer ** 1e-7))

    def info_cd(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        distances1 = self._info_distance(d1)
        distances2 = self._info_distance(d2)
        return (torch.sum(distances1) + torch.sum(distances2)) / (2 * p1.shape[0])

    def info_cd_partial(self, pcd1, pcd2):
        d1, _, _, _ = self.chamfer_dist(pcd1, pcd2)
        distances1 = self._info_distance(d1)
        return torch.sum(distances1) / pcd1.shape[0]

    def emd_loss(self, p1, p2):
        d1, _ = self.EMD(p1, p2, eps=0.005, iters=50)
        d = torch.sqrt(d1).mean(1).mean()

        return d

    def get_loss(self, gen, gt):
        loss = self.metric(gen, gt)
        return loss






if __name__ == '__main__':
    gt = torch.randn(10, 2048, 3).cuda()
    pc = torch.randn(10, 256, 3).cuda()
    p1 = torch.randn(10, 512, 3).cuda()
    p2 = torch.randn(10, 1024, 3).cuda()
    p3 = torch.randn(10, 2048, 3).cuda()


    # loss = get_loss([pc, p1, p2, p3], gt, gt, 'emd')[0]
    # print(loss.item())
