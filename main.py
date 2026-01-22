import yaml
import torch
from munch import Munch
from utils.dataUtils import *
from DepthPrompting import DepthPrompting
from ScaleAdapter import *
from utils.loss_util import Completionloss
from fpsample import fps_sampling


def metric(flag):
    """计算CD和EMD指标"""
    # 读取GT和预测结果
    gt = o3d.io.read_point_cloud(f"data/GT/{flag}.ply")
    pred = o3d.io.read_point_cloud(f"workspace/{flag}/{flag}_fused.ply")
    
    # 获取点云坐标
    gt_points = np.asarray(gt.points).astype(np.float32)
    pred_points = np.asarray(pred.points).astype(np.float32)

    gt_indices = fps_sampling(gt_points, 16384)
    pred_indices = fps_sampling(pred_points, 16384)
    gt_xyz = gt_points[gt_indices]
    pred_xyz = pred_points[pred_indices]

    gt_tensor = torch.from_numpy(gt_xyz).unsqueeze(0).float().cuda()
    pred_tensor = torch.from_numpy(pred_xyz).unsqueeze(0).float().cuda()
    
    completion_cd = Completionloss(loss_func='cd_l1')
    completion_emd = Completionloss(loss_func='emd')
    
    cd = completion_cd.get_loss(gen=pred_tensor, gt=gt_tensor)
    emd = completion_emd.get_loss(gen=pred_tensor, gt=gt_tensor)
    
    print(f"Flag: {getCategory(flag)}, CD: {cd.item() * 100:.3f}, EMD: {emd.item() * 100:.3f}")
    return cd.item(), emd.item()

def main(cfg):
    """主函数：处理多个样本"""
    # flags = ['01184', '05117', '05452', '06127', '06145', '06188', '06830', '07136', '07306', '09639']
    # flags = ['01184', '05117', '05452', '06127', '06145', '06188', '06830', '07136', '09639']
    flags = ['05117']
    
    results = []

    dp = DepthPrompting(cfg)
    for flag in flags:
        print(f'Processing {flag}...')

        # 加载点云数据 - 修复：正确解包返回值
        xyz_np, rgb_np = load_xyz(f'./data/{flag}.ply')
        xyz = torch.tensor(xyz_np).to(cfg.device)
        rgb = torch.tensor(rgb_np).to(cfg.device)
        dp.getImage(xyz=xyz, flag=flag, rgb=rgb, depth_gen=True, img_gen=True) 

    sa = ScaleAdapter(cfg)
    for flag in flags:
        xyz, rgb = torch.tensor(load_xyz(f'./data/{flag}.ply')).to(cfg.device)
        sa.scaleAdapter(xyz, flag)
        sa.scaleReg(flag)
        
        # 计算指标
        cd, emd = metric(flag)
        results.append({
            'flag': getCategory(flag),  # 修复：不使用集合括号
            'cd': cd, 
            'emd': emd
        })
    
    # 打印总结
    print("\n=== 结果总结 ===")
    for result in results:
        print(f"Category: {result['flag']}, CD: {result['cd'] * 100:.6f}, EMD: {result['emd'] * 100:.6f}")
    
    avg_cd = sum(r['cd'] for r in results) / len(results)
    avg_emd = sum(r['emd'] for r in results) / len(results)
    print(f"平均 CD: {avg_cd * 100:.6f}")
    print(f"平均 EMD: {avg_emd * 100:.6f}")
    
    return results

if __name__ == '__main__':
    cfg_txt = open('./configs/config.yaml', "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    main(cfg)
