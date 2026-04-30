#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import tqdm
import numpy as np
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from networks.deeplabv3 import *

# 设置随机种子保证可复现性
seed = 3377
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    # 请根据你的实际路径修改 default 值
    # 注意引号前面的那个小写 r
    parser.add_argument('--model-file', type=str,
                        default=r"C:\Users\18268\Desktop\PLPB-main\logs\Domain1\20260429_224100.174927\best_model.pth")
    parser.add_argument('--dataset', type=str, default='Domain1')
    parser.add_argument('--batchsize', type=int, default=4)  # 8GB 建议设为 4
    parser.add_argument('--data-dir', default='C:/Users/18268/Desktop/Fundus/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--sync-bn', type=bool, default=False)  # 8GB 必须关掉[cite: 2]
    parser.add_argument('--freeze-bn', type=bool, default=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 1. 加载数据流 (注意：此处需与 custom_transforms.py 配合)[cite: 1, 4]
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    # 使用目标域训练集来生成伪标签[cite: 4, 5]
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs',
                                     transform=composed_transforms_test)
    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # 2. 初始化模型[cite: 2]
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    # 加载权重[cite: 4]
    if os.path.exists(args.model_file):
        checkpoint = torch.load(args.model_file)

        # 自动判断存档格式
        if 'model_state_dict' in checkpoint:
            # 如果是大字典格式 (如 checkpoint_latest.pth.tar)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果是纯权重格式 (如 best_model.pth)
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model: {args.model_file}")
    else:
        print("Warning: Model file not found!")

    model.train()  # 必须开启 train 模式以启用 Dropout 进行不确定性估计[cite: 4]

    # 用于存储结果的字典[cite: 4]
    pseudo_label_dic, uncertain_dic, proto_pseudo_dic = {}, {}, {}
    dist_0_obj_dic, dist_0_bck_dic, dist_1_obj_dic, dist_1_bck_dic = {}, {}, {}, {}

    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Generating Labels"):
            data, img_name = sample['image'].cuda(), sample['img_name']

            # --- 🚀 显存优化核心：累加模式代替大张量存储 ---[cite: 4]
            num_rounds = 10
            sum_preds = 0
            sum_sq_preds = 0
            sum_features = 0

            for _ in range(num_rounds):
                # 执行推理，得到分割图预测和最后一层特征[cite: 2, 4]
                out_p, _, out_f = model(data)
                prob = torch.sigmoid(out_p / 2.0)  # 概率平滑[cite: 4]

                sum_preds += prob
                sum_sq_preds += prob ** 2
                sum_features += out_f

            # 计算平均预测值、标准差(不确定性)和平均特征[cite: 4]
            prediction = sum_preds / num_rounds
            std_map = torch.sqrt(torch.abs((sum_sq_preds / num_rounds) - prediction ** 2))
            feature = sum_features / num_rounds

            # 1. 生成初步伪标签 (阈值 0.75)[cite: 4]
            pseudo_label = (prediction > 0.75).float()

            # 2. ProDA 原型计算准备[cite: 4]
            # 缩放掩码以匹配特征图尺寸 (通常是输入尺寸的 1/4 或 1/8)[cite: 2, 4]
            feat_size = feature.size()[2:]
            target_0_obj = F.interpolate(pseudo_label[:, 0:1, ...], size=feat_size, mode='nearest')
            target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feat_size, mode='nearest')
            mask_reliable = F.interpolate((std_map < 0.05).float(), size=feat_size, mode='nearest')

            # 计算类别中心 (Centroid) - 这里展示简化后的逻辑[cite: 4]
            def get_centroid(f, m):
                # 只计算高置信度区域的特征均值[cite: 4, 5]
                denom = torch.sum(m, dim=[0, 2, 3], keepdim=True) + 1e-6
                return torch.sum(f * m, dim=[0, 2, 3], keepdim=True) / denom

            c0_obj = get_centroid(feature, target_0_obj * mask_reliable[:, 0:1, ...])
            c0_bck = get_centroid(feature, (1 - target_0_obj) * mask_reliable[:, 0:1, ...])
            c1_obj = get_centroid(feature, target_1_obj * mask_reliable[:, 1:, ...])
            c1_bck = get_centroid(feature, (1 - target_1_obj) * mask_reliable[:, 1:, ...])

            # 计算欧氏距离平方[cite: 4]
            dist_0_obj = torch.sum((feature - c0_obj) ** 2, dim=1, keepdim=True)
            dist_0_bck = torch.sum((feature - c0_bck) ** 2, dim=1, keepdim=True)
            dist_1_obj = torch.sum((feature - c1_obj) ** 2, dim=1, keepdim=True)
            dist_1_bck = torch.sum((feature - c1_bck) ** 2, dim=1, keepdim=True)

            # 根据原型距离生成伪标签[cite: 4]
            proto_0 = (dist_0_obj < dist_0_bck).float()
            proto_1 = (dist_1_obj < dist_1_bck).float()
            proto_pseudo = torch.cat([proto_0, proto_1], dim=1)
            proto_pseudo = F.interpolate(proto_pseudo, size=data.size()[2:], mode='nearest')

            # 3. 存储结果[cite: 4]
            for i in range(data.shape[0]):
                name = img_name[i]
                pseudo_label_dic[name] = pseudo_label[i].cpu().numpy()
                uncertain_dic[name] = std_map[i].cpu().numpy()
                proto_pseudo_dic[name] = proto_pseudo[i].cpu().numpy()
                dist_0_obj_dic[name] = dist_0_obj[i].cpu().numpy()
                dist_0_bck_dic[name] = dist_0_bck[i].cpu().numpy()

    # 4. 保存为 NPZ 文件供下一步训练使用[cite: 4]
    save_path = f'./results/mask/{args.dataset}.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, pseudo_label_dic, uncertain_dic, proto_pseudo_dic, dist_0_obj_dic, dist_0_bck_dic)
    print(f"Pseudo labels saved to {save_path}")


if __name__ == '__main__':
    main()