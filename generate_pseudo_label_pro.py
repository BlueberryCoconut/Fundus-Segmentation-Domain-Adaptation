#!/usr/bin/env python
import argparse
import os
import torch
import tqdm
import numpy as np
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from networks.deeplabv3 import DeepLab

# 保证可复现性
seed = 3377
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    # 🌟 修复 1：指向你真正的源域模型路径
    parser.add_argument('--model-file', type=str,
                        default=r"C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth")
    parser.add_argument('--dataset', type=str, default='Domain1')
    parser.add_argument('--batchsize', type=int, default=1)  # 生成伪标签建议 batchsize=1 保证对应关系
    parser.add_argument('--data-dir', default='C:/Users/18268/Desktop/Fundus/')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    # 加载目标域训练集
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs',
                                     transform=composed_transforms_test)
    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=0)

    model = DeepLab(num_classes=3, backbone='mobilenet', output_stride=16).cuda()

    if os.path.exists(args.model_file):
        checkpoint = torch.load(args.model_file)
        # 🌟 修复 2：加入 strict=False 解决 ASPP 命名微差问题[cite: 5]
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 成功加载模型: {args.model_file}")
    else:
        print(f"❌ 未找到模型文件: {args.model_file}")
        exit()

    model.eval()  # 先整体设为评估模式
    for m in model.modules():
        # 只把 Dropout 层设为训练模式，这样既能算不确定性，又不会让 BatchNorm 报错
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    print("✅ 已激活 MC Dropout 模式 (BatchNorm 已固定)")

    pseudo_label_dic = {}
    print("🚀 正在生成伪标签...")

    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            data = sample['image'].cuda()
            img_names = sample['img_name']

            num_rounds = 10  # MC Dropout 次数[cite: 5]
            sum_preds = 0

            for _ in range(num_rounds):
                out_p, _, _ = model(data)
                sum_preds += torch.softmax(out_p, dim=1)

            prediction = sum_preds / num_rounds
            # 取概率最大的类别[cite: 5]
            pseudo_labels = torch.argmax(prediction, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(data.shape[0]):
                # 🌟 修复 3：直接保存提取出的 numpy 子数组，不再调用 .cpu()[cite: 5]
                name = os.path.basename(img_names[i])
                pseudo_label_dic[name] = pseudo_labels[i]

    # 🌟 修复 4：保存路径和解包方式[cite: 5]
    save_dir = './results/mask'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{args.dataset}.npz')

    # 使用 ** 解包字典，确保 train_target.py 能正确读取[cite: 5]
    np.savez(save_path, **pseudo_label_dic)
    print(f"🎉 伪标签已成功保存至: {save_path}")


if __name__ == '__main__':
    main()