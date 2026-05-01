import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import DeepLab
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator
from train_process.Trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Fundus Segmentation Training - Fixed")

    # --- 基础配置 ---
    parser.add_argument('--dataset', type=str, default='Domain1')
    parser.add_argument('--data-dir', default=r'C:\Users\18268\Desktop\Fundus')
    parser.add_argument('--out', default='logs/weighted_run')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr-dis', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=40)

    args = parser.parse_args()

    # 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.makedirs(args.out, exist_ok=True)

    # 1. 定义数据增强
    transform_train = transforms.Compose([tr.Resize(512), tr.Normalize_tf(), tr.ToTensor()])
    transform_val = transforms.Compose([tr.Resize(512), tr.Normalize_tf(), tr.ToTensor()])

    # 2. 加载数据集 (基于 Domain1 真实路径)
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs',
                                     transform=transform_train)
    train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    db_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset='Domain1', split='test/ROIs',
                                   transform=transform_val)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # 3. 初始化模型
    model_gen = DeepLab(num_classes=3, backbone='mobilenet', output_stride=16).cuda()
    model_dis = BoundaryDiscriminator(in_channels=3).cuda()
    model_dis2 = UncertaintyDiscriminator(in_channels=3).cuda()

    # 4. 配置优化器
    opt_gen = torch.optim.Adam(model_gen.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr_dis, betas=(0.9, 0.999))
    opt_dis2 = torch.optim.Adam(model_dis2.parameters(), lr=args.lr_dis, betas=(0.9, 0.999))

    # 5. 调用 Trainer 启动训练 (严格匹配 Trainer_6.py 的参数顺序)
    trainer = Trainer(
        cuda=True,
        model_gen=model_gen,
        model_dis=model_dis,
        model_uncertainty_dis=model_dis2,
        optimizer_gen=opt_gen,
        optimizer_dis=opt_dis,
        optimizer_uncertainty_dis=opt_dis2,

        # 🌟 修复核心：传入 Trainer_6.py 要求的 4 个缺失参数
        lr_gen=args.lr,
        lr_dis=args.lr_dis,
        lr_decrease_rate=0.1,
        val_loader=val_loader,
        domain_loaderS=train_loader,
        domain_loaderT=train_loader,  # 暂时使用 S 作为 T 占位
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.max_epoch,

        # 可选参数
        interval_validate=1,
        batch_size=args.batch_size,
        warmup_epoch=10
    )

    print(f"🚀 [Domain1 训练启动] 总轮数: {args.max_epoch} | 结果保存至: {args.out}")
    trainer.train()


if __name__ == '__main__':
    main()