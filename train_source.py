import argparse
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 导入项目关联模块
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import DeepLab
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator
from train_process.Trainer import Trainer


def main():
    # 使用 argparse 来接收命令行输入的参数
    parser = argparse.ArgumentParser(description="Fundus Segmentation Training")

    # --- 训练核心参数 ---
    parser.add_argument('--max-epoch', type=int, default=40, help='Total training epochs')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes: bg, disc, cup')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for generator')

    # --- 路径与环境 ---
    parser.add_argument('--data-dir', default=r'C:/Users/18268/Desktop/Fundus/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='Domain1', help='Select dataset domain')
    parser.add_argument('--out', default='logs/weighted_run', help='Directory to save logs and models')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    # --- 对抗学习相关 ---
    parser.add_argument('--stop-epoch', type=int, default=40)
    parser.add_argument('--warmup-epoch', type=int, default=10, help='Epoch to start adversarial training')
    parser.add_argument('--lr-dis', type=float, default=1e-4)
    parser.add_argument('--lr-decrease-rate', type=float, default=0.1)

    args = parser.parse_args()

    # 基础环境配置
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    if not osp.exists(args.out):
        os.makedirs(args.out)

    # 1. 定义数据增强
    composed_transforms_train = transforms.Compose([
        tr.Resize(512),
        tr.RandomHorizontalFlip(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_val = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    # 2. 加载数据集
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs',
                                     transform=composed_transforms_train)
    train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 目标域数据 (用于对抗学习)
    db_target = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs',
                                      transform=composed_transforms_train)
    target_loader = DataLoader(db_target, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 验证集
    db_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs',
                                   transform=composed_transforms_val)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # 3. 初始化网络模型
    # 生成器：DeepLabV3+
    model_gen = DeepLab(num_classes=args.num_classes, backbone='mobilenet', output_stride=16, sync_bn=False,
                        freeze_bn=True)

    # 判别器：根据 GAN.py 源码定义的类名和参数名[cite: 1]
    model_dis = BoundaryDiscriminator(in_channels=args.num_classes)
    model_dis2 = UncertaintyDiscriminator(in_channels=args.num_classes)

    if cuda:
        model_gen = model_gen.cuda()
        model_dis = model_dis.cuda()
        model_dis2 = model_dis2.cuda()

    # 4. 配置优化器
    opt_gen = torch.optim.Adam(model_gen.parameters(), lr=args.lr, betas=(0.9, 0.99))
    opt_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr_dis, betas=(0.9, 0.99))
    opt_dis2 = torch.optim.Adam(model_dis2.parameters(), lr=args.lr_dis, betas=(0.9, 0.99))

    # 5. 调用 Trainer 启动训练
    trainer = Trainer(
        cuda=cuda,
        model_gen=model_gen,
        model_dis=model_dis,
        model_uncertainty_dis=model_dis2,
        optimizer_gen=opt_gen,
        optimizer_dis=opt_dis,
        optimizer_uncertainty_dis=opt_dis2,
        lr_gen=args.lr,
        lr_dis=args.lr_dis,
        lr_decrease_rate=args.lr_decrease_rate,
        val_loader=val_loader,
        domain_loaderS=train_loader,
        domain_loaderT=target_loader,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch
    )

    print(f"🚀 准备就绪！训练轮数: {args.max_epoch} | 结果路径: {args.out} | 类别数: {args.num_classes}")
    trainer.train()


if __name__ == '__main__':
    main()