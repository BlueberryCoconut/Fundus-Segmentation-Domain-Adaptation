from datetime import datetime
import os
import os.path as osp
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from train_process import Trainer
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator

here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--datasetS', type=str, default='Domain3')
    parser.add_argument('--datasetT', type=str, default='Domain1')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--warmup-epoch', type=int, default=20)
    parser.add_argument('--data-dir', default=r'C:\Users\18268\Desktop\Fundus')
    parser.add_argument('--lr-gen', type=float, default=1e-3)
    parser.add_argument('--lr-dis', type=float, default=2.5e-5)

    args = parser.parse_args()
    args.model = 'FCN8s'
    now = datetime.now()
    args.out = osp.join(here, 'logs/', args.datasetT, now.strftime('%Y%m%d_%H%M%S.%f'))

    if not os.path.exists(args.out): os.makedirs(args.out)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        # 🌟 开启硬件优化，加速卷积计算
        torch.backends.cudnn.benchmark = True

    # --- 数据增强配置 ---
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomHorizontalFlip(),
        tr.adjust_light(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    # --- DataLoader 设置 ---
    # 🌟 提速：num_workers 设为 2；pin_memory 设为 True
    domain = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='train/ROIs/',
                                   transform=composed_transforms_tr)
    domain_loaderS = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train/ROIs/',
                                     transform=composed_transforms_tr)
    domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='test/ROIs/',
                                       transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=True)

    # --- 模型初始化 ---
    # 🌟 关键：num_classes 设为 3[cite: 4, 8]
    model_gen = DeepLab(num_classes=3, backbone='mobilenet', output_stride=16).cuda()

    # 🌟 关键：判别器 in_channels 设为 3 以匹配三色预测图
    model_dis = BoundaryDiscriminator(in_channels=3).cuda()
    model_dis2 = UncertaintyDiscriminator(in_channels=3).cuda()

    # --- 优化器 ---
    optim_gen = torch.optim.Adam(model_gen.parameters(), lr=args.lr_gen, betas=(0.9, 0.99))
    optim_dis = torch.optim.SGD(model_dis.parameters(), lr=args.lr_dis, momentum=0.99, weight_decay=0.0005)
    optim_dis2 = torch.optim.SGD(model_dis2.parameters(), lr=args.lr_dis, momentum=0.99, weight_decay=0.0005)

    # --- 启动训练器 ---
    trainer = Trainer.Trainer(
        cuda=True,
        model_gen=model_gen,
        model_dis=model_dis,
        model_uncertainty_dis=model_dis2,
        optimizer_gen=optim_gen,
        optimizer_dis=optim_dis,
        optimizer_uncertainty_dis=optim_dis2,
        lr_gen=args.lr_gen,
        lr_dis=args.lr_dis,
        lr_decrease_rate=0.1,
        val_loader=domain_loader_val,
        domain_loaderS=domain_loaderS,
        domain_loaderT=domain_loaderT,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.max_epoch,
        interval_validate=1,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch,  # 🌟 20 轮预热[cite: 4, 8]
    )
    trainer.train()


if __name__ == '__main__':
    main()