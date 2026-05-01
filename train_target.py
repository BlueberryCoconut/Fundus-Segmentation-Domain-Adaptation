import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import DeepLab
from train_process.Trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Final Training")

    # 🌟 1. 把目标域改成 Domain2
    parser.add_argument('--dataset', type=str, default='Domain2')

    # 路径保持指向你的 0.81 分真神模型
    parser.add_argument('--model-file', type=str,
                        default=r'C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth')

    parser.add_argument('--data-dir', default=r'C:\Users\18268\Desktop\Fundus')

    # 🌟 2. 强烈建议：把输出文件夹改个名，防止覆盖你刚才跑出来的 Domain1 0.70 分结果！
    parser.add_argument('--out', default='logs/final_uda_run_D2')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 归一化对齐[cite: 7]
    tsf = transforms.Compose([tr.Resize(512), tr.Normalize_tf(), tr.ToTensor()])

    # 加载数据
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs',
                                     transform=tsf,
                                     pseudo_path=os.path.join(r'C:\Users\18268\Desktop\PLPB-main\results\mask',
                                                              f'{args.dataset}.npz'))
    train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    db_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=tsf)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False)

    model_gen = DeepLab(num_classes=3, backbone='mobilenet').cuda()

    # 🌟 稳健加载权重[cite: 8]
    if os.path.exists(args.model_file):
        ckpt = torch.load(args.model_file, map_location='cuda')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # 自动清洗分布式训练产生的 module. 前缀[cite: 8]
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_gen.load_state_dict(new_state_dict, strict=False)
        print(f"🔥 [实锤] 0.81 模型已从真正的路径注入: {args.model_file}")

    opt_gen = torch.optim.Adam(model_gen.parameters(), lr=args.lr)

    trainer = Trainer(
        cuda=True, model_gen=model_gen, optimizer_gen=opt_gen,
        val_loader=val_loader, domain_loaderS=train_loader,
        out=args.out, max_epoch=40, interval_validate=1
    )
    trainer.train()


if __name__ == '__main__':
    main()



