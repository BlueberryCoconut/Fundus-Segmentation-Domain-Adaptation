#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from PIL import Image
import tqdm
from torchvision import transforms

# 导入你项目里的数据处理和模型文件
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader
from networks.deeplabv3 import DeepLab

# 设置随机种子保证可复现
seed = 3377
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description="生成视盘(灰)/视杯(黑)/背景(白)的三色可视化图像")

    # 默认路径已替换为你最新的 best_model.pth
    parser.add_argument('--model_file', type=str,
                        default=r"C:\Users\18268\Desktop\PLPB-main\logs\Domain1\20260429_224100.174927\best_model.pth",
                        help='模型权重文件路径')
    # 数据集路径
    parser.add_argument('--data_dir', type=str,
                        default='C:/Users/18268/Desktop/Fundus/',
                        help='数据集根目录')
    parser.add_argument('--dataset', type=str, default='Domain1')

    # 输出图片保存路径，默认建在项目根目录下的 visual_results 文件夹
    parser.add_argument('--output_dir', type=str,
                        default='./visual_results',
                        help='生成的三色图片保存路径')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=1)  # 可视化强制设为 1，防止打乱顺序
    parser.add_argument('--out_stride', type=int, default=16)

    return parser.parse_args()


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 1. 创建输出文件夹
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. 数据加载
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    print(f"📁 正在加载数据集: {args.dataset}")
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                     split='train/ROIs', transform=composed_transforms_test)
    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # 3. 初始化模型
    print("🚀 正在初始化 DeepLab 模型...")
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=False, freeze_bn=True).cuda()

    # 4. 加载模型权重
    print(f"🔍 正在加载权重: {args.model_file}")
    if os.path.exists(args.model_file):
        checkpoint = torch.load(args.model_file, map_location='cpu')
        # 兼容不同格式的存档
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型权重加载成功！")
    else:
        raise FileNotFoundError(f"❌ 找不到模型文件，请检查路径: {args.model_file}")

    model.eval()  # 切换到评估模式

    print(f"🎨 开始生成三色图 (白:背景, 灰:视盘, 黑:视杯)...")

    with torch.no_grad():
        for sample in tqdm.tqdm(train_loader, desc="Generating Images"):
            data, img_name = sample['image'].cuda(), sample['img_name']

            # 模型推理
            out_p, _, _ = model(data)

            # 去除 batch 维度并转到 CPU，概率化 (0~1)
            # 形状变成 [2, H, W]
            probs = torch.sigmoid(out_p[0]).cpu()

            # --- 核心：三色涂色法 ---
            # 背景：创建一张和输入等大的纯白画布 (255)
            visual_label = torch.ones_like(probs[0]) * 255.0

            # 视盘 (Disc)：灰色 (128)
            # 根据网络结构，通道 1 是视盘。概率 > 0.5 涂成灰色
            visual_label[probs[1] > 0.5] = 128.0

            # 视杯 (Cup)：纯黑 (0)
            # 通道 0 是视杯。概率 > 0.5 涂成黑色，覆盖在中间
            visual_label[probs[0] > 0.5] = 0.0

            # --- 保存图片 ---
            # 转换为 8位无符号整型
            visual_label_np = visual_label.numpy().astype(np.uint8)
            out_img = Image.fromarray(visual_label_np)

            # 处理文件名并保存
            # 处理文件名并保存
            name_str = img_name[0] if isinstance(img_name, (list, tuple)) else img_name

            # 🌟 核心修复：把前面的子文件夹路径全部砍掉，只留纯文件名
            name_str = os.path.basename(name_str)

            # 确保保存为 png 格式以防压缩导致颜色失真
            if not name_str.endswith('.png'):
                name_str = name_str.split('.')[0] + '.png'

            save_path = os.path.join(args.output_dir, name_str)

            # 安全起见，如果在 output_dir 里还有子目录，自动创建它
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            out_img.save(save_path)

    print(f"🎉 搞定！所有图片已保存在这个文件夹: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()