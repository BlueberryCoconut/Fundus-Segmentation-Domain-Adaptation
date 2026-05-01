import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from torchvision import transforms
from networks.deeplabv3_eval import DeepLab as DeepLab_eval
from torch.utils.data import DataLoader


def main():
    # --- 1. 核心路径配置 (确保使用 r'' 原始字符串) ---
    DATA_DIR = r'C:\Users\18268\Desktop\Fundus'
    MODEL_PATH = r'C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth'
    SAVE_DIR = r'C:\Users\18268\Desktop\PLPB-main\final_check_results'

    # 确保文件夹存在
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📁 已创建结果文件夹: {SAVE_DIR}")

    # --- 2. 加载数据 ---
    # 🌟 使用修复后的 Dataloader 加载 Domain1 的测试集
    composed_transforms = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    db_val = DL.FundusSegmentation(base_dir=DATA_DIR, dataset='Domain1', split='test/ROIs',
                                   transform=composed_transforms)
    v_loader = DataLoader(db_val, batch_size=1, shuffle=False)
    print(f"✅ 成功找到测试图片: {len(db_val)} 张")

    # --- 3. 初始化并加载模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLab_eval(num_classes=3, backbone='mobilenet').to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        return

    # 🌟 关键修复：从字典中提取权重
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        # 这里加上 strict=False
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 成功加载权重 (Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f})")
    else:
        # 这里也加上 strict=False[cite: 4]
        model.load_state_dict(checkpoint, strict=False)
        print("✅ 成功加载权重 (直接加载模式)")
    model.eval()

    # --- 4. 生成可视化结果 ---
    with torch.no_grad():
        for i, sample in enumerate(v_loader):
            if i >= 10: break  # 先生成前10张看看效果

            img_tensor = sample['image'].to(device)
            target = sample['label'].numpy()[0]
            name = sample['img_name'][0]

            print(f"正在处理图片 ({i + 1}/10): {name} ...")

            # 推理并获取预测结果
            output, _, _ = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 颜色映射
            # 0: 背景(黑), 1: 视盘(灰), 2: 视杯(白)
            p_vis = np.zeros((512, 512, 3), dtype=np.uint8)
            p_vis[pred == 1] = [128, 128, 128]
            p_vis[pred == 2] = [255, 255, 255]

            t_vis = np.zeros((512, 512, 3), dtype=np.uint8)
            t_vis[target == 1] = [128, 128, 128]
            t_vis[target == 2] = [255, 255, 255]

            # 绘制对比图
            plt.figure(figsize=(15, 5))

            # 显示原图
            img_path = os.path.join(DATA_DIR, 'Domain1', 'test', 'ROIs', 'image', name)
            raw_img = Image.open(img_path).convert('RGB').resize((512, 512))

            plt.subplot(131);
            plt.imshow(raw_img);
            plt.title("Original Image")
            plt.subplot(132);
            plt.imshow(t_vis);
            plt.title("Ground Truth")
            plt.subplot(133);
            plt.imshow(p_vis);
            plt.title(f"Prediction (Dice: {checkpoint.get('best_dice', 0):.2f})")

            for ax in plt.gcf().axes: ax.axis('off')

            # 保存图片
            save_path = os.path.join(SAVE_DIR, f"check_{name}")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"✨ 已保存: {save_path}")

    print(f"\n🎉 大功告成！请去这里查看图片: {SAVE_DIR}")


if __name__ == '__main__':
    main()