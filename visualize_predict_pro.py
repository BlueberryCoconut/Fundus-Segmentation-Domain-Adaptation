import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from networks.deeplabv3 import DeepLab

# --- 🚀 请确认这里的路径和你电脑上的一致 ---
TEST_IMG_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\image"
TEST_MASK_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask"
MODEL_WEIGHT_PATH = r"C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth"
SAVE_DIR = "final_predictions"  # 出图会保存在这个新文件夹里
os.makedirs(SAVE_DIR, exist_ok=True)

# 必须和 custom_transforms.py 里的 Normalize_tf 保持完全一致！
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def main():
    print("🚀 正在加载全新的神级金丹...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 确认是三分类模型
    model = DeepLab(num_classes=3, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=True).to(device)

    # 安全加载权重
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    print("✅ 模型加载成功！开始预测...")

    img_names = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png') or f.endswith('.jpg')][:5]

    with torch.no_grad():
        for img_name in img_names:
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            mask_path = os.path.join(TEST_MASK_DIR, img_name.replace('.jpg', '.png'))

            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # 模型预测
            output, _, _ = model(img_tensor)
            pred_tensor = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 🌟 核心：我们纯正的三分类颜色映射！
            pred_vis = np.zeros((512, 512), dtype=np.uint8)
            pred_vis[pred_tensor == 0] = 0  # 类别0 (背景) -> 黑色
            pred_vis[pred_tensor == 1] = 128  # 类别1 (视盘) -> 灰色
            pred_vis[pred_tensor == 2] = 255  # 类别2 (视杯) -> 纯白

            # 🌟 处理真实的 Ground Truth (用来对比)
            if os.path.exists(mask_path):
                mask_np = np.array(Image.open(mask_path).convert('L').resize((512, 512), Image.NEAREST))
                gt_vis = np.zeros((512, 512), dtype=np.uint8)
                gt_vis[mask_np > 192] = 0  # 原图白底 -> 黑背景
                gt_vis[(mask_np >= 64) & (mask_np <= 192)] = 128  # 灰盘不变
                gt_vis[mask_np < 64] = 255  # 原图黑杯 -> 白杯
            else:
                gt_vis = np.zeros((512, 512), dtype=np.uint8)

            # 画图并保存
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_pil.resize((512, 512)))
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(gt_vis, cmap='gray', vmin=0, vmax=255)
            axes[1].set_title('Ground Truth (Corrected)')
            axes[1].axis('off')

            axes[2].imshow(pred_vis, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title('Prediction (New Model)')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(SAVE_DIR, f"result_{img_name.split('.')[0]}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"📸 {img_name} 预测图已保存!")

    print(f"\n🎉 大功告成！快去左边目录找找 '{SAVE_DIR}' 文件夹，点开看看吧！")


if __name__ == '__main__':
    main()