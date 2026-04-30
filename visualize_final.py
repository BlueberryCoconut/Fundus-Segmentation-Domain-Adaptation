import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from networks.deeplabv3 import DeepLab

# --- 配置参数 ---
TEST_IMG_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\image"
TEST_MASK_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask"
MODEL_WEIGHT_PATH = r"C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth"
SAVE_DIR = "predict_results_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# 保持与最初始训练一致的 Normalize
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def main():
    print("🚀 启动最终版视觉翻译器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLab(num_classes=3, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=True).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device)['model_state_dict'] if 'model_state_dict' in torch.load(MODEL_WEIGHT_PATH, map_location=device) else torch.load(MODEL_WEIGHT_PATH, map_location=device))
    model.eval()
    print("✅ 巅峰金丹加载成功！")

    img_names = os.listdir(TEST_IMG_DIR)[:5]

    with torch.no_grad():
        for img_name in img_names:
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            # 兼容 .png 标签
            mask_path = os.path.join(TEST_MASK_DIR, img_name.replace('.jpg', '.png'))

            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            output, _, _ = model(img_tensor)
            pred_tensor = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 🌟 核心视觉翻转：把模型的“反色”翻译成“黑底白杯”！
            pred_vis = np.zeros((512, 512), dtype=np.uint8)
            pred_vis[pred_tensor == 2] = 0    # 模型认为类别2是背景，我们涂成黑色
            pred_vis[pred_tensor == 1] = 128  # 模型认为类别1是视盘，我们涂成灰色
            pred_vis[pred_tensor == 0] = 255  # 模型认为类别0是视杯，我们涂成纯白！

            # 🌟 处理 Ground Truth
            if os.path.exists(mask_path):
                mask_np = np.array(Image.open(mask_path).convert('L').resize((512, 512), Image.NEAREST))
                gt_vis = np.zeros((512, 512), dtype=np.uint8)
                gt_vis[mask_np >= 192] = 0      # 原图白背景 -> 转成黑背景
                gt_vis[(mask_np >= 64) & (mask_np < 192)] = 128  # 灰盘 -> 灰盘
                gt_vis[mask_np < 64] = 255      # 原图黑杯 -> 转成白杯！
            else:
                gt_vis = np.zeros((512, 512), dtype=np.uint8)

            # 画图展示
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_pil.resize((512, 512)))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(gt_vis, cmap='gray', vmin=0, vmax=255)
            axes[1].set_title('Ground Truth (Corrected)')
            axes[1].axis('off')
            axes[2].imshow(pred_vis, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title(f'Prediction (Dice: 0.78 Model)')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(SAVE_DIR, f"result_{img_name}")
            plt.savefig(save_path)
            plt.close()

    print(f"\n🎉 视觉翻译完成！去 {SAVE_DIR} 验货真正的黑底白杯吧！")

if __name__ == '__main__':
    main()