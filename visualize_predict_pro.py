import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 🌟 破案核心 1：必须导入作者专门写的 eval 测试网络！
from networks.deeplabv3_eval import DeepLab as DeepLab_eval

# --- 🚀 路径保持不变 ---
TEST_IMG_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\image"
TEST_MASK_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask"
MODEL_WEIGHT_PATH = r"C:\Users\18268\Desktop\PLPB-main\logs\uda_run\best_uda_model.pth"
SAVE_DIR = "final_predictions_fixed"  # 🌟 我们建一个新文件夹来放完美预测图
os.makedirs(SAVE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def main():
    print("🚀 正在加载全新的神级金丹...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 🌟 破案核心 2：使用 DeepLab_eval 进行初始化
    model = DeepLab_eval(num_classes=3, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=True).to(
        device)

    # 🌟 破案核心 3：安全加载并过滤多余权重（像你在 train_target 里做的那样）
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    pretrained_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model.load_state_dict(pretrained_dict)
    model.eval()
    print("✅ 模型加载并成功过滤！开始真正的预测...")

    img_names = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png') or f.endswith('.jpg')][:5]

    with torch.no_grad():
        for img_name in img_names:
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            mask_path = os.path.join(TEST_MASK_DIR, img_name.replace('.jpg', '.png'))

            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            output, _, _ = model(img_tensor)
            pred_tensor = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 颜色映射
            pred_vis = np.zeros((512, 512), dtype=np.uint8)
            pred_vis[pred_tensor == 0] = 0  # 背景 -> 黑
            pred_vis[pred_tensor == 1] = 128  # 视盘 -> 灰
            pred_vis[pred_tensor == 2] = 255  # 视杯 -> 白

            if os.path.exists(mask_path):
                mask_np = np.array(Image.open(mask_path).convert('L').resize((512, 512), Image.NEAREST))
                gt_vis = np.zeros((512, 512), dtype=np.uint8)
                gt_vis[mask_np > 192] = 0
                gt_vis[(mask_np >= 64) & (mask_np <= 192)] = 128
                gt_vis[mask_np < 64] = 255
            else:
                gt_vis = np.zeros((512, 512), dtype=np.uint8)

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
            print(f"📸 {img_name} 真·预测图已保存!")

    print(f"\n🎉 大功告成！快去左边找找 '{SAVE_DIR}' 文件夹看看真正的效果吧！")


if __name__ == '__main__':
    main()