import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 🌟 必须改成你实际生成器的导入路径，比如 networks.GAN 里的 Generator
from networks.deeplabv3 import DeepLab

# --- 配置参数 ---
TEST_IMG_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\image"
# 🌟 注意：这里我改成了 mask 文件夹，确保它存在！
TEST_MASK_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask"
MODEL_WEIGHT_PATH = r"C:\Users\18268\Desktop\PLPB-main\logs\Domain1\20260430_183600.059137\best_model.pth"
SAVE_DIR = r"predict_results"

os.makedirs(SAVE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def main():
    print("🚀 开始加载模型进行预测...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 🌟 初始化模型 (必须和你训练时用的结构一模一样)
    # 🌟 这里的参数必须和 train_source.py 里初始化模型时一模一样！
    model = DeepLab(num_classes=3, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=True).to(device)

    if not os.path.exists(MODEL_WEIGHT_PATH):
        print(f"❌ 找不到权重文件: {MODEL_WEIGHT_PATH}")
        return

    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ 模型权重加载成功！")

    img_names = os.listdir(TEST_IMG_DIR)[:5]

    with torch.no_grad():
        for img_name in img_names:
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            mask_path = os.path.join(TEST_MASK_DIR, img_name)

            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            output, _, _ = model(img_tensor)
            pred_tensor = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            mask_pil = None
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path)
                mask_np = np.array(mask_pil.resize((512, 512), Image.NEAREST))

                gt_mask = np.zeros_like(mask_np)
                gt_mask[(mask_np >= 64) & (mask_np < 192)] = 1
                gt_mask[mask_np >= 192] = 2
            else:
                gt_mask = np.zeros((512, 512))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_pil.resize((512, 512)))
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=2)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_tensor, cmap='gray', vmin=0, vmax=2)
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(SAVE_DIR, f"result_{img_name}")
            plt.savefig(save_path)
            plt.close()
            print(f"✔️ 已保存预测结果: {save_path}")

    print(f"\n🎉 全部完成！请前往 {SAVE_DIR} 文件夹查看图片。")


if __name__ == '__main__':
    main()