import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 导入你的生成器
from networks.deeplabv3 import DeepLab

# --- 配置参数 ---
TEST_IMG_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\image"
TEST_MASK_DIR = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask"
MODEL_WEIGHT_PATH = r"C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth"
SAVE_DIR = r"predict_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# 保持与训练完全一致的数据预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def main():
    print("🚀 开始加载金丹模型进行出图...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = DeepLab(num_classes=3, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=True).to(device)

    if not os.path.exists(MODEL_WEIGHT_PATH):
        print(f"❌ 找不到权重文件: {MODEL_WEIGHT_PATH}")
        return

    # 查岗：确认权重文件的修改时间
    import time
    mtime = os.path.getmtime(MODEL_WEIGHT_PATH)
    print(f"🕒 当前读取的权重文件最后修改时间: {time.ctime(mtime)}")

    # 加载权重
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ 模型权重加载成功！\n" + "-" * 30)

    img_names = os.listdir(TEST_IMG_DIR)[:5]

    with torch.no_grad():
        for img_name in img_names:
            img_path = os.path.join(TEST_IMG_DIR, img_name)

            # 🌟 核心修复：把原图的 .jpg 后缀换成标签图常用的 .png
            # （如果你的 mask 是 .bmp 或 .tif，请在这里修改）
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(TEST_MASK_DIR, mask_name)

            # 🌟 照妖镜：打印路径，看看到底存不存在
            print(f"🔍 正在处理图片: {img_name}")
            print(f"👉 寻找标签图: {mask_path} -> 存在吗？ {os.path.exists(mask_path)}")

            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            output, _, _ = model(img_tensor)
            pred_tensor = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 强制进行绝对的灰度颜色映射
            pred_vis = np.zeros((512, 512), dtype=np.uint8)
            pred_vis[pred_tensor == 1] = 128  # 视盘涂成中间灰
            pred_vis[pred_tensor == 2] = 255  # 视杯涂成纯白

            # 强制转为单通道灰度图 'L'
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path).convert('L')
                mask_np = np.array(mask_pil.resize((512, 512), Image.NEAREST))

                gt_vis = np.zeros((512, 512), dtype=np.uint8)
                gt_vis[(mask_np >= 64) & (mask_np < 192)] = 128  # 标签视盘
                gt_vis[mask_np >= 192] = 255  # 标签视杯
            else:
                gt_vis = np.zeros((512, 512), dtype=np.uint8)
                print("   ⚠️ 警告：因为找不到标签图，Ground Truth 将显示为全黑！")

            # 画图部分
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_pil.resize((512, 512)))
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(gt_vis, cmap='gray', vmin=0, vmax=255)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_vis, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(SAVE_DIR, f"result_{img_name}")
            plt.savefig(save_path)
            plt.close()
            print(f"✔️ 已保存预测结果: {save_path}\n")


if __name__ == '__main__':
    main()