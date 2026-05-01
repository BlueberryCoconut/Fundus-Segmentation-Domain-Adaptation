import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from networks.deeplabv3 import DeepLab
from glob import glob

# --- 1. 配置路径 (指向我们的真神模型) ---
MODEL_PATH = r'C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth'
IMAGE_DIR = r'C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\image'
MASK_DIR = r'C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask'
SAVE_DIR = r'C:\Users\18268\Desktop\PLPB-main\vis_results'

os.makedirs(SAVE_DIR, exist_ok=True)


def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = DeepLab(num_classes=3, backbone='mobilenet').to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()

    # 获取测试图片
    img_list = glob(os.path.join(IMAGE_DIR, "*.png"))[:5]  # 先看5张

    print(f"🚀 正在生成可视化图像，请稍后...")

    for img_p in img_list:
        name = os.path.basename(img_p)

        # 1. 处理原图
        raw_img = Image.open(img_p).convert('RGB').resize((512, 512))
        img_np = (np.array(raw_img).astype(np.float32) / 127.5) - 1.0  # 归一化对齐
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # 2. 推理
        with torch.no_grad():
            output, _, _ = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # 3. 处理真实标签 (映射方案 A: 0=黑, 1=灰, 2=白)
        mask_p = os.path.join(MASK_DIR, name)
        gt_raw = np.array(Image.open(mask_p).convert('L').resize((512, 512), Image.NEAREST))
        gt = np.zeros_like(gt_raw)
        gt[gt_raw < 64] = 0  # 视杯
        gt[(gt_raw >= 64) & (gt_raw <= 192)] = 1  # 视盘
        gt[gt_raw > 192] = 2  # 背景

        # 4. 染色逻辑：为了好看，我们将背景设为深色，视盘设为绿色，视杯设为红色
        def colorize(mask):
            color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
            color_mask[mask == 0] = [255, 0, 0]  # 0 视杯 -> 红色
            color_mask[mask == 1] = [0, 255, 0]  # 1 视盘 -> 绿色
            color_mask[mask == 2] = [30, 30, 30]  # 2 背景 -> 深灰色
            return color_mask

        # 5. 绘图
        plt.figure(figsize=(15, 5))
        plt.subplot(131);
        plt.imshow(raw_img);
        plt.title("Original Image")
        plt.subplot(132);
        plt.imshow(colorize(gt));
        plt.title("Ground Truth")
        plt.subplot(133);
        plt.imshow(colorize(pred));
        plt.title("Model Prediction")

        plt.savefig(os.path.join(SAVE_DIR, f"res_{name}"))
        plt.close()

    print(f"✅ 可视化完成！快去这个文件夹看图：\n{SAVE_DIR}")


if __name__ == '__main__':
    visualize()