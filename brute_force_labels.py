import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from networks.deeplabv3 import DeepLab  # 确保路径正确

# --- 1. 核心路径配置 ---
MODEL_PATH = r'C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth'
IMAGE_DIR = r'C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\image'
MASK_DIR = r'C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask'

# --- 2. 定义 6 种标签映射方案 ---
permutations = [
    {"name": "方案 A: 黑=0, 灰=1, 白=2", "map": [0, 1, 2]},
    {"name": "方案 B: 黑=0, 灰=2, 白=1", "map": [0, 2, 1]},
    {"name": "方案 C: 黑=1, 灰=0, 白=2", "map": [1, 0, 2]},
    {"name": "方案 D: 黑=1, 灰=2, 白=0", "map": [1, 2, 0]},
    {"name": "方案 E: 黑=2, 灰=0, 白=1", "map": [2, 0, 1]},
    {"name": "方案 F: 黑=2, 灰=1, 白=0", "map": [2, 1, 0]},
]


def get_mapped_mask(mask_path, mapping):
    # 🌟 关键修复：Mask 也必须 resize 到 512，且必须用 NEAREST 插值
    mask_pil = Image.open(mask_path).convert('L').resize((512, 512), Image.NEAREST)
    mask_np = np.array(mask_pil)
    label = np.zeros_like(mask_np)
    label[mask_np < 64] = mapping[0]  # 黑色位置映射
    label[(mask_np >= 64) & (mask_np <= 192)] = mapping[1]  # 灰色位置映射
    label[mask_np > 192] = mapping[2]  # 白色位置映射
    return label


def calculate_dice(pred, target, n_classes=3):
    dices = []
    for i in range(n_classes):
        p = (pred == i).float()
        t = (target == i).float()
        inter = (p * t).sum()
        # 加上 smooth 防止分母为 0[cite: 7]
        dice = (2. * inter + 1e-5) / (p.sum() + t.sum() + 1e-5)
        dices.append(dice.item())
    return dices


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型[cite: 4]
    model = DeepLab(num_classes=3, backbone='mobilenet').to(device)

    # 加载 0.81 分权重[cite: 8]
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()

    image_paths = glob(os.path.join(IMAGE_DIR, "*.png"))[:10]
    print(f"🧐 像素对齐完成！开始对 {len(image_paths)} 张图片进行 6 种组合测试...\n")

    for perm in permutations:
        perm_dices = []
        for img_p in image_paths:
            # 预处理图片[cite: 7]
            img = Image.open(img_p).convert('RGB').resize((512, 512))
            img_np = (np.array(img).astype(np.float32) / 127.5) - 1.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                out, _, _ = model(img_tensor)
                pred = torch.argmax(out, dim=1).squeeze(0)

            # 读取并映射标签
            mask_p = os.path.join(MASK_DIR, os.path.basename(img_p))
            target_np = get_mapped_mask(mask_p, perm['map'])
            target_tensor = torch.from_numpy(target_np).to(device)

            dice_scores = calculate_dice(pred, target_tensor)
            perm_dices.append(dice_scores)

        avg_dices = np.mean(perm_dices, axis=0)

        # 判定逻辑：背景通常 Dice 极高 (>0.95)，目标类只要有一个 >0.7 基本就是对的[cite: 7]
        is_candidate = sum(d > 0.7 for d in avg_dices) >= 2
        status = "✨ [命中候选]" if is_candidate else ""

        print(f"{perm['name']}")
        print(
            f"   Dice 详情 -> 索引0: {avg_dices[0]:.4f}, 索引1: {avg_dices[1]:.4f}, 索引2: {avg_dices[2]:.4f} {status}")


if __name__ == "__main__":
    test()