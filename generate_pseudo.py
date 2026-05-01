import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from networks.deeplabv3 import DeepLab
from tqdm import tqdm

# --- 配置路径 ---
MODEL_PATH = r'C:\Users\18268\Desktop\PLPB-main\logs\weighted_run\best_model.pth'
IMG_DIR = r'C:\Users\18268\Desktop\Fundus\Domain2\train\ROIs\image'
SAVE_PATH = r'C:\Users\18268\Desktop\PLPB-main\results\mask\Domain2.npz'


def main():
    print("🚀 开始重铸 Domain2 伪标签库...")

    # 1. 加载 0.81 金丹模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLab(num_classes=3, backbone='mobilenet').to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()
    print("✅ 模型加载成功！")

    # 2. 获取所有图片
    img_paths = glob(os.path.join(IMG_DIR, '*.png'))
    print(f"📂 发现 {len(img_paths)} 张 Domain2 图片待处理")

    pseudo_dict = {}

    # 3. 逐张推理
    with torch.no_grad():
        for img_p in tqdm(img_paths, desc="生成伪标签"):
            name = os.path.basename(img_p)

            # 读取并进行真理归一化 (与之前验证的 0.70+ 逻辑一致)
            raw_img = Image.open(img_p).convert('RGB').resize((512, 512), Image.BILINEAR)
            img_np = (np.array(raw_img).astype(np.float32) / 127.5) - 1.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            # 推理
            output, _, _ = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # 保存到字典，键名必须是文件名 (如 'G-1-L.png')
            pseudo_dict[name] = pred

    # 4. 打包保存为 .npz
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, **pseudo_dict)

    print("=" * 50)
    print(f"🎉 大功告成！已成功将 {len(pseudo_dict)} 张伪标签保存至:")
    print(f"📁 {SAVE_PATH}")
    print("现在可以去运行 train_target.py 啦！")


if __name__ == '__main__':
    main()