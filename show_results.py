import numpy as np
import cv2
import os

npz_path = './results/mask/Domain1.npz'
save_path = './results/vis_refined/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

data = np.load(npz_path, allow_pickle=True)

for name in data.files:
    content = data[name]
    if isinstance(content.item(), dict):
        mask = content.item().get('mask', list(content.item().values())[0])
    else:
        mask = content

    mask = np.array(mask)  # 预期形状 (2, 512, 512)

    # 优化：如果只有2维，补齐到3维
    if len(mask.shape) == 2:
        mask = mask[np.newaxis, :]

    # 导出每一个通道，看看视盘到底藏在哪
    for i in range(mask.shape[0]):
        # 将概率转为 0-255 灰度图
        m_img = (mask[i] * 255).astype(np.uint8)
        # 保存文件名：原名_通道号.png
        cv2.imwrite(os.path.join(save_path, f"{name}_ch{i}.png"), m_img)

print(f"✅ 优化后的图片已存至: {save_path}")