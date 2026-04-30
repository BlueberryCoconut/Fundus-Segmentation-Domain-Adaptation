import os
import numpy as np
from PIL import Image

# 🌟 请把下面的路径替换成你 Domain1 测试集里的任意一张 mask/label 图片的实际路径
# 注意：找 label 文件夹下的图片，不是 image 文件夹
img_path = r"C:\Users\18268\Desktop\Fundus\Domain1\test\ROIs\mask\gdrishtiGS_001.png" # 假设叫 mask

if os.path.exists(img_path):
    mask = Image.open(img_path)
    mask_np = np.array(mask)

    print("这张标签图里包含的像素值有：", np.unique(mask_np))
else:
    print("找不到图片，请检查路径是否正确！")