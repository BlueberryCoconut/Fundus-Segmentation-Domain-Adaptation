import torch
import random
import numpy as np
from PIL import Image


class Normalize_tf(object):
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        # 🌟 保持原样，这就是模型习惯的视力！
        img = (img / 127.5) - 1.0
        sample['image'] = img
        return sample

class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # 处理图像：HWC -> CHW
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2, 0, 1).float()

        # 处理标签：确保是 Long 类型且值域正确
        mask_np = np.array(mask).astype(np.int64)
        mask_np = np.clip(mask_np, 0, 2)
        mask_tensor = torch.from_numpy(mask_np).long()

        sample.update({'image': img_tensor, 'label': mask_tensor})
        return sample

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, sample):
        sample['image'] = sample['image'].resize((self.size, self.size), Image.BILINEAR)
        sample['label'] = sample['label'].resize((self.size, self.size), Image.NEAREST)
        return sample