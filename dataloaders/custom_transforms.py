import torch
import random
import numpy as np
from PIL import Image
# 🌟 必须保留这行导入，否则 adjust_light 会报错
from torchvision import transforms


class RandomScaleCrop(object):
    def __init__(self, base_size):
        self.base_size = base_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size

        # 确保缩放后的短边至少是 base_size + 1
        short_size = random.randint(self.base_size + 1, int(self.base_size * 1.5))

        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        img = img.resize((ow, oh), Image.BILINEAR)
        # 标签必须用 NEAREST，尽量减少插值模糊
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Crop
        w, h = img.size
        x1 = random.randint(0, w - self.base_size)
        y1 = random.randint(0, h - self.base_size)
        img = img.crop((x1, y1, x1 + self.base_size, y1 + self.base_size))
        mask = mask.crop((x1, y1, x1 + self.base_size, y1 + self.base_size))

        return {'image': img, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': mask}


class RandomRotate(object):
    def __init__(self, degree=15):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        return {'image': img, 'label': mask}


class adjust_light(object):
    def __call__(self, sample):
        img = sample['image']
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        img = transforms.functional.adjust_brightness(img, brightness)
        img = transforms.functional.adjust_contrast(img, contrast)
        return {'image': img, 'label': sample['label']}


class Normalize_tf(object):
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        img /= 127.5
        img -= 1.0

        # 将原有的其他键值一起传递下去（很重要，否则 img_name 会丢失）
        out_sample = {'image': img, 'label': sample['label']}
        for key in sample.keys():
            if key not in ['image', 'label']:
                out_sample[key] = sample[key]

        return out_sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # --- 1. 处理图片 ---
        # 如果经过了 Normalize_tf，img 已经是 numpy 数组，直接转 Tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            # 防错：如果没经过归一化，默认转 numpy 并除以 255
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).float()

        # --- 2. 处理标签 (干净利落，绝不除以 255) ---
        mask_np = np.array(mask).astype(np.int64)

        # 🌟 极限防错：强行确保标签只包含 0, 1, 2，砍掉任何越界值或插值产生的毛刺
        mask_np = np.clip(mask_np, 0, 2)

        # 分割任务的标签必须是 LongTensor
        mask_tensor = torch.from_numpy(mask_np).long()

        # --- 3. 组装返回 ---
        sample_dict = {'image': img_tensor, 'label': mask_tensor}

        # 动态传递所有额外信息（如 img_name 等，防止报 KeyError）
        for key in sample.keys():
            if key not in ['image', 'label']:
                sample_dict[key] = sample[key]

        return sample_dict


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        return {'image': img, 'label': mask}