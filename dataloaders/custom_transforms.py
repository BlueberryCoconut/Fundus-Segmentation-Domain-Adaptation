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

        # 🌟 修复 ValueError：确保缩放后的短边至少是 base_size + 1
        short_size = random.randint(self.base_size + 1, int(self.base_size * 1.5))

        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        img = img.resize((ow, oh), Image.BILINEAR)
        # 🌟 标签必须用 NEAREST，尽量减少插值模糊
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
        # 🌟 同样，旋转也要用 NEAREST
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        return {'image': img, 'label': mask}


class adjust_light(object):
    def __call__(self, sample):
        img = sample['image']
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        # 🌟 使用补全的 transforms 模块进行光照增强
        img = transforms.functional.adjust_brightness(img, brightness)
        img = transforms.functional.adjust_contrast(img, contrast)
        return {'image': img, 'label': sample['label']}


class Normalize_tf(object):
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        img /= 127.5
        img -= 1.0
        return {'image': img, 'label': sample['label']}


class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        mask = np.array(sample['label']).astype(np.int64)

        # 🌟 终极杀手锏：区间阈值法
        # 即使数据加载时发生了插值模糊（Bilinear），也能强行拉回正确分类
        tmp_mask = np.zeros_like(mask)

        # [64, 192) 之间的灰色过渡像素 -> 强制归为视盘 (1)
        tmp_mask[(mask >= 64) & (mask < 192)] = 1

        # >= 192 的高亮像素 -> 强制归为视杯 (2)
        tmp_mask[mask >= 192] = 2

        mask = tmp_mask

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return {'image': img, 'label': mask}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        return {'image': img, 'label': mask}