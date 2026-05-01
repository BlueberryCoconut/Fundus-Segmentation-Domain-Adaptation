from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from dataloaders.mypath import MYPath

class FundusSegmentation(Dataset):
    def __init__(self, base_dir=MYPath.db_root_dir('fundus'), dataset='Domain1', split='train/ROIs',
                 transform=None, pseudo_path=None):
        self._base_dir = base_dir
        self.dataset = dataset
        self.pseudo_path = pseudo_path
        self.pseudo_labels = None
        self.pseudo_keys = {}

        # 1. 加载伪标签 (仅训练阶段使用)[cite: 8]
        if self.pseudo_path is not None and os.path.exists(self.pseudo_path):
            self.pseudo_labels = np.load(self.pseudo_path)
            self.pseudo_keys = {os.path.basename(k).lower(): k for k in self.pseudo_labels.files}

        # 2. 扫描图像路径[cite: 8]
        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        imagelist = glob(self._image_dir + "/*.png") + glob(self._image_dir + "/*.jpg")

        self.image_list = []
        for image_path in imagelist:
            full_name = os.path.basename(image_path).lower()
            if self.pseudo_path is None or full_name in self.pseudo_keys:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})

        self.transform = transform
        print(f'✅ 加载 {len(self.image_list)} 张图片 (模式: {"伪标签训练" if self.pseudo_path else "标准验证"})')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _full_name = os.path.basename(self.image_list[index]['image'])
        _pure_name = _full_name.lower()

        _target = None
        if self.pseudo_labels is not None:
            target_key = self.pseudo_keys.get(_pure_name) or self.pseudo_keys.get(_pure_name.rsplit('.', 1)[0])
            if target_key:
                _target = Image.fromarray(self.pseudo_labels[target_key])

        if _target is None:
            _target_np = np.array(Image.open(self.image_list[index]['label']).convert('L'))
            label = np.zeros_like(_target_np)

            # 🌟 真理映射 (方案 A)：0=黑(杯), 1=灰(盘), 2=白(背景)
            label[_target_np < 64] = 0  # 黑色 -> 0
            label[(_target_np >= 64) & (_target_np <= 192)] = 1  # 灰色 -> 1
            label[_target_np > 192] = 2  # 白色 -> 2

            _target = Image.fromarray(label)

        sample = {'image': _img, 'label': _target, 'img_name': _full_name}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    