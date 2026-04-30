from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
import imgaug.augmenters as iaa
import dataloaders.net as net
from dataloaders.utils import *
from dataloaders.mypath import MYPath
import cv2
from torch.utils.data import DataLoader
import torch.nn as nn


class FundusSegmentation(Dataset):
    """
    负责加载训练集 (REFUGE)
    🚨 官方格式：黑底(0), 灰盘(128), 白杯(255)
    """

    def __init__(self,
                 base_dir=MYPath.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        print(self._image_dir)
        imagelist = glob(self._image_dir + "/*.png")
        for image_path in imagelist:
            gt_path = image_path.replace('image/', 'mask/')
            self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target_pil = Image.open(self.image_list[index]['label'])

        if _target_pil.mode == 'RGB':
            _target_pil = _target_pil.convert('L')

        _target_np = np.array(_target_pil)
        label = np.zeros_like(_target_np)

        # 🌟 REFUGE 专属翻译规则 (黑底白杯)
        label[_target_np < 64] = 0       # 黑底 -> 类别 0 (背景)
        label[(_target_np >= 64) & (_target_np <= 192)] = 1  # 灰盘 -> 类别 1 (视盘)
        label[_target_np > 192] = 2      # 白杯 -> 类别 2 (视杯)

        _target = Image.fromarray(label)
        _img_name = self.image_list[index]['image'].split('/')[-1]

        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'image1': _img}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample


class FundusSegmentation_pgd(Dataset):
    """
    负责加载测试集 (Domain1)
    🚨 官方格式：白底(255), 灰盘(128), 黑杯(0)
    """

    def __init__(self,
                 base_dir=MYPath.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        print(self._image_dir + '/pgd')
        imagelist = glob(self._image_dir + "/*.png")

        for image_path in imagelist:
            gt_path = image_path.replace('image', 'mask')
            p1_path = image_path.replace('Domain1/test/ROIs/image/', 'PGD/DPL/Domain1/')
            self.image_list.append({'image': p1_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target_pil = Image.open(self.image_list[index]['label'])

        if _target_pil.mode == 'RGB':
            _target_pil = _target_pil.convert('L')

        _target_np = np.array(_target_pil)
        label = np.zeros_like(_target_np)

        # 🌟 Domain1 专属翻译规则 (白底黑杯)
        label[_target_np > 192] = 0      # 白底 -> 类别 0 (背景)
        label[(_target_np >= 64) & (_target_np <= 192)] = 1  # 灰盘 -> 类别 1 (视盘)
        label[_target_np < 64] = 2       # 黑杯 -> 类别 2 (视杯)

        _target = Image.fromarray(label)
        _img_name = self.image_list[index]['image'].split('/')[-1]

        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample
    
    