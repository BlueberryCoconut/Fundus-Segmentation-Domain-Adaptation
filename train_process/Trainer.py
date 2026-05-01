import os
import torch
import tqdm
import numpy as np
import cv2
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


class Trainer(object):
    def __init__(self, **kwargs):
        # 基础属性初始化
        for k, v in kwargs.items(): setattr(self, k, v)
        self.epoch, self.best_dice = 0, 0.0

        # 🌟 关键点 1：必须设为 none，以便后续对每个像素应用不同的去噪权重
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()
        self.scaler = GradScaler()

        # 🌟 ProDA 核心配置
        self.lambda_momentum = 0.999  # 质心动量更新系数
        self.prototypes = None  # 全局类质心寄存器
        self.tau = 10.0  # 距离缩放温度系数

    def train(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.interval_validate == 0:
                self.validate()

    def train_epoch(self):
        self.model_gen.train()
        # 锁定 BN 统计量[cite: 3]
        for m in self.model_gen.modules():
            if isinstance(m, torch.nn.BatchNorm2d): m.eval()

        tqdm_gen = tqdm.tqdm(enumerate(self.domain_loaderS), total=len(self.domain_loaderS), desc=f"Epoch {self.epoch}")
        for _, sample in tqdm_gen:
            img = sample['image'].cuda()

            # 第一步：获取当前模型的“实时猜测” (不计算梯度)[cite: 4]
            with torch.no_grad():
                out_soft, _, feat_raw = self.model_gen(img)
                # 🌟 创新点：不再使用 static_mask，改用模型当前的预测作为在线伪标签
                live_label = torch.argmax(out_soft, dim=1)

            self.optimizer_gen.zero_grad()
            with autocast(device_type='cuda'):
                # 第二步：正常前向传播
                out, _, feat = self.model_gen(img)

                # 第三步：利用 ProDA 计算当前预测的可信度[cite: 4]
                # 这里传的是实时生成的 live_label
                weights = self.get_proda_weights(feat, live_label)

                # 第四步：计算 Loss
                # 让模型去拟合它自己认为“对”的像素，但 ProDA 权重会压制那些乱猜的噪声像素
                ce_loss_map = self.criterion(out, live_label)

                weights_full = F.interpolate(weights.unsqueeze(1),
                                             size=(img.shape[2], img.shape[3]),
                                             mode='bilinear', align_corners=True).squeeze(1)

                # 最终 Loss
                loss = (ce_loss_map * weights_full).mean() + 0.1 * self.calculate_prototype_loss(feat, live_label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_gen)
            self.scaler.update()
            tqdm_gen.set_postfix(loss=f"{loss.item():.4f}")
    def get_proda_weights(self, feature, mask):
        """ 核心创新：计算 ProDA 在线去噪权重[cite: 1] """
        B, C, H, W = feature.size()
        feat_flat = feature.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]

        # 标签对齐到特征图大小
        label_down = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()

        # 1. 计算当前 Batch 的局部类质心[cite: 1]
        curr_prototypes = torch.zeros(3, C).cuda()
        for k in range(3):  # 0:背景, 1:视杯, 2:视盘
            m = (label_down == k)
            if m.any():
                curr_prototypes[k] = feature.permute(0, 2, 3, 1)[m].mean(0)

        # 2. 动量更新全局质心 η[cite: 1]
        if self.prototypes is None:
            self.prototypes = curr_prototypes.detach()
        else:
            self.prototypes = self.lambda_momentum * self.prototypes + \
                              (1 - self.lambda_momentum) * curr_prototypes.detach()

        # 3. 计算欧式距离距离并转化为 trust weight[cite: 1]
        # 基于公式 (4)：ω = softmax(-dist/τ)
        dists = torch.cdist(feat_flat, self.prototypes, p=2)
        weights_all = F.softmax(-dists / self.tau, dim=1)

        # 提取伪标签指定类别的权重
        label_flat = label_down.reshape(-1, 1)
        pixel_weights = torch.gather(weights_all, 1, label_flat).reshape(B, H, W)

        # 🌟 类别重平衡：进一步压低背景（类0）的引力，强迫模型关注病灶
        class_importance = torch.ones_like(pixel_weights)
        class_importance[label_down == 0] = 0.1

        return pixel_weights * class_importance

    def validate(self):
        self.model_gen.eval()
        dices = []
        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):
                data, target = sample['image'].cuda(), sample['label'].cuda()
                output, _, _ = self.model_gen(data)
                pred = torch.argmax(output, dim=1)

                if i == 0:  # 每一轮保存第一张做对齐检查[cite: 3]
                    p_img = (pred[0].cpu().numpy() * 120).astype(np.uint8)
                    t_img = (target[0].cpu().numpy() * 120).astype(np.uint8)
                    os.makedirs(self.out, exist_ok=True)
                    cv2.imwrite(os.path.join(self.out, f"EP{self.epoch}_PRED.png"), p_img)
                    cv2.imwrite(os.path.join(self.out, f"EP{self.epoch}_GT.png"), t_img)

                dices.append(self.calculate_dice(pred, target))

        cur_dice = np.mean(dices)
        print(f"  [Validation] Dice: {cur_dice:.4f}")
        if cur_dice > self.best_dice:
            self.best_dice = cur_dice
            torch.save(self.model_gen.state_dict(), os.path.join(self.out, 'best_target_model.pth'))

    def calculate_dice(self, pred, target):
        """ 计算 0(视杯) 和 1(视盘) 的平均 Dice[cite: 3] """
        smooth = 1e-5
        dice_list = []
        for i in range(0, 2):
            p = (pred == i).float()
            t = (target == i).float()
            intersection = (p * t).sum()
            dice = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
            dice_list.append(dice.item())
        return np.mean(dice_list)

    def calculate_prototype_loss(self, feature, label):
        """ 原有的特征聚拢约束[cite: 3] """
        B, C, H, W = feature.size()
        label_down = F.interpolate(label.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
        loss_pro, count = 0, 0
        for i in range(3):
            mask = (label_down == i).unsqueeze(1).expand(-1, C, -1, -1)
            if mask.any():
                proto = (feature * mask).sum(dim=(0, 2, 3)) / (mask.sum(dim=(0, 2, 3)) + 1e-6)
                loss_pro += self.mse_loss(feature * mask, proto.view(1, C, 1, 1) * mask)
                count += 1
        return loss_pro / (count + 1e-6)