import os
import torch
import tqdm
import numpy as np
import cv2
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


class Trainer(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
        self.epoch, self.best_dice = 0, 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.scaler = GradScaler()

    def train(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.interval_validate == 0:
                self.validate()

    def train_epoch(self):
        self.model_gen.train()
        tqdm_gen = tqdm.tqdm(enumerate(self.domain_loaderS), total=len(self.domain_loaderS), desc=f"Epoch {self.epoch}")
        for _, sample in tqdm_gen:
            img, mask = sample['image'].cuda(), sample['label'].cuda()
            self.optimizer_gen.zero_grad()
            with autocast(device_type='cuda'):
                out, _, feat = self.model_gen(img)
                loss = self.criterion(out, mask.long()) + 0.1 * self.calculate_prototype_loss(feat, mask)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_gen)
            self.scaler.update()
            tqdm_gen.set_postfix(loss=f"{loss.item():.4f}")

    def validate(self):
        self.model_gen.eval()
        dices = []
        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):
                data, target = sample['image'].cuda(), sample['label'].cuda()
                output, _, _ = self.model_gen(data)
                pred = torch.argmax(output, dim=1)


                if i == 0:  # 每一轮保存第一张做对齐检查
                    # 可视化缩放：0->0, 1->120(灰), 2->240(白)
                    p_img = (pred[0].cpu().numpy() * 120).astype(np.uint8)
                    t_img = (target[0].cpu().numpy() * 120).astype(np.uint8)
                    cv2.imwrite(os.path.join(self.out, f"EP{self.epoch}_PRED.png"), p_img)
                    cv2.imwrite(os.path.join(self.out, f"EP{self.epoch}_GT.png"), t_img)

                dices.append(self.calculate_dice(pred, target))

        cur_dice = np.mean(dices)
        print(f"  [Validation] Dice: {cur_dice:.4f}")
        if cur_dice > self.best_dice:
            self.best_dice = cur_dice
            torch.save(self.model_gen.state_dict(), os.path.join(self.out, 'best_target_model.pth'))


    def calculate_dice(self, pred, target):
        """ 计算 0(视杯) 和 1(视盘) 的平均 Dice """
        smooth = 1e-5
        dice_list = []
        # 🌟 关键：计算索引 0 和 1
        for i in range(0, 2):
            p = (pred == i).float()
            t = (target == i).float()
            intersection = (p * t).sum()
            dice = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
            dice_list.append(dice.item())
        return np.mean(dice_list)

    def calculate_prototype_loss(self, feature, label):
        B, C, H, W = feature.size()
        label_down = F.interpolate(label.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
        loss_pro, count = 0, 0
        for i in range(3):  # 遍历 0, 1, 2 三个类别[cite: 7]
            mask = (label_down == i).unsqueeze(1).expand(-1, C, -1, -1)
            if mask.any():
                proto = (feature * mask).sum(dim=(0, 2, 3)) / (mask.sum(dim=(0, 2, 3)) + 1e-6)
                loss_pro += self.mse_loss(feature * mask, proto.view(1, C, 1, 1) * mask)
                count += 1
        return loss_pro / (count + 1e-6)