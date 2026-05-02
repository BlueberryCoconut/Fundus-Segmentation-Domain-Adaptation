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

        # 🌟 必杀技 1.1：设置 ignore_index=255，让低于置信度的像素不参与 Loss 计算
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=255)
        self.mse_loss = torch.nn.MSELoss()
        self.scaler = GradScaler()

        # 🌟 必杀技 3：动量调优（加快质心更新）
        self.lambda_momentum = 0.90
        self.prototypes = None
        self.tau = 10.0

    def train(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.interval_validate == 0:
                self.validate()

    def train_epoch(self):
        self.model_gen.train()

        # 🌟 必杀技 2：动态温度衰减 (从 15 降到 2)
        current_tau = 15.0 - (13.0 * (self.epoch / max(1, self.max_epoch - 1)))
        self.tau = max(2.0, current_tau)
        if self.epoch % self.interval_validate == 0:
            print(f"  [ProDA Info] Current Temperature (tau): {self.tau:.2f}")

        for m in self.model_gen.modules():
            if isinstance(m, torch.nn.BatchNorm2d): m.eval()

        tqdm_gen = tqdm.tqdm(enumerate(self.domain_loaderS), total=len(self.domain_loaderS), desc=f"Epoch {self.epoch}")
        for _, sample in tqdm_gen:
            img = sample['image'].cuda()

            # --- Live Label 生成与置信度截断 ---
            with torch.no_grad():
                out_raw, _, feat_raw = self.model_gen(img)
                out_prob = F.softmax(out_raw, dim=1)
                max_prob, live_label = torch.max(out_prob, dim=1)

                # 🌟 必杀技 1.2：置信度低于 0.75 的直接丢弃 (设为 255)
                live_label[max_prob < 0.75] = 255

            self.optimizer_gen.zero_grad()
            with autocast(device_type='cuda'):
                out, _, feat = self.model_gen(img)

                # 计算 ProDA 权重
                weights = self.get_proda_weights(feat, live_label)

                ce_loss_map = self.criterion(out, live_label)
                valid_mask = (live_label != 255).float()
                weights_full = F.interpolate(weights.unsqueeze(1), size=(img.shape[2], img.shape[3]), mode='bilinear',
                                             align_corners=True).squeeze(1)

                valid_count = valid_mask.sum() + 1e-6
                weighted_ce_loss = (ce_loss_map * weights_full * valid_mask).sum() / valid_count

                loss = weighted_ce_loss + 0.1 * self.calculate_prototype_loss(feat, live_label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_gen)
            self.scaler.update()
            tqdm_gen.set_postfix(loss=f"{loss.item():.4f}")

    def get_proda_weights(self, feature, mask):
        B, C, H, W = feature.size()
        feat_flat = feature.permute(0, 2, 3, 1).reshape(-1, C)

        label_down = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()

        curr_prototypes = torch.zeros(3, C).cuda()
        for k in range(3):
            # 🌟 必杀技 1.3：非 255 的有效像素才参与质心计算
            m = (label_down == k) & (label_down != 255)
            if m.any():
                curr_prototypes[k] = feature.permute(0, 2, 3, 1)[m].mean(0)

        if self.prototypes is None:
            self.prototypes = curr_prototypes.detach()
        else:
            self.prototypes = self.lambda_momentum * self.prototypes + (
                        1 - self.lambda_momentum) * curr_prototypes.detach()

        dists = torch.cdist(feat_flat, self.prototypes, p=2)
        weights_all = F.softmax(-dists / self.tau, dim=1)

        label_flat = label_down.reshape(-1, 1)
        safe_label_flat = torch.clamp(label_flat, max=2)
        pixel_weights = torch.gather(weights_all, 1, safe_label_flat).reshape(B, H, W)
        pixel_weights[label_down == 255] = 0.0

        # 🌟 核心修正：压制背景，此时背景的标签是 2！
        class_importance = torch.ones_like(pixel_weights)
        class_importance[label_down == 2] = 0.1

        return pixel_weights * class_importance

    def validate(self):
        self.model_gen.eval()
        cup_dices, disc_dices = [], []

        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):
                data, target = sample['image'].cuda(), sample['label'].cuda()
                output, _, feat = self.model_gen(data)
                pred = torch.argmax(output, dim=1)

                if i == 0:
                    p_img = (pred[0].cpu().numpy() * 120).astype(np.uint8)
                    t_img = (target[0].cpu().numpy() * 120).astype(np.uint8)
                    os.makedirs(self.out, exist_ok=True)
                    cv2.imwrite(os.path.join(self.out, f"EP{self.epoch}_PRED.png"), p_img)
                    cv2.imwrite(os.path.join(self.out, f"EP{self.epoch}_GT.png"), t_img)

                    # 保存权重热力图
                    if self.prototypes is not None:
                        weights = self.get_proda_weights(feat, pred)
                        w_map = weights[0].cpu().numpy()
                        w_gray = (w_map * 255).astype(np.uint8)
                        w_color = cv2.applyColorMap(w_gray, cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join(self.out, f"EP{self.epoch}_WEIGHT.png"), w_color)

                d_cup, d_disc = self.calculate_dice(pred, target)
                cup_dices.append(d_cup)
                disc_dices.append(d_disc)

        avg_cup = np.mean(cup_dices)
        avg_disc = np.mean(disc_dices)
        cur_dice = (avg_cup + avg_disc) / 2.0

        print(f"  [Validation] Cup: {avg_cup:.4f} | Disc: {avg_disc:.4f} | Avg: {cur_dice:.4f}")

        if cur_dice > self.best_dice:
            self.best_dice = cur_dice
            print(f"  🏆 New Best Avg Dice: {cur_dice:.4f}! Model saved.")
            torch.save(self.model_gen.state_dict(), os.path.join(self.out, 'best_target_model.pth'))

    def calculate_dice(self, pred, target):
        smooth = 1e-5

        # 🌟 核心修正：0 是视杯
        p_cup = (pred == 0).float()
        t_cup = (target == 0).float()
        dice_cup = (2. * (p_cup * t_cup).sum() + smooth) / (p_cup.sum() + t_cup.sum() + smooth)

        # 🌟 核心修正：0(杯) + 1(盘边缘) = 完整视盘
        p_disc = (pred <= 1).float()
        t_disc = (target <= 1).float()
        dice_disc = (2. * (p_disc * t_disc).sum() + smooth) / (p_disc.sum() + t_disc.sum() + smooth)

        return dice_cup.item(), dice_disc.item()

    def calculate_prototype_loss(self, feature, label):
        B, C, H, W = feature.size()
        label_down = F.interpolate(label.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
        loss_pro, count = 0, 0
        for i in range(3):
            # 🌟 仅在非 255 像素上计算
            mask = ((label_down == i) & (label_down != 255)).unsqueeze(1).expand(-1, C, -1, -1)
            if mask.any():
                proto = (feature * mask).sum(dim=(0, 2, 3)) / (mask.sum(dim=(0, 2, 3)) + 1e-6)
                loss_pro += self.mse_loss(feature * mask, proto.view(1, C, 1, 1) * mask)
                count += 1
        return loss_pro / (count + 1e-6)