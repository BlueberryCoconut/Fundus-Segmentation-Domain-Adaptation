import os
import os.path as osp
import torch
import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler

class Trainer(object):
    def __init__(self, cuda, model_gen, model_dis, model_uncertainty_dis,
                 optimizer_gen, optimizer_dis, optimizer_uncertainty_dis,
                 lr_gen, lr_dis, lr_decrease_rate, val_loader,
                 domain_loaderS, domain_loaderT, out, max_epoch, stop_epoch,
                 interval_validate=1, batch_size=4, warmup_epoch=20):
        self.cuda = cuda
        self.model_gen = model_gen
        self.model_dis = model_dis
        self.model_dis2 = model_uncertainty_dis
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis
        self.optimizer_dis2 = optimizer_uncertainty_dis
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.lr_decrease_rate = lr_decrease_rate
        self.val_loader = val_loader
        self.domain_loaderS = domain_loaderS
        self.domain_loaderT = domain_loaderT
        self.out = out
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch
        self.interval_validate = interval_validate
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch

        self.epoch = 0
        self.iteration = 0
        self.best_dice = 0.0

        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        # 🌟 兼容 AMP 混合精度的安全对抗损失函数
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        self.scaler = GradScaler()

    def train(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.interval_validate == 0:
                self.validate()

            # 学习率衰减
            if (self.epoch + 1) % 40 == 0:
                for param_group in self.optimizer_gen.param_groups:
                    param_group['lr'] *= self.lr_decrease_rate

    def train_epoch(self):
        self.model_gen.train()
        self.model_dis.train()
        self.model_dis2.train()

        loaderT_iter = iter(self.domain_loaderT)

        tqdm_gen = tqdm.tqdm(
            enumerate(self.domain_loaderS), total=len(self.domain_loaderS),
            desc='Train epoch=%d' % self.epoch, ncols=100, leave=False)

        for batch_idx, sampleS in tqdm_gen:
            # 自动重置目标域数据迭代器
            try:
                sampleT = next(loaderT_iter)
            except StopIteration:
                loaderT_iter = iter(self.domain_loaderT)
                sampleT = next(loaderT_iter)

            dataS, labelS = sampleS['image'], sampleS['label']
            dataT = sampleT['image']

            if self.cuda:
                dataS, labelS = dataS.cuda(), labelS.cuda()
                dataT = dataT.cuda()

            # --- 1. 训练生成器 ---
            self.optimizer_gen.zero_grad()

            with autocast():
                out_p, _, _ = self.model_gen(dataS)
                loss_seg = self.criterion(out_p, labelS.long())
                loss_total = loss_seg

                # 预热期结束后，加入对抗训练
                if self.epoch >= self.warmup_epoch:
                    out_p_T, _, _ = self.model_gen(dataT)
                    D_out = self.model_dis(torch.softmax(out_p_T, dim=1))

                    loss_adv = self.bce_loss(D_out, torch.ones_like(D_out))
                    loss_total += 0.001 * loss_adv

            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer_gen)
            self.scaler.update()

            # --- 2. 训练判别器 ---
            if self.epoch >= self.warmup_epoch:
                self.optimizer_dis.zero_grad()
                with autocast():
                    out_p_real, _, _ = self.model_gen(dataS)
                    D_real = self.model_dis(torch.softmax(out_p_real.detach(), dim=1))
                    loss_D_real = self.bce_loss(D_real, torch.ones_like(D_real))

                    out_p_fake, _, _ = self.model_gen(dataT)
                    D_fake = self.model_dis(torch.softmax(out_p_fake.detach(), dim=1))
                    loss_D_fake = self.bce_loss(D_fake, torch.zeros_like(D_fake))

                    loss_D = (loss_D_real + loss_D_fake) / 2

                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.optimizer_dis)
                self.scaler.update()

            tqdm_gen.set_postfix(loss='%.4f' % loss_total.item())
            self.iteration += 1

    def validate(self):
        self.model_gen.eval()
        dices = []
        with torch.no_grad():
            for sample in self.val_loader:
                data, target = sample['image'], sample['label']
                if self.cuda:
                    data, target = data.cuda(), target.cuda()

                output, _, _ = self.model_gen(data)
                pred = torch.argmax(output, dim=1)

                dice = self.calculate_dice(pred, target)
                dices.append(dice)

        current_dice = np.mean(dices)
        print(f'\n[Epoch {self.epoch}] Validation Dice: {current_dice:.4f}')

        # 保存最优模型
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model_gen.state_dict(),
                'best_dice': self.best_dice,
            }, osp.join(self.out, 'best_model.pth'))

    def calculate_dice(self, pred, target):
        smooth = 1e-5
        dice_list = []
        for i in range(1, 3):  # 计算视盘(1)和视杯(2)
            p = (pred == i).float()
            t = (target == i).float()
            intersection = (p * t).sum()
            dice = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
            dice_list.append(dice.item())

        return np.mean(dice_list)