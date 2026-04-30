import torch.nn as nn
import torch

class BoundaryDiscriminator(nn.Module):
    def __init__(self, in_channels=3): # 🌟 核心修复：支持 3 通道预测输入[cite: 7, 8]
        super(BoundaryDiscriminator, self).__init__()
        filter_num_list = [64, 128, 256, 512, 1]

        # 第一层卷积输入必须与预测图通道数（3）一致[cite: 7, 8]
        self.conv1 = nn.Conv2d(in_channels, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        return self.conv5(x)

class UncertaintyDiscriminator(nn.Module):
    def __init__(self, in_channels=3): # 🌟 同步修改为 3 通道[cite: 7, 8]
        super(UncertaintyDiscriminator, self).__init__()
        filter_num_list = [64, 128, 256, 512, 1]
        self.conv1 = nn.Conv2d(in_channels, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        return self.conv5(x)

