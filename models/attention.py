import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelGate(nn.Module):
    """Channel attention: shared MLP for average/max pooling."""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()

        hidden = max(channels // reduction_ratio, 1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)))

        attention = self.mlp(avg_pool) + self.mlp(max_pool)
        scale = torch.sigmoid(attention).view(
            x.shape[0], x.shape[1], 1, 1
        )

        return x * scale


class SpatialGate(nn.Module):
    """Spatial attention with 7x7 convolution."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            1,
            eps=1e-5,
            momentum=0.01
        )

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        compressed = torch.cat(
            [max_pool, avg_pool], dim=1
        )

        scale = torch.sigmoid(
            self.bn(self.conv(compressed))
        )

        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()

        self.channel_gate = ChannelGate(
            channels, reduction_ratio
        )

        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x


class ECA(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()

        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        k_size = max(k_size, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * y
