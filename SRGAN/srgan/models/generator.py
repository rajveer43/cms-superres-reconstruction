from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.1 * self.block(x)


class Generator(nn.Module):
    """Bicubic-residual generator.

    Upsizes via bicubic interpolation to target_size, then adds a learned
    residual correction. target_size is passed at forward time so the same
    checkpoint works for any output resolution.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_blocks: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_channels))
        layers.extend([
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, lr: Tensor, target_size: tuple[int, int]) -> Tensor:
        lr_up = F.interpolate(lr, size=target_size, mode="bicubic", align_corners=False)
        return lr_up + self.net(lr_up)
