from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class UpsampleHead(nn.Module):
    """ConvTranspose2d 2x upsample then center-crop to target then refine.

    Default path: 96-ch (64x64) -> 48-ch (128x128) -> center-crop to (out_h, out_w)
    -> Conv-GELU-Conv -> out_channels.
    """

    def __init__(
        self,
        in_channels: int = 96,
        mid_channels: int = 48,
        out_channels: int = 3,
        out_size: tuple[int, int] = (125, 125),
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, H, W, C). Returns (B, out_channels, out_h, out_w)."""
        x = tokens.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.up(x)                                # 2x spatial
        x = self._center_crop(x, self.out_size)
        x = self.refine(x)
        return x

    @staticmethod
    def _center_crop(x: Tensor, size: tuple[int, int]) -> Tensor:
        _, _, H, W = x.shape
        th, tw = size
        if H == th and W == tw:
            return x
        top = max(0, (H - th) // 2)
        left = max(0, (W - tw) // 2)
        return x[:, :, top : top + th, left : left + tw]
