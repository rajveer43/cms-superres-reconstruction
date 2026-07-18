from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """Stride-1 3x3 Conv patch embedding (each pixel becomes a token).

    Input:  (B, in_ch, H, W) raw image
    Output: (B, H, W, embed_dim) token grid + LayerNorm applied on channel dim
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, norm: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)            # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = self.norm(x)
        return x
