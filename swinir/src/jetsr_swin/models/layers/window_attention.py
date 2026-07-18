from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """(B, H, W, C) -> (num_windows * B, window_size, window_size, C)."""
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, (
        f"H={H}, W={W} must be divisible by window_size={window_size}"
    )
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """(num_windows * B, window_size, window_size, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with eta-phi relative positional bias.

    The bias is indexed by relative (delta_eta, delta_phi) coordinates within a
    window. For calorimeter images, dim -2 represents eta (pseudorapidity) and
    dim -1 represents phi (azimuth) by convention. Within a small window
    (window_size <= 4 in our config), phi wrap-around is negligible so we treat
    it as a planar relative bias. A pluggable cyclic variant can replace this.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Learned relative positional bias for (2*Wh-1) * (2*Ww-1) positions per head.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index for each token pair inside a window.
        coords_eta = torch.arange(window_size)
        coords_phi = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_eta, coords_phi, indexing="ij"))  # (2, Wh, Ww)
        coords_flat = coords.flatten(1)                                              # (2, Wh*Ww)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]                      # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()                                      # (N, N, 2)
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_pos_index = rel.sum(-1)                                                  # (N, N)
        self.register_buffer("relative_position_index", rel_pos_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._last_attn: Tensor | None = None  # for visualization

    def get_relative_position_bias(self) -> Tensor:
        N = self.window_size * self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        return bias  # (num_heads, N, N)

    def forward(self, x: Tensor, mask: Tensor | None = None, store_attn: bool = False) -> Tensor:
        """x: (num_windows*B, N=Wh*Ww, C); mask: (num_windows, N, N) or None."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # (3, B_, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)          # (B_, H, N, N)
        attn = attn + self.get_relative_position_bias().unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        if store_attn:
            self._last_attn = attn.detach()
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
