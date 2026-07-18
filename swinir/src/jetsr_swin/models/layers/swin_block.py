from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .window_attention import WindowAttention, window_partition, window_reverse


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwinTransformerBlock(nn.Module):
    """Single Swin block: LN -> (S)W-MSA -> residual -> LN -> MLP -> residual.

    Operates on a fixed-resolution token grid (H, W). Padding is added if H or W
    is not divisible by window_size.

    FiLM hook: if `film_layer` is provided, after the first LN the tokens are
    modulated as x = gamma * x + beta with (gamma, beta) computed from metadata
    by the caller (passed via the `cond` argument). When cond/film_layer is None
    this block reduces to vanilla Swin.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        film_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

        self.film_layer = film_layer

    def _attn_mask(self, H: int, W: int, device: torch.device) -> Tensor | None:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = h_slices
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(
        self,
        x: Tensor,
        cond: tuple[Tensor, Tensor] | None = None,
        store_attn: bool = False,
    ) -> Tensor:
        """x: (B, H, W, C); cond: optional (gamma, beta) each of shape (B, C)."""
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)
        if self.film_layer is not None and cond is not None:
            x = self.film_layer(x, cond)

        # Pad so H and W are divisible by window_size.
        ws = self.window_size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        # Cyclic shift.
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        # Partition + attention.
        x_windows = window_partition(shifted, ws).view(-1, ws * ws, C)
        attn_mask = self._attn_mask(Hp, Wp, x.device) if self.shift_size > 0 else None
        attn_windows = self.attn(x_windows, mask=attn_mask, store_attn=store_attn)
        attn_windows = attn_windows.view(-1, ws, ws, C)
        shifted = window_reverse(attn_windows, ws, Hp, Wp)

        # Reverse cyclic shift.
        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
