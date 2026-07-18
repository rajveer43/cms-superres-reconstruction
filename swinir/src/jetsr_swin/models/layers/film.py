from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MetadataEncoder(nn.Module):
    """Encode (pt_norm, m0_norm, y_onehot[K]) -> latent of dim `out_dim`."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        in_dim = 2 + num_classes
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pt: Tensor, m0: Tensor, y: Tensor) -> Tensor:
        y_oh = nn.functional.one_hot(y.long(), num_classes=self.num_classes).to(pt.dtype)
        x = torch.cat([pt.unsqueeze(-1), m0.unsqueeze(-1), y_oh], dim=-1)
        return self.net(x)


class FiLMLayer(nn.Module):
    """Per-block FiLM: takes a metadata latent and produces (gamma, beta) of dim C.

    Applied to LayerNorm-ed tokens as: x = (1 + gamma) * x + beta.
    The `1 +` makes the identity (gamma=0, beta=0) a safe initial state, so an
    untrained FiLM doesn't destabilize the base model.
    """

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.feature_dim = feature_dim

    def forward(self, x: Tensor, cond: tuple[Tensor, Tensor] | Tensor) -> Tensor:
        """x: (B, H, W, C) or (B, N, C); cond: latent (B, cond_dim) or pre-split tuple."""
        if isinstance(cond, tuple):
            gamma, beta = cond
        else:
            params = self.proj(cond)                          # (B, 2C)
            gamma, beta = params.chunk(2, dim=-1)

        # Broadcast to x rank.
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return (1.0 + gamma) * x + beta

    def compute_params(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta
