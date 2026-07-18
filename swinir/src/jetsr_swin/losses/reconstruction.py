from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class L1Loss(nn.Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(pred, target)


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant: sqrt((x-y)^2 + eps^2)."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()
