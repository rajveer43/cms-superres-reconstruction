from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.normalization import ChannelStats, denormalize


class EnergyResponseLoss(nn.Module):
    """Direct energy-response constraint: mean |Σpred/Σtarget - 1|.

    Penalizes the fractional deviation of predicted total energy from the
    target, so the model cannot hide a 3% bias behind log compression.
    Inputs are in normalized space; denormalized before summing.
    """

    def __init__(self, per_channel: bool = True) -> None:
        super().__init__()
        self.per_channel = per_channel

    def forward(self, pred_norm: Tensor, target_norm: Tensor, stats: ChannelStats) -> Tensor:
        pred_raw = denormalize(pred_norm, stats)
        target_raw = denormalize(target_norm, stats)
        dims = (2, 3) if self.per_channel else (1, 2, 3)
        pred_e = pred_raw.sum(dim=dims)
        target_e = target_raw.sum(dim=dims)
        # Clamp denominator to avoid division by zero on empty jets.
        ratio = pred_e / torch.clamp(target_e, min=1e-6)
        return (ratio - 1.0).abs().mean()


class CombinedLoss(nn.Module):
    """SwinIR objective: lambda_l1 * L1(norm) + lambda_phys * EnergyResponseLoss(raw).

    No adversarial term (intentional ablation vs the GAN baseline).
    """

    def __init__(
        self,
        lambda_l1: float = 50.0,
        lambda_phys: float = 12.0,
        recon: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_phys = lambda_phys
        self.recon = recon if recon is not None else nn.L1Loss()
        self.physics = EnergyResponseLoss(per_channel=True)

    def forward(
        self, pred_norm: Tensor, target_norm: Tensor, stats: ChannelStats
    ) -> tuple[Tensor, dict[str, Tensor]]:
        l1 = self.recon(pred_norm, target_norm)
        phys = self.physics(pred_norm, target_norm, stats)
        total = self.lambda_l1 * l1 + self.lambda_phys * phys
        components = {"l1": l1.detach(), "phys": phys.detach(), "total": total.detach()}
        return total, components
