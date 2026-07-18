from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


def l1_raw(pred: Tensor, target: Tensor) -> float:
    return F.l1_loss(pred, target).item()


def psnr(pred: Tensor, target: Tensor, max_val: float | None = None) -> float:
    """PSNR in dB. If max_val is None, use the target's max (per-batch)."""
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return float("inf")
    if max_val is None:
        max_val = float(target.max().item())
        if max_val <= 0:
            max_val = 1.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def psnr_norm(pred_norm: Tensor, target_norm: Tensor) -> float:
    """PSNR in dB using max_val=1.0 on channel-normalized tensors.

    Comparable to the GAN baseline which reports PSNR on normalized data.
    Use this metric for cross-model comparisons, not the raw-space psnr().
    """
    return psnr(pred_norm, target_norm, max_val=1.0)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    return window


def ssim_2d(pred: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """SSIM averaged across batch and channels. Inputs (B, C, H, W) in raw scale."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have same shape")
    B, C, H, W = pred.shape
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2

    data_range = float(max(target.max().item(), pred.max().item(), 1.0))
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def energy_response(pred_raw: Tensor, target_raw: Tensor) -> tuple[float, float]:
    """Per-jet (Σ pred / Σ target). Returns (mean, std) across the batch."""
    pe = pred_raw.sum(dim=(1, 2, 3))
    te = target_raw.sum(dim=(1, 2, 3))
    r = (pe / torch.clamp(te, min=1e-8)).detach().cpu()
    return float(r.mean().item()), float(r.std().item())


@dataclass
class MetricAccumulator:
    """Streaming accumulator for per-batch metrics (means + per-class slices)."""

    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_class_totals: dict[tuple[int, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    per_class_counts: dict[tuple[int, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def update(self, name: str, value: float, count: int = 1) -> None:
        self.totals[name] += value * count
        self.counts[name] += count

    def update_per_class(self, name: str, value_by_class: dict[int, tuple[float, int]]) -> None:
        for cls, (val, cnt) in value_by_class.items():
            self.per_class_totals[(cls, name)] += val * cnt
            self.per_class_counts[(cls, name)] += cnt

    def mean(self, name: str) -> float:
        c = self.counts[name]
        return self.totals[name] / c if c > 0 else 0.0

    def class_mean(self, cls: int, name: str) -> float:
        c = self.per_class_counts[(cls, name)]
        return self.per_class_totals[(cls, name)] / c if c > 0 else 0.0

    def summary(self) -> dict[str, float]:
        out = {k: self.mean(k) for k in self.counts}
        # per-class slices
        classes = sorted({c for (c, _) in self.per_class_counts})
        for cls in classes:
            for name in {n for (_, n) in self.per_class_counts}:
                out[f"class{cls}/{name}"] = self.class_mean(cls, name)
        # response gap (|class0 - class1|) on energy_response_mean if present
        if 0 in classes and 1 in classes and "energy_response_mean" in self.counts:
            out["response_gap"] = abs(
                self.class_mean(0, "energy_response_mean") - self.class_mean(1, "energy_response_mean")
            )
        return out
