from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

CHANNEL_NAMES = ("Tracks", "ECAL", "HCAL")  # CMS jet image channel convention


def _to_numpy_log(img: Tensor) -> np.ndarray:
    return np.log1p(np.clip(img.detach().cpu().float().numpy(), a_min=0.0, a_max=None))


def sample_panel(
    lr_raw: Tensor,
    pred_raw: Tensor,
    hr_raw: Tensor,
    sample_idx: int = 0,
) -> plt.Figure:
    """Build a per-channel LR | Pred | HR | Residual panel for one sample.

    Inputs are RAW (denormalized) intensities of shape (B, 3, H, W).
    """
    lr = _to_numpy_log(lr_raw[sample_idx])
    pred = _to_numpy_log(pred_raw[sample_idx])
    hr = _to_numpy_log(hr_raw[sample_idx])
    C = lr.shape[0]

    # Upsample LR for visual comparison.
    lr_up = F.interpolate(lr_raw[sample_idx : sample_idx + 1], size=hr_raw.shape[-2:], mode="bicubic", align_corners=False)
    lr_up_log = _to_numpy_log(lr_up[0])

    fig, axes = plt.subplots(C, 4, figsize=(12, 3 * C))
    if C == 1:
        axes = axes[None, :]
    for c in range(C):
        vmax = max(pred[c].max(), hr[c].max(), 1e-6)
        axes[c, 0].imshow(lr_up_log[c], cmap="inferno", vmin=0, vmax=vmax)
        axes[c, 0].set_title(f"{CHANNEL_NAMES[c]} | LR (bicubic up)")
        axes[c, 1].imshow(pred[c], cmap="inferno", vmin=0, vmax=vmax)
        axes[c, 1].set_title(f"{CHANNEL_NAMES[c]} | Pred")
        axes[c, 2].imshow(hr[c], cmap="inferno", vmin=0, vmax=vmax)
        axes[c, 2].set_title(f"{CHANNEL_NAMES[c]} | HR")
        diff = pred[c] - hr[c]
        absmax = max(abs(diff.min()), abs(diff.max()), 1e-6)
        axes[c, 3].imshow(diff, cmap="seismic", vmin=-absmax, vmax=absmax)
        axes[c, 3].set_title(f"{CHANNEL_NAMES[c]} | Pred - HR")
        for ax in axes[c]:
            ax.axis("off")
    fig.tight_layout()
    return fig


def residual_map(pred_raw: Tensor, hr_raw: Tensor, sample_idx: int = 0) -> plt.Figure:
    pred = pred_raw[sample_idx].detach().cpu().float().numpy()
    hr = hr_raw[sample_idx].detach().cpu().float().numpy()
    diff = pred - hr
    C = diff.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
    if C == 1:
        axes = [axes]
    for c in range(C):
        absmax = max(abs(diff[c].min()), abs(diff[c].max()), 1e-6)
        im = axes[c].imshow(diff[c], cmap="seismic", vmin=-absmax, vmax=absmax)
        axes[c].set_title(f"{CHANNEL_NAMES[c]} residual")
        axes[c].axis("off")
        fig.colorbar(im, ax=axes[c], fraction=0.046)
    fig.tight_layout()
    return fig
