"""Multi-scale LR generation — the core of the Option A study.

The experiment downsamples a fixed HR ground truth to several LR scales
(e.g. 64, 32, 16) and trains an independent GAN to reconstruct HR from each.
This module centralises the downsampling so parquet and HDF5 paths behave
identically.

Energy preservation: we use mode="area" for downsampling. Area (average)
pooling sums-then-averages over the receptive cell, so the *mean* per cell is
preserved; total energy after downsample scales by (lr_h*lr_w)/(hr_h*hr_w).
This is deterministic and exactly invertible in expectation under the
log1p+z-score pipeline, and it does not introduce the ringing/negative
overshoot that bicubic produces on sparse calorimeter deposits.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def downscale_hr(hr: Tensor, scale: int) -> Tensor:
    """Downsample an HR image (or batch) to (scale, scale) via area averaging.

    Accepts (C, H, W) or (B, C, H, W); returns the same rank. Output is
    clamped to be non-negative (energies cannot go below zero).
    """
    squeeze = False
    if hr.dim() == 3:
        hr = hr.unsqueeze(0)
        squeeze = True
    elif hr.dim() != 4:
        raise ValueError(f"Expected (C,H,W) or (B,C,H,W), got shape {tuple(hr.shape)}")

    lr = F.interpolate(hr.float(), size=(scale, scale), mode="area").clamp_min(0.0)
    return lr.squeeze(0) if squeeze else lr
