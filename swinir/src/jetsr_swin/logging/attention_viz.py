from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from ..models.swinir import SwinIRGenerator


def log_attention_maps(model: SwinIRGenerator, logger, step: int, max_blocks: int = 4) -> None:
    """Log the most recent attention weights cached on each WindowAttention module.

    Requires running a forward pass with `store_attn=True` first.
    """
    figs = {}
    for i, block in enumerate(model.blocks[:max_blocks]):
        attn: Tensor | None = block.attn._last_attn
        if attn is None:
            continue
        # attn: (B_, num_heads, N, N). Average across batch and heads for a clean picture.
        am = attn.mean(dim=(0, 1)).cpu().numpy()
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(am, cmap="viridis")
        ax.set_title(f"block {i} attention (mean over heads, batch)")
        ax.set_xlabel("key token")
        ax.set_ylabel("query token")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        figs[f"attention/block_{i}"] = fig

    for name, fig in figs.items():
        logger.log_image(name, fig, step=step)


def log_rel_pos_bias(model: SwinIRGenerator, logger, step: int) -> None:
    """Log learned (eta, phi) relative-positional-bias matrices per block (head 0)."""
    for i, block in enumerate(model.blocks):
        bias = block.attn.get_relative_position_bias()  # (H, N, N)
        bias_head0 = bias[0].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(bias_head0, cmap="coolwarm")
        ax.set_title(f"block {i} rel-pos bias (head 0)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        logger.log_image(f"rel_pos_bias/block_{i}", fig, step=step)
