from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import (
    FiLMLayer,
    MetadataEncoder,
    PatchEmbed,
    SwinTransformerBlock,
    UpsampleHead,
)


@dataclass
class SwinIRConfig:
    in_channels: int = 3
    embed_dim: int = 96
    num_blocks: int = 4
    num_heads: int = 6
    window_size: int = 4
    mlp_ratio: float = 4.0
    drop: float = 0.0
    attn_drop: float = 0.0
    out_size: tuple[int, int] = (125, 125)
    upsample_mid_channels: int = 48
    use_global_skip: bool = True
    # FiLM conditioning (Phase 2)
    use_film: bool = False
    num_classes: int = 2
    cond_hidden: int = 64
    cond_dim: int = 128


class SwinIRGenerator(nn.Module):
    """SwinIR-style transformer generator for jet image super-resolution.

    Pipeline:
        LR (B, 3, 64, 64)
          -> PatchEmbed -> tokens (B, 64, 64, 96)
          -> N x SwinTransformerBlock (alternating W-MSA / SW-MSA, optional FiLM)
          -> UpsampleHead (ConvT 2x -> crop to 125 -> refine) -> (B, 3, 125, 125)
          -> (+) bicubic(LR -> 125) global skip
    """

    def __init__(self, cfg: SwinIRConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(in_channels=cfg.in_channels, embed_dim=cfg.embed_dim)

        # Optional FiLM conditioning.
        self.metadata_encoder: MetadataEncoder | None = None
        film_layers: list[nn.Module | None] = [None] * cfg.num_blocks
        if cfg.use_film:
            self.metadata_encoder = MetadataEncoder(
                num_classes=cfg.num_classes, hidden_dim=cfg.cond_hidden, out_dim=cfg.cond_dim
            )
            film_layers = [FiLMLayer(cond_dim=cfg.cond_dim, feature_dim=cfg.embed_dim) for _ in range(cfg.num_blocks)]
            self.film_layers = nn.ModuleList(film_layers)
        else:
            self.film_layers = nn.ModuleList()

        # Swin blocks: alternate shifted / non-shifted.
        blocks: list[nn.Module] = []
        for i in range(cfg.num_blocks):
            shift = 0 if (i % 2 == 0) else cfg.window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    window_size=cfg.window_size,
                    shift_size=shift,
                    mlp_ratio=cfg.mlp_ratio,
                    drop=cfg.drop,
                    attn_drop=cfg.attn_drop,
                    film_layer=film_layers[i],
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.body_norm = nn.LayerNorm(cfg.embed_dim)
        self.body_conv = nn.Conv2d(cfg.embed_dim, cfg.embed_dim, kernel_size=3, padding=1)

        self.upsample = UpsampleHead(
            in_channels=cfg.embed_dim,
            mid_channels=cfg.upsample_mid_channels,
            out_channels=cfg.in_channels,
            out_size=cfg.out_size,
        )

    def encode_metadata(self, pt: Tensor, m0: Tensor, y: Tensor) -> Tensor | None:
        if self.metadata_encoder is None:
            return None
        return self.metadata_encoder(pt, m0, y)

    def forward(
        self,
        lr: Tensor,
        meta: dict[str, Tensor] | None = None,
        store_attn: bool = False,
    ) -> Tensor:
        """lr: (B, 3, H_lr, W_lr). meta optional dict with pt, m0, y if use_film."""
        cond = None
        if self.cfg.use_film:
            assert meta is not None, "FiLM model requires meta={'pt','m0','y'}"
            cond = self.encode_metadata(meta["pt"], meta["m0"], meta["y"])

        tokens = self.patch_embed(lr)  # (B, H, W, C)
        residual = tokens

        for block in self.blocks:
            tokens = block(tokens, cond=cond, store_attn=store_attn)

        tokens = self.body_norm(tokens)
        # Body refinement conv in token space.
        feat = tokens.permute(0, 3, 1, 2).contiguous()
        feat = self.body_conv(feat)
        feat = feat + residual.permute(0, 3, 1, 2).contiguous()
        tokens = feat.permute(0, 2, 3, 1).contiguous()

        out = self.upsample(tokens)  # (B, 3, out_h, out_w)

        if self.cfg.use_global_skip:
            lr_up = F.interpolate(lr, size=self.cfg.out_size, mode="bicubic", align_corners=False)
            out = out + lr_up
        return out


def build_model(name: str, **overrides) -> SwinIRGenerator:
    from .registry import MODEL_REGISTRY

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    cfg_cls = MODEL_REGISTRY[name]
    cfg = cfg_cls(**overrides)
    return SwinIRGenerator(cfg)
