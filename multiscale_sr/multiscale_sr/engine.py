"""Training/eval engine: losses, metrics, evaluation, and figure rendering.

Shared by train.py and evaluate.py so the loss formulation and metric
definitions are identical between training and standalone evaluation.

Loss (matches the SRGAN baseline):
    L_G   = L_adv + lambda_l1 * L_l1 + lambda_physics * L_phys
    L_adv = 0.5 * mean[(D(fake) - 1)^2]                  (LSGAN generator)
    L_D   = 0.5 * mean[(D(real) - real_label)^2 + D(fake)^2]
    L_l1  = mean|G(lr) - hr|                              (normalized tensors)
    L_phys= mean|sum(E_pred)/sum(E_true) - 1|            (denormalized tensors)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data.normalization import ChannelStats, denormalize, normalize


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def discriminator_loss(real_logits: Tensor, fake_logits: Tensor, real_label: float) -> Tensor:
    """LSGAN discriminator loss with one-sided label smoothing on real."""
    return 0.5 * ((real_logits - real_label) ** 2 + fake_logits ** 2).mean()


def generator_adv_loss(fake_logits: Tensor) -> Tensor:
    """LSGAN generator loss — push fake logits toward 1."""
    return 0.5 * ((fake_logits - 1.0) ** 2).mean()


def energy_response(pred_raw: Tensor, target_raw: Tensor) -> Tensor:
    """Per-sample sum(E_pred)/sum(E_true) on denormalized (raw energy) tensors."""
    pred_energy = pred_raw.sum(dim=(1, 2, 3))
    target_energy = target_raw.sum(dim=(1, 2, 3))
    return pred_energy / target_energy.clamp_min(1e-6)


def physics_loss(pred_raw: Tensor, target_raw: Tensor) -> Tensor:
    """Direct relative-response constraint |response - 1| on raw energies."""
    return (energy_response(pred_raw, target_raw) - 1.0).abs().mean()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def psnr_normalized(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    """PSNR on normalized tensors with a fixed max_val=1.0 reference.

    Fixed reference (not target.max()) keeps PSNR comparable across scales and
    runs — see the SwinIR open issue where raw target.max() gave non-comparable
    ~52 dB values.
    """
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    loader,
    stats: ChannelStats,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute val L1 (normalized), PSNR (normalized), and energy response."""
    generator.eval()
    l1_sum = 0.0
    pix_count = 0
    psnr_sum = 0.0
    resp_sum = 0.0
    n_batches = 0

    for batch in loader:
        lr = normalize(batch["lr"].to(device), stats)
        hr = normalize(batch["hr"].to(device), stats)
        fake = generator(lr, hr.shape[-2:])

        l1_sum += F.l1_loss(fake, hr, reduction="sum").item()
        pix_count += hr.numel()
        psnr_sum += psnr_normalized(fake, hr)

        fake_raw = denormalize(fake.float(), stats)
        hr_raw = denormalize(hr.float(), stats)
        resp_sum += energy_response(fake_raw, hr_raw).mean().item()

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    n = max(n_batches, 1)
    return {
        "l1": l1_sum / max(pix_count, 1),
        "psnr_norm": psnr_sum / n,
        "energy_response": resp_sum / n,
    }


# --------------------------------------------------------------------------- #
# Tagging-efficiency data prep
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_tagging_tensors(
    generator: nn.Module,
    loader,
    stats: ChannelStats,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Tensor]:
    """Materialize normalized HR / LR-upsampled / SR images + labels from a loader.

    All three image tensors share HR spatial size (LR is bicubic-upsampled), so a
    single tagger architecture consumes any of them. Returned tensors live on CPU
    to keep memory bounded; the caller moves batches to device during training.

    Returns dict with keys: "hr", "lr", "sr" (each (N,3,H,W)) and "y" ((N,)).
    Also returns "pt" and "m0" ((N,)) when the batch carries them (parquet),
    for downstream physics-correlation plots; absent otherwise.
    Raises ValueError if the dataset has no usable (>1 class) labels.
    """
    generator.eval()
    hr_list: list[Tensor] = []
    lr_list: list[Tensor] = []
    sr_list: list[Tensor] = []
    y_list: list[Tensor] = []
    pt_list: list[Tensor] = []
    m0_list: list[Tensor] = []
    seen = 0

    for batch in loader:
        lr = normalize(batch["lr"].to(device), stats)
        hr = normalize(batch["hr"].to(device), stats)
        target_hw = hr.shape[-2:]
        sr = generator(lr, target_hw)
        lr_up = F.interpolate(lr, size=target_hw, mode="bicubic", align_corners=False)

        hr_list.append(hr.cpu())
        lr_list.append(lr_up.cpu())
        sr_list.append(sr.cpu())
        y_list.append(batch["y"].view(-1).cpu())
        if "pt" in batch:
            pt_list.append(batch["pt"].view(-1).cpu())
        if "m0" in batch:
            m0_list.append(batch["m0"].view(-1).cpu())

        seen += hr.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

    if not y_list:
        raise ValueError("No samples collected for tagging efficiency.")
    y = torch.cat(y_list)
    if y.unique().numel() < 2:
        raise ValueError(
            "Tagging efficiency needs at least two classes in the `y` label, but the "
            "dataset provided only one (HDF5/CaloChallenge has no class label). "
            "Tagging efficiency is parquet-only."
        )
    out = {
        "hr": torch.cat(hr_list),
        "lr": torch.cat(lr_list),
        "sr": torch.cat(sr_list),
        "y": y,
    }
    if pt_list:
        out["pt"] = torch.cat(pt_list)
    if m0_list:
        out["m0"] = torch.cat(m0_list)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
@torch.no_grad()
def render_sample_grid(
    generator: nn.Module,
    batch: dict[str, Tensor],
    stats: ChannelStats,
    device: torch.device,
    path: Path,
    n_samples: int = 4,
) -> np.ndarray:
    """Save an [LR (bicubic up) | SR | HR] comparison grid as a PNG.

    Returns the rendered RGB array (H,W,3) so callers can also log it to W&B
    without re-reading the file. Channels are summed to a single intensity map
    (calorimeter energy) and log-scaled for visibility on sparse deposits.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generator.eval()
    lr = normalize(batch["lr"].to(device), stats)
    hr = normalize(batch["hr"].to(device), stats)
    fake = generator(lr, hr.shape[-2:])

    lr_raw = denormalize(lr.float(), stats).cpu()
    fake_raw = denormalize(fake.float(), stats).cpu()
    hr_raw = denormalize(hr.float(), stats).cpu()

    n = min(n_samples, lr_raw.shape[0])
    target_hw = hr_raw.shape[-2:]
    # Bicubic-upsample LR purely for side-by-side viewing at HR size.
    lr_up = F.interpolate(lr_raw[:n], size=target_hw, mode="bicubic", align_corners=False).clamp_min(0.0)

    def _to_img(t: Tensor) -> np.ndarray:
        img = t.sum(dim=0)  # sum channels -> energy map
        return np.log1p(img.numpy())

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    col_titles = ["LR (bicubic up)", "Super-Resolved", "Ground Truth (HR)"]
    for row in range(n):
        for col, src in enumerate((lr_up[row], fake_raw[row], hr_raw[row])):
            ax = axes[row, col]
            ax.imshow(_to_img(src), cmap="inferno", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.canvas.draw()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return rgb


def render_metrics_plot(history: list[dict], path: Path) -> None:
    """Plot G/D loss and val L1 over epochs to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    epochs = [h["epoch"] for h in history]
    g = [h.get("train_g_loss", float("nan")) for h in history]
    d = [h.get("train_d_loss", float("nan")) for h in history]
    val_l1 = [h.get("val_l1", float("nan")) for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, g, label="G loss", color="tab:blue")
    ax1.plot(epochs, d, label="D loss", color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_l1, label="val L1", color="tab:green", linestyle="--")
    ax2.set_ylabel("val L1 (normalized)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
