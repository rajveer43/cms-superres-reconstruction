"""Generate all experiment artifacts: figures, physics metrics, correlation metrics, report.

Run from task/SRGAN/:
    # HDF5 (CaloChallenge)
    python save_experiment.py \
        --run-dir experiments/2026-06-11/hdf5_run_01 \
        --dataset-path ../datasets/calochallenge_dataset2 \
        --dataset-type hdf5 --hr-size 45 144 --scale-factor 2.0

    # Parquet (QCDToGGQQ)
    python save_experiment.py \
        --run-dir experiments/2026-06-11/parquet_run_01 \
        --dataset-path ../datasets \
        --dataset-type parquet --hr-size 125 125
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from srgan import (
    ChannelStats,
    Generator,
    batch_to_tensor,
    denormalize,
    discover_parquet_files,
    normalize,
    resolve_env,
    split_files,
)
from srgan.data.hdf5_dataset import build_hdf5_dataset


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", **kw)
    plt.close(fig)


def _load_run(run_dir: Path) -> tuple[list[dict], ChannelStats, Generator, dict]:
    metrics = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            metrics.append(json.loads(line))
    if not metrics:
        raise RuntimeError(f"No metrics in {run_dir}/metrics.jsonl")
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    stats = ChannelStats(
        mean=torch.tensor(ckpt["stats"]["mean"]),
        std=torch.tensor(ckpt["stats"]["std"]),
    )
    gen = Generator(
        base_channels=ckpt["args"].get("gen_channels", 64),
        num_blocks=ckpt["args"].get("gen_blocks", 8),
    )
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    return metrics, stats, gen, ckpt["args"]


def _ssim_manual(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """Per-image SSIM via avg_pool2d. Input: (B,1,H,W) in [0,1]."""
    k, pad = window_size, window_size // 2
    mu1 = F.avg_pool2d(pred, k, stride=1, padding=pad)
    mu2 = F.avg_pool2d(target, k, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    s1 = F.avg_pool2d(pred * pred, k, stride=1, padding=pad) - mu1_sq
    s2 = F.avg_pool2d(target * target, k, stride=1, padding=pad) - mu2_sq
    s12 = F.avg_pool2d(pred * target, k, stride=1, padding=pad) - mu12
    ssim_map = ((2*mu12 + C1) * (2*s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean(dim=(-1, -2, -3))


# ---------------------------------------------------------------------------
# Dataset-specific collectors
# ---------------------------------------------------------------------------

def _collect_hdf5(gen, stats, data_dir, device, hr_size, val_ratio, scale_factor, max_samples):
    full_ds = build_hdf5_dataset(data_dir, scale_factor=scale_factor, hr_size=None)
    n_total = len(full_ds)  # type: ignore[arg-type]
    n_val_start = n_total - max(1, int(round(n_total * val_ratio)))
    val_ds = torch.utils.data.Subset(full_ds, list(range(n_val_start, n_total)))
    n_train, n_val = n_val_start, len(val_ds)

    all_lr, all_hr, all_gan, all_bic, all_pt, all_m0, all_y = [], [], [], [], [], [], []
    gen.to(device)
    with torch.no_grad():
        for i in range(len(val_ds)):  # type: ignore[arg-type]
            item = val_ds[i]
            lr = item["lr"].unsqueeze(0).to(device)
            hr = item["hr"].unsqueeze(0).to(device)
            if hr.shape[-2:] != torch.Size(hr_size):
                hr = F.interpolate(hr.float(), size=hr_size, mode="bicubic",
                                   align_corners=False).clamp_min(0.0)
            lr_n = normalize(lr, stats)
            hr_n = normalize(hr, stats)
            fake_n = gen(lr_n, hr_size)
            lr_raw = denormalize(lr_n.cpu(), stats)
            hr_raw = denormalize(hr_n.cpu(), stats)
            gan_raw = denormalize(fake_n.cpu(), stats)
            bic_raw = F.interpolate(lr_raw, size=hr_size, mode="bicubic",
                                    align_corners=False).clamp_min(0.0)
            all_lr.append(lr_raw.squeeze(0).numpy())
            all_hr.append(hr_raw.squeeze(0).numpy())
            all_gan.append(gan_raw.squeeze(0).numpy())
            all_bic.append(bic_raw.squeeze(0).numpy())
            all_pt.append(item["pt"].item())
            all_m0.append(item["m0"].item())
            all_y.append(item["y"].item())
            if max_samples and len(all_hr) >= max_samples:
                break

    return {
        "lr": np.stack(all_lr), "hr": np.stack(all_hr),
        "gan": np.stack(all_gan), "bicubic": np.stack(all_bic),
        "pt": np.array(all_pt), "m0": np.array(all_m0), "y": np.array(all_y),
        "n_train": n_train, "n_val": n_val,
        "dataset_name": "CaloChallenge Dataset 2",
        "channels": ["Calo layer (ch0)", "Calo layer (ch1)", "Calo layer (ch2)"],
    }


def _collect_parquet(gen, stats, data_dir, device, hr_size, val_ratio, max_samples):
    files = discover_parquet_files(data_dir)
    _, val_files = split_files(files, val_ratio)

    all_lr, all_hr, all_gan, all_bic, all_pt, all_m0, all_y = [], [], [], [], [], [], []
    gen.to(device)
    with torch.no_grad():
        for path in val_files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=64,
                columns=["X_jets_LR", "X_jets", "pt", "m0", "y"],
                use_threads=True,
            ):
                lr_b = batch_to_tensor(batch.column(0))
                hr_b = batch_to_tensor(batch.column(1))
                pt_b = batch.column(2).to_pylist()
                m0_b = batch.column(3).to_pylist()
                y_b  = batch.column(4).to_pylist()

                for i in range(lr_b.shape[0]):
                    lr = lr_b[i:i+1].to(device)
                    hr = hr_b[i:i+1].to(device)
                    if hr.shape[-2:] != torch.Size(hr_size):
                        hr = F.interpolate(hr.float(), size=hr_size, mode="bicubic",
                                           align_corners=False).clamp_min(0.0)
                    lr_n = normalize(lr, stats)
                    hr_n = normalize(hr, stats)
                    fake_n = gen(lr_n, hr_size)
                    lr_raw = denormalize(lr_n.cpu(), stats)
                    hr_raw = denormalize(hr_n.cpu(), stats)
                    gan_raw = denormalize(fake_n.cpu(), stats)
                    bic_raw = F.interpolate(lr_raw, size=hr_size, mode="bicubic",
                                            align_corners=False).clamp_min(0.0)
                    all_lr.append(lr_raw.squeeze(0).numpy())
                    all_hr.append(hr_raw.squeeze(0).numpy())
                    all_gan.append(gan_raw.squeeze(0).numpy())
                    all_bic.append(bic_raw.squeeze(0).numpy())
                    all_pt.append(float(pt_b[i]))
                    all_m0.append(float(m0_b[i]))
                    all_y.append(int(y_b[i]))
                    if max_samples and len(all_hr) >= max_samples:
                        break
                if max_samples and len(all_hr) >= max_samples:
                    break
            if max_samples and len(all_hr) >= max_samples:
                break

    n_train_files, _ = split_files(files, val_ratio)
    return {
        "lr": np.stack(all_lr), "hr": np.stack(all_hr),
        "gan": np.stack(all_gan), "bicubic": np.stack(all_bic),
        "pt": np.array(all_pt), "m0": np.array(all_m0), "y": np.array(all_y),
        "n_train": -1, "n_val": len(all_hr),
        "dataset_name": "QCDToGGQQ CMS Jet Images",
        "channels": ["ECAL", "HCAL", "Tracks"],
    }


# ---------------------------------------------------------------------------
# B2 — Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(metrics: list[dict], fig_dir: Path) -> None:
    epochs = [m["epoch"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, label in zip(axes.flat,
            ["train_g_loss", "train_d_loss", "train_l1", "val_l1"],
            ["Generator loss", "Discriminator loss", "Train L1", "Val L1"]):
        if key in metrics[0]:
            ax.plot(epochs, [m[key] for m in metrics], marker="o")
        ax.set_title(label); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    fig.suptitle("Training Curves"); fig.tight_layout()
    _savefig(fig, fig_dir / "training" / "loss_curves.png")

    if "val_response" in metrics[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [m["val_response"] for m in metrics], marker="o", color="tab:orange")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal")
        ax.set_title("Val Physics Response (Σ_GAN / Σ_HR)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Response"); ax.legend(); ax.grid(True, alpha=0.3)
        _savefig(fig, fig_dir / "training" / "physics_response_curve.png")

    if "val_psnr_norm" in metrics[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [m["val_psnr_norm"] for m in metrics], marker="o", color="tab:green")
        ax.set_title("Val PSNR (max_val=1.0)"); ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR (dB)"); ax.grid(True, alpha=0.3)
        _savefig(fig, fig_dir / "training" / "psnr_curve.png")


# ---------------------------------------------------------------------------
# B3 — Reconstruction
# ---------------------------------------------------------------------------

def plot_reconstruction(data: dict, fig_dir: Path) -> None:
    hr, gan, lr = data["hr"], data["gan"], data["lr"]
    ch_names = data["channels"]
    rdir = fig_dir / "reconstruction"

    for n in range(min(4, len(hr))):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        hr_sum = hr[n].sum(0)
        gan_sum = gan[n].sum(0)
        lr_up = F.interpolate(torch.from_numpy(lr[n]).unsqueeze(0).float(),
                              size=hr_sum.shape, mode="bicubic",
                              align_corners=False).squeeze(0).numpy().sum(0)
        vmax = max(float(np.log1p(np.abs(hr_sum)).max()),
                   float(np.log1p(np.abs(gan_sum)).max()), 1e-6)
        for ax, img, title in zip(axes, [lr_up, gan_sum, hr_sum],
                                  ["LR (upsampled)", "GAN output", "HR target"]):
            im = ax.imshow(np.log1p(np.clip(img, 0, None)), cmap="magma",
                           vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(f"{title}\nΣE={img.sum():.1f}")
            ax.set_xlabel("bin W"); ax.set_ylabel("bin H")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Sample {n} — log1p energy (all channels summed)")
        fig.tight_layout()
        _savefig(fig, rdir / f"sample_grid_{n}.png")

    # Per-channel mean shower (parquet has 3 meaningful channels)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for c, ch in enumerate(ch_names):
        mean_lr_c = lr.mean(0)[c]
        mean_hr_c = hr.mean(0)[c]
        mean_gan_c = gan.mean(0)[c]
        lr_up_c = F.interpolate(torch.from_numpy(mean_lr_c)[None, None].float(),
                                size=mean_hr_c.shape, mode="bicubic",
                                align_corners=False).squeeze().numpy()
        vmax = max(mean_hr_c.max(), mean_gan_c.max(), 1e-6)
        for ax, img, title in zip(axes[c],
                                  [lr_up_c, mean_gan_c, mean_hr_c],
                                  ["Mean LR (upsampled)", "Mean GAN", "Mean HR"]):
            im = ax.imshow(img, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(f"{title}\n{ch}"); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-channel mean shower (linear scale)")
    fig.tight_layout()
    _savefig(fig, rdir / "mean_shower_comparison.png")

    # Residual map (summed channels)
    residual = gan.mean(0).sum(0) - hr.mean(0).sum(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    vabs = float(np.abs(residual).max())
    im = ax.imshow(residual, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    ax.set_title("Mean Residual (GAN − HR)")
    ax.set_xlabel("bin W"); ax.set_ylabel("bin H")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig(fig, rdir / "residual_map.png")


# ---------------------------------------------------------------------------
# B4 — Physics metrics
# ---------------------------------------------------------------------------

def compute_physics_metrics(data: dict, fig_dir: Path, dataset_type: str) -> dict:
    hr, gan, bic = data["hr"], data["gan"], data["bicubic"]
    pt, y = data["pt"], data["y"]
    pdir = fig_dir / "physics"

    e_true = hr.sum(axis=(1, 2, 3))
    e_gan  = gan.sum(axis=(1, 2, 3))
    e_bic  = bic.sum(axis=(1, 2, 3))
    e_inc  = np.where(pt > 0, pt, np.nan)

    resp_gan = e_gan / np.where(e_true > 0, e_true, np.nan)
    resp_bic = e_bic / np.where(e_true > 0, e_true, np.nan)
    sf_gan   = e_gan / e_inc
    sf_bic   = e_bic / e_inc
    sf_true  = e_true / e_inc
    rel_err  = (e_gan - e_true) / np.where(e_true > 0, e_true, np.nan)

    # Response histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    kw = dict(bins=50, range=(0, 2), alpha=0.6, density=True)
    ax.hist(resp_gan[np.isfinite(resp_gan)],
            label=f"GAN  μ={np.nanmean(resp_gan):.3f} σ={np.nanstd(resp_gan):.3f}", **kw)
    ax.hist(resp_bic[np.isfinite(resp_bic)],
            label=f"Bicubic  μ={np.nanmean(resp_bic):.3f} σ={np.nanstd(resp_bic):.3f}", **kw)
    ax.axvline(1.0, color="black", linestyle="--")
    ax.set_xlabel("Response  E_pred / E_true"); ax.set_ylabel("Density")
    ax.set_title("Energy Response Distribution"); ax.legend()
    _savefig(fig, pdir / "energy_response_hist.png")

    # Per-class response (parquet has real class labels 0/1)
    if dataset_type == "parquet":
        classes = sorted(np.unique(y[np.isfinite(y)]).astype(int).tolist())
        if len(classes) > 1:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            for cls, col in zip(classes, colors):
                mask = (y == cls) & np.isfinite(resp_gan)
                if mask.sum() > 0:
                    ax.hist(resp_gan[mask], bins=40, range=(0, 2), alpha=0.55,
                            density=True, color=col,
                            label=f"class {cls}  μ={resp_gan[mask].mean():.3f}")
            ax.axvline(1.0, color="black", linestyle="--")
            ax.set_xlabel("Response  E_GAN / E_true"); ax.set_ylabel("Density")
            ax.set_title("GAN Energy Response by Class (QCD jet label)")
            ax.legend(); _savefig(fig, pdir / "response_by_class.png")

        # pt vs response scatter
        fig, ax = plt.subplots(figsize=(7, 4))
        mask = np.isfinite(resp_gan) & (pt > 0)
        ax.scatter(pt[mask], resp_gan[mask], s=4, alpha=0.3)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_xscale("log"); ax.set_xlabel("pT (GeV)"); ax.set_ylabel("Response")
        ax.set_title("GAN Response vs pT")
        _savefig(fig, pdir / "response_vs_pt.png")

        # m0 vs response scatter
        fig, ax = plt.subplots(figsize=(7, 4))
        mask2 = np.isfinite(resp_gan) & (data["m0"] > 0)
        ax.scatter(data["m0"][mask2], resp_gan[mask2], s=4, alpha=0.3, color="tab:orange")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_xscale("log"); ax.set_xlabel("m0 (GeV)"); ax.set_ylabel("Response")
        ax.set_title("GAN Response vs m0")
        _savefig(fig, pdir / "response_vs_m0.png")

    # Sampling fraction
    if np.any(np.isfinite(sf_true) & (sf_true > 0)):
        fig, ax = plt.subplots(figsize=(7, 4))
        for arr, label in [(sf_true, "True"), (sf_gan, "GAN"), (sf_bic, "Bicubic")]:
            clean = arr[np.isfinite(arr) & (arr > 0)]
            if len(clean):
                ax.hist(clean, bins=50, alpha=0.5, label=label, density=True)
        ax.set_xscale("log")
        ax.set_xlabel("Sampling fraction  E_dep / E_pt")
        ax.set_ylabel("Density"); ax.set_title("Sampling Fraction Distribution"); ax.legend()
        _savefig(fig, pdir / "sampling_fraction_hist.png")

    # Energy scatter log-log
    fig, ax = plt.subplots(figsize=(6, 6))
    mask = e_true > 0
    ax.scatter(e_true[mask], e_gan[mask], s=5, alpha=0.3, label="GAN")
    ax.scatter(e_true[mask], e_bic[mask], s=5, alpha=0.3, label="Bicubic")
    lo, hi = e_true[mask].min(), e_true[mask].max() * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="identity")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("HR total energy"); ax.set_ylabel("Predicted total energy")
    ax.set_title("Energy Scatter (log-log)"); ax.legend(markerscale=3)
    _savefig(fig, pdir / "energy_scatter.png")

    # Relative error scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    mask2 = np.isfinite(rel_err) & (e_true > 0)
    ax.scatter(e_true[mask2], rel_err[mask2], s=5, alpha=0.3, color="tab:blue")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("HR total energy"); ax.set_ylabel("(E_GAN − E_HR) / E_HR")
    ax.set_title("Energy Relative Error vs True Energy")
    _savefig(fig, pdir / "energy_scatter_residual.png")

    # Per-class physics table (parquet)
    by_class: dict = {}
    if dataset_type == "parquet":
        for cls in sorted(np.unique(y[np.isfinite(y)]).astype(int).tolist()):
            mask = y == cls
            by_class[f"class_{cls}"] = {
                "n": int(mask.sum()),
                "response_gan": {"mean": float(np.nanmean(resp_gan[mask])),
                                 "std":  float(np.nanstd(resp_gan[mask]))},
                "response_bicubic": {"mean": float(np.nanmean(resp_bic[mask])),
                                     "std":  float(np.nanstd(resp_bic[mask]))},
            }

    return {
        "n_samples": int(len(e_true)),
        "response_gan":     {"mean": float(np.nanmean(resp_gan)),
                             "std":  float(np.nanstd(resp_gan)),
                             "median": float(np.nanmedian(resp_gan))},
        "response_bicubic": {"mean": float(np.nanmean(resp_bic)),
                             "std":  float(np.nanstd(resp_bic)),
                             "median": float(np.nanmedian(resp_bic))},
        "sampling_fraction_gan":  {"mean": float(np.nanmean(sf_gan)),
                                   "std":  float(np.nanstd(sf_gan))},
        "sampling_fraction_true": {"mean": float(np.nanmean(sf_true)),
                                   "std":  float(np.nanstd(sf_true))},
        "relative_error_gan": {"mean":     float(np.nanmean(rel_err)),
                               "std":      float(np.nanstd(rel_err)),
                               "abs_mean": float(np.nanmean(np.abs(rel_err)))},
        "by_class": by_class,
    }


# ---------------------------------------------------------------------------
# B5 — Spatial correlation metrics
# ---------------------------------------------------------------------------

def compute_correlation_metrics(data: dict, fig_dir: Path, hr_size: tuple[int, int]) -> dict:
    hr, gan, bic, lr = data["hr"], data["gan"], data["bicubic"], data["lr"]
    H, W = hr_size
    cdir = fig_dir / "correlations"

    # Radial profile (mean over W, samples, channels)
    hr_rad  = hr.sum(1).mean(axis=(0, 2))
    gan_rad = gan.sum(1).mean(axis=(0, 2))
    bic_rad = bic.sum(1).mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(H), hr_rad,  label="HR",      linewidth=1.5)
    ax.plot(np.arange(H), gan_rad, label="GAN",     linewidth=1.5, linestyle="--")
    ax.plot(np.arange(H), bic_rad, label="Bicubic", linewidth=1.5, linestyle=":")
    ax.set_xlabel("Row bin"); ax.set_ylabel("Mean deposited energy (a.u.)")
    ax.set_title("Row (Radial) Profile"); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, cdir / "radial_profile.png")

    # Azimuthal profile (mean over H, samples, channels)
    hr_az  = hr.sum(1).mean(axis=(0, 1))
    gan_az = gan.sum(1).mean(axis=(0, 1))
    bic_az = bic.sum(1).mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(W), hr_az,  label="HR",      linewidth=1.5)
    ax.plot(np.arange(W), gan_az, label="GAN",     linewidth=1.5, linestyle="--")
    ax.plot(np.arange(W), bic_az, label="Bicubic", linewidth=1.5, linestyle=":")
    ax.set_xlabel("Column bin (φ)"); ax.set_ylabel("Mean deposited energy (a.u.)")
    ax.set_title("Column (Azimuthal) Profile"); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, cdir / "azimuthal_profile.png")

    # Pixel correlation scatter (log1p, 5000 random pixels, up to 16 images)
    n_imgs = min(16, len(hr))
    hr_flat  = np.log1p(np.clip(hr[:n_imgs].reshape(-1), 0, None))
    gan_flat = np.log1p(np.clip(gan[:n_imgs].reshape(-1), 0, None))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(hr_flat), size=min(5000, len(hr_flat)), replace=False)
    pearson_r = float(np.corrcoef(hr_flat[idx], gan_flat[idx])[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(hr_flat[idx], gan_flat[idx], s=3, alpha=0.3)
    lim = max(hr_flat[idx].max(), gan_flat[idx].max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlabel("HR pixel (log1p)"); ax.set_ylabel("GAN pixel (log1p)")
    ax.set_title(f"Pixel Correlation  r = {pearson_r:.4f}")
    _savefig(fig, cdir / "pixel_correlation.png")

    # SSIM histogram (summed channels, normalised to [0,1] per image)
    hr_t  = torch.from_numpy(hr[:n_imgs].sum(1, keepdims=True)).float()
    gan_t = torch.from_numpy(gan[:n_imgs].sum(1, keepdims=True)).float()
    hr_max = hr_t.flatten(1).max(1).values.view(-1,1,1,1).clamp_min(1e-8)
    ssim_vals = _ssim_manual((gan_t/hr_max).clamp(0,1), (hr_t/hr_max).clamp(0,1)).numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ssim_vals, bins=16, color="tab:blue", alpha=0.8)
    ax.axvline(ssim_vals.mean(), color="red", linestyle="--",
               label=f"mean={ssim_vals.mean():.4f}")
    ax.set_xlabel("SSIM"); ax.set_ylabel("Count")
    ax.set_title("Per-sample SSIM (GAN vs HR)"); ax.legend()
    _savefig(fig, cdir / "ssim_hist.png")

    # Sparsity comparison
    threshold = 1e-4
    lr_up = np.stack([
        F.interpolate(torch.from_numpy(lr[i]).unsqueeze(0).float(),
                      size=hr_size, mode="bicubic", align_corners=False
                      ).squeeze(0).numpy() for i in range(len(lr))
    ])
    sp_lr  = float((lr_up  < threshold).mean())
    sp_gan = float((gan    < threshold).mean())
    sp_hr  = float((hr     < threshold).mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["LR (upsampled)", "GAN output", "HR target"],
                  [sp_lr, sp_gan, sp_hr],
                  color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylabel("Fraction of near-zero cells (< 1e-4)")
    ax.set_title("Sparsity Comparison"); ax.set_ylim(0, 1)
    for bar, v in zip(bars, [sp_lr, sp_gan, sp_hr]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=10)
    _savefig(fig, cdir / "sparsity_comparison.png")

    # Channel-wise energy correlation heatmap (parquet: 3 channels)
    n_ch = hr.shape[1]
    if n_ch >= 2:
        corr_hr  = np.corrcoef(hr[:, :, :, :].reshape(len(hr), n_ch, -1).mean(-1).T)
        corr_gan = np.corrcoef(gan[:, :, :, :].reshape(len(gan), n_ch, -1).mean(-1).T)
        ch_names = data["channels"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, mat, title in zip(axes, [corr_hr, corr_gan], ["HR channels", "GAN channels"]):
            im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(n_ch)); ax.set_yticks(range(n_ch))
            ax.set_xticklabels(ch_names, rotation=30, ha="right")
            ax.set_yticklabels(ch_names)
            ax.set_title(f"Channel correlation — {title}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for i in range(n_ch):
                for j in range(n_ch):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        _savefig(fig, cdir / "channel_correlation_heatmap.png")

    rad_mse = float(np.mean((gan_rad - hr_rad)**2))
    az_mse  = float(np.mean((gan_az  - hr_az)**2))

    return {
        "pearson_r_pixel": pearson_r,
        "ssim_mean": float(ssim_vals.mean()),
        "ssim_std":  float(ssim_vals.std()),
        "radial_profile_mse":    rad_mse,
        "azimuthal_profile_mse": az_mse,
        "sparsity": {"lr_mean": sp_lr, "gan_mean": sp_gan, "hr_mean": sp_hr},
    }


# ---------------------------------------------------------------------------
# B6 — Experiment report
# ---------------------------------------------------------------------------

def write_report(run_dir, metrics, train_args, physics, corr,
                 env_str, dataset_type, hr_size, data) -> None:
    best = min(metrics, key=lambda m: m.get("val_l1", float("inf")))
    n_train, n_val = data["n_train"], data["n_val"]
    ds_name = data["dataset_name"]

    lines = [
        "# Experiment Report", "",
        f"**Dataset:** {ds_name}  ",
        f"**Run directory:** `{run_dir}`  ",
        f"**Platform:** {env_str}", "",
        "## Run command", "",
        f"```bash",
        f"python train_srgan.py \\",
        f"    --dataset-type {dataset_type} \\",
        f"    --hr-size {hr_size[0]} {hr_size[1]} \\",
        f"    --epochs {train_args.get('epochs','?')} \\",
        f"    --batch-size {train_args.get('batch_size','?')} \\",
        f"    --output-dir {run_dir}",
        f"```", "",
        "## Dataset", "",
        "| Property | Value |", "| --- | --- |",
        f"| Name | {ds_name} |",
        f"| N train | {n_train if n_train > 0 else 'N/A (streaming)'} |",
        f"| N val | {n_val} |",
        f"| HR shape | {hr_size} |",
        f"| Scale factor | {train_args.get('scale_factor', 2.0)} |", "",
        "## Training config", "",
        "| Param | Value |", "| --- | --- |",
    ]
    for k in ["epochs","batch_size","lr","lambda_l1","lambda_physics",
              "gen_channels","gen_blocks","d_lr_ratio","n_critic"]:
        if k in train_args:
            lines.append(f"| {k} | {train_args[k]} |")

    lines += [
        "", "## Best epoch", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Best epoch | {best['epoch']} |",
        f"| val_l1 | {best.get('val_l1', 'N/A'):.5f} |",
        f"| val_psnr_norm | {best.get('val_psnr_norm', 'N/A'):.2f} dB |",
        f"| val_response | {best.get('val_response', 'N/A'):.4f} |", "",
        "## Physics metrics", "",
        "| Metric | GAN | Bicubic |", "| --- | --- | --- |",
        f"| Response mean   | {physics['response_gan']['mean']:.4f} | {physics['response_bicubic']['mean']:.4f} |",
        f"| Response std    | {physics['response_gan']['std']:.4f}  | {physics['response_bicubic']['std']:.4f}  |",
        f"| Response median | {physics['response_gan']['median']:.4f} | {physics['response_bicubic']['median']:.4f} |",
        f"| |rel error| mean | {physics['relative_error_gan']['abs_mean']:.4f} | — |",
    ]

    if physics.get("by_class"):
        lines += ["", "### Per-class (QCD label)", "",
                  "| Class | N | GAN response μ | GAN response σ | Bicubic response μ |",
                  "| --- | --- | --- | --- | --- |"]
        for cls, v in physics["by_class"].items():
            lines.append(
                f"| {cls} | {v['n']} "
                f"| {v['response_gan']['mean']:.4f} "
                f"| {v['response_gan']['std']:.4f} "
                f"| {v['response_bicubic']['mean']:.4f} |"
            )

    lines += [
        "", "## Correlation metrics", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Pearson r (pixel) | {corr['pearson_r_pixel']:.4f} |",
        f"| SSIM mean | {corr['ssim_mean']:.4f} ± {corr['ssim_std']:.4f} |",
        f"| Radial profile MSE | {corr['radial_profile_mse']:.4f} |",
        f"| Azimuthal profile MSE | {corr['azimuthal_profile_mse']:.4f} |",
        f"| GAN sparsity | {corr['sparsity']['gan_mean']:.3f} |",
        f"| HR sparsity  | {corr['sparsity']['hr_mean']:.3f} |", "",
        "## Figures", "",
        "### Training",
        "![Loss curves](figures/training/loss_curves.png)",
        "![Response](figures/training/physics_response_curve.png)",
        "![PSNR](figures/training/psnr_curve.png)", "",
        "### Reconstruction",
        "![Sample 0](figures/reconstruction/sample_grid_0.png)",
        "![Mean shower](figures/reconstruction/mean_shower_comparison.png)",
        "![Residual](figures/reconstruction/residual_map.png)", "",
        "### Physics",
        "![Response hist](figures/physics/energy_response_hist.png)",
        "![Energy scatter](figures/physics/energy_scatter.png)",
        "![Residual scatter](figures/physics/energy_scatter_residual.png)", "",
        "### Correlations",
        "![Radial](figures/correlations/radial_profile.png)",
        "![Azimuthal](figures/correlations/azimuthal_profile.png)",
        "![Pixel corr](figures/correlations/pixel_correlation.png)",
        "![SSIM](figures/correlations/ssim_hist.png)",
        "![Sparsity](figures/correlations/sparsity_comparison.png)",
        "![Channel corr](figures/correlations/channel_correlation_heatmap.png)", "",
        "## Known limitations", "",
        "1. **Synthetic LR has no real noise model** — LR is bicubic downsampled; real detector noise not simulated.",
        "2. **Single calorimeter layer replicated to 3 channels** (HDF5 only) — approximation of ECAL/HCAL/Tracks.",
        "3. **Energy response not calibrated to 1.0** — physics loss drives it toward 1.0 but a post-hoc "
        "calibration step is needed for publication-level accuracy.",
    ]
    (run_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",      type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-type", choices=["hdf5","parquet"], default="hdf5")
    parser.add_argument("--hr-size",      type=int, nargs=2, default=[45,144], metavar=("H","W"))
    parser.add_argument("--scale-factor", type=float, default=2.0)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    run_dir  = Path(args.run_dir)
    fig_dir  = run_dir / "figures"
    hr_size: tuple[int,int] = tuple(args.hr_size)  # type: ignore[assignment]
    env      = resolve_env()
    print(f"[env] {env}")

    metrics, stats, gen, train_args = _load_run(run_dir)
    best_ep = min(metrics, key=lambda m: m.get("val_l1", float("inf")))["epoch"]
    print(f"[info] best epoch by val_l1: {best_ep}")

    data_dir   = Path(args.dataset_path)
    val_ratio  = train_args.get("val_ratio", 0.33)

    print(f"[inference] collecting val samples ({args.max_val_samples or 'all'})…")
    if args.dataset_type == "hdf5":
        data = _collect_hdf5(gen, stats, data_dir, env.device, hr_size,
                             val_ratio, args.scale_factor, args.max_val_samples)
    else:
        data = _collect_parquet(gen, stats, data_dir, env.device, hr_size,
                                val_ratio, args.max_val_samples)
    print(f"[inference] done — {len(data['hr'])} samples")

    print("[figures] training curves…")
    plot_training_curves(metrics, fig_dir)

    print("[figures] reconstruction…")
    plot_reconstruction(data, fig_dir)

    print("[figures] physics…")
    physics = compute_physics_metrics(data, fig_dir, args.dataset_type)
    (run_dir / "physics_metrics.json").write_text(json.dumps(physics, indent=2), encoding="utf-8")

    print("[figures] correlations…")
    corr = compute_correlation_metrics(data, fig_dir, hr_size)
    (run_dir / "correlation_metrics.json").write_text(json.dumps(corr, indent=2), encoding="utf-8")

    print("[report] writing EXPERIMENT_REPORT.md…")
    write_report(run_dir, metrics, train_args, physics, corr,
                 str(env), args.dataset_type, hr_size, data)

    n_figs = len(list(fig_dir.rglob("*.png")))
    print(f"[done] {n_figs} figures saved to {fig_dir}")

    best = min(metrics, key=lambda m: m.get("val_l1", float("inf")))
    print("\n=== Summary ===")
    print(f"  best epoch      : {best['epoch']}")
    print(f"  val_l1          : {best.get('val_l1','N/A'):.5f}")
    print(f"  val_psnr_norm   : {best.get('val_psnr_norm','N/A'):.2f} dB")
    print(f"  val_response    : {best.get('val_response','N/A'):.4f}")
    print(f"  response_gan    : {physics['response_gan']['mean']:.4f} ± {physics['response_gan']['std']:.4f}")
    print(f"  pearson_r_pixel : {corr['pearson_r_pixel']:.4f}")
    print(f"  ssim_mean       : {corr['ssim_mean']:.4f}")
    if physics.get("by_class"):
        for cls, v in physics["by_class"].items():
            print(f"  {cls} response   : {v['response_gan']['mean']:.4f} ± {v['response_gan']['std']:.4f}")


if __name__ == "__main__":
    main()
