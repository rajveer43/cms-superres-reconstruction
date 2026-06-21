"""Plot ROC curve and AUC for the SRGAN discriminator.

The discriminator is a binary classifier: real HR (label=1) vs GAN-generated
fake HR (label=0). ROC/AUC measures how well it separates the two distributions
after training — a perfect GAN should have AUC close to 0.5 (D can't tell them
apart). A high AUC means the GAN still has room to improve.

Usage
-----
# HDF5 run
python plot_roc_auc.py \
    --run-dir experiments/2026-06-12/hdf5_run_01 \
    --dataset-type hdf5 \
    --dataset-path ../datasets/calochallenge_dataset2 \
    --hr-size 45 144

# Parquet run
python plot_roc_auc.py \
    --run-dir experiments/2026-06-11/parquet_run_02 \
    --dataset-type parquet \
    --dataset-path ../datasets \
    --hr-size 125 125

# Compare multiple runs on the same plot
python plot_roc_auc.py \
    --run-dir experiments/2026-06-11/hdf5_run_01 experiments/2026-06-12/hdf5_run_01 \
    --dataset-type hdf5 \
    --dataset-path ../datasets/calochallenge_dataset2 \
    --hr-size 45 144
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Make sure srgan package is importable when run from SRGAN/
sys.path.insert(0, str(Path(__file__).parent))

from srgan.data.factory import get_dataloader
from srgan.data.normalization import ChannelStats
from srgan.models.discriminator import Discriminator
from srgan.models.generator import Generator
from srgan.utils.env import resolve_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(run_dir: Path, device: torch.device) -> dict:
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        # Fall back to latest epoch checkpoint
        ckpts = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {run_dir / 'checkpoints'}")
        ckpt_path = ckpts[-1]
        print(f"[warn] best.pt not found, using {ckpt_path.name}")
    print(f"[ckpt] loading {ckpt_path}")
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _collect_scores(
    run_dir: Path,
    dataset_type: str,
    dataset_path: Path,
    hr_size: tuple[int, int],
    scale_factor: float,
    max_samples: int,
    device: torch.device,
    env,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) arrays for ROC computation.

    scores: raw discriminator logits averaged over spatial patches, sigmoid'd
    labels: 1 = real HR, 0 = GAN fake HR
    """
    ckpt = _load_checkpoint(run_dir, device)
    saved_args = ckpt.get("args", {})

    # Build models
    gen_channels = saved_args.get("gen_channels", 64)
    gen_blocks   = saved_args.get("gen_blocks", 8)
    generator     = Generator(in_channels=3, base_channels=gen_channels, num_blocks=gen_blocks).to(device)
    discriminator = Discriminator(in_channels=3).to(device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    generator.eval()
    discriminator.eval()

    stats = ChannelStats.from_dict(ckpt["stats"])

    # Val dataloader
    val_loader, _ = get_dataloader(
        dataset_type=dataset_type,
        path=dataset_path,
        split="val",
        env=env,
        batch_size=32,
        val_ratio=saved_args.get("val_ratio", 0.15),
        scale_factor=scale_factor,
        hr_size=hr_size,
        stats_cache_path=run_dir / "normalization.json",
    )

    all_scores: list[float] = []
    all_labels: list[int]   = []
    n_collected = 0

    mean = torch.tensor(stats.mean, device=device).view(1, -1, 1, 1)
    std  = torch.tensor(stats.std,  device=device).view(1, -1, 1, 1)

    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            # Resize HR to target size if needed
            if hr.shape[-2:] != torch.Size(hr_size):
                hr = F.interpolate(hr, size=hr_size, mode="bicubic", align_corners=False)

            # Normalize
            lr_n = (lr - mean) / (std + 1e-8)
            hr_n = (hr - mean) / (std + 1e-8)

            # Generator forward → fake HR
            fake_hr = generator(lr_n, target_size=hr_size)
            if fake_hr.shape[-2:] != torch.Size(hr_size):
                fake_hr = F.interpolate(fake_hr, size=hr_size, mode="bicubic", align_corners=False)

            # Discriminator scores: average over spatial patch map, then sigmoid
            real_logits = discriminator(lr_n, hr_n).mean(dim=[1, 2, 3])   # (B,)
            fake_logits = discriminator(lr_n, fake_hr).mean(dim=[1, 2, 3]) # (B,)

            real_scores = torch.sigmoid(real_logits).cpu().numpy()
            fake_scores = torch.sigmoid(fake_logits).cpu().numpy()

            all_scores.extend(real_scores.tolist())
            all_labels.extend([1] * len(real_scores))
            all_scores.extend(fake_scores.tolist())
            all_labels.extend([0] * len(fake_scores))

            n_collected += len(real_scores) + len(fake_scores)
            if n_collected >= max_samples * 2:
                break

    return np.array(all_scores, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute FPR, TPR, AUC without sklearn."""
    thresholds = np.linspace(1.0, 0.0, 500)
    fpr_list, tpr_list = [], []
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tpr_list.append(tp / (pos + 1e-9))
        fpr_list.append(fp / (neg + 1e-9))
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    auc = float(np.trapezoid(tpr, fpr))  # negative because fpr is decreasing
    if auc < 0:
        auc = -auc
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC/AUC for SRGAN discriminator")
    parser.add_argument("--run-dir",      nargs="+", type=Path, required=True,
                        help="One or more experiment run directories (each must have checkpoints/best.pt)")
    parser.add_argument("--dataset-type", choices=["parquet", "hdf5"], required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--hr-size",      type=int, nargs=2, default=[45, 144], metavar=("H", "W"))
    parser.add_argument("--scale-factor", type=float, default=2.0)
    parser.add_argument("--max-samples",  type=int, default=2000,
                        help="Max real samples to score (same number of fakes). Default 2000.")
    parser.add_argument("--out-dir",      type=Path, default=None,
                        help="Where to save figures. Defaults to first --run-dir/figures/")
    args = parser.parse_args()

    env    = resolve_env()
    device = env.device
    hr_size = tuple(args.hr_size)
    print(f"[env] {env}")

    out_dir = args.out_dir or (args.run_dir[0] / "figures" / "roc")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── collect scores for each run ────────────────────────────────────────
    results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []

    for run_dir in args.run_dir:
        label = run_dir.name
        print(f"\n[run] {label}")
        scores, labels = _collect_scores(
            run_dir=run_dir,
            dataset_type=args.dataset_type,
            dataset_path=args.dataset_path,
            hr_size=hr_size,
            scale_factor=args.scale_factor,
            max_samples=args.max_samples,
            device=device,
            env=env,
        )
        n_real = (labels == 1).sum()
        n_fake = (labels == 0).sum()
        print(f"  real={n_real} fake={n_fake} "
              f"real_score_mean={scores[labels==1].mean():.3f} "
              f"fake_score_mean={scores[labels==0].mean():.3f}")

        fpr, tpr, auc = _roc_curve(scores, labels)
        results.append((label, fpr, tpr, scores, labels, auc))
        print(f"  AUC = {auc:.4f}  (0.5 = perfect GAN, 1.0 = D easily separates real/fake)")

    # ── Figure 1: ROC curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors

    for i, (label, fpr, tpr, scores, labels, auc) in enumerate(results):
        ax.plot(fpr, tpr, color=colors[i % 10], lw=2,
                label=f"{label}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate  (fake classified as real)")
    ax.set_ylabel("True Positive Rate  (real classified as real)")
    ax.set_title("Discriminator ROC Curve\n"
                 "(AUC → 0.5 means GAN fools discriminator perfectly)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = out_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"\n[saved] {roc_path}")

    # ── Figure 2: Score distributions (real vs fake) ───────────────────────
    n_runs = len(results)
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5), squeeze=False)

    for i, (label, fpr, tpr, scores, labels, auc) in enumerate(results):
        ax = axes[0, i]
        real_s = scores[labels == 1]
        fake_s = scores[labels == 0]
        bins = np.linspace(0, 1, 40)
        ax.hist(real_s, bins=bins, alpha=0.6, color="steelblue", label=f"Real HR  (n={len(real_s)})", density=True)
        ax.hist(fake_s, bins=bins, alpha=0.6, color="tomato",    label=f"GAN fake (n={len(fake_s)})", density=True)
        ax.axvline(0.5, color="k", lw=1, ls="--", label="D threshold = 0.5")
        ax.set_xlabel("Discriminator score  σ(logit)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}\nAUC = {auc:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Discriminator Score Distributions: Real vs GAN", fontsize=12)
    fig.tight_layout()
    dist_path = out_dir / "score_distributions.png"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {dist_path}")

    # ── Figure 3: AUC bar chart (multi-run comparison) ─────────────────────
    if n_runs > 1:
        fig, ax = plt.subplots(figsize=(max(5, 2 * n_runs), 4))
        run_names = [r[0] for r in results]
        aucs      = [r[5] for r in results]
        bars = ax.bar(run_names, aucs, color=colors[:n_runs], alpha=0.8, edgecolor="k", linewidth=0.8)
        ax.axhline(0.5, color="k", ls="--", lw=1, label="Ideal GAN (AUC=0.5)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC")
        ax.set_title("Discriminator AUC Comparison")
        ax.legend()
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{auc:.4f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        bar_path = out_dir / "auc_comparison.png"
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        print(f"[saved] {bar_path}")

    print(f"\nDone. All figures saved to {out_dir}/")
    print("\nInterpretation guide:")
    print("  AUC = 0.50 → discriminator can't tell real from fake  (ideal GAN)")
    print("  AUC = 0.70 → D has some advantage, GAN needs more training")
    print("  AUC > 0.85 → D dominates, generator is failing (mode collapse risk)")


if __name__ == "__main__":
    main()
