"""SRGAN training entry point — production-grade, platform-agnostic.

Run from the SRGAN/ directory:
    python train_srgan.py [OPTIONS]

All CLI flags from the original script are preserved with identical names
and defaults.  New flags are additive only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Add SRGAN/ to sys.path so `srgan` package is importable when invoked directly.
sys.path.insert(0, str(Path(__file__).parent))

from srgan import (
    ChannelStats,
    Discriminator,
    Generator,
    denormalize,
    get_dataloader,
    normalize,
    resolve_env,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _psnr_norm(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    """PSNR on z-score normalised tensors (max_val=1.0 for cross-run comparability)."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val**2 / mse)


@torch.no_grad()
def evaluate(
    generator: torch.nn.Module,
    loader,
    stats: ChannelStats,
    env,
    hr_size: tuple[int, int],
    max_batches: int | None = None,
) -> dict[str, float]:
    generator.eval()
    l1_total = 0.0
    psnr_total = 0.0
    response_total = 0.0
    pixel_total = 0
    num_batches = 0

    for batch in loader:
        nb = env.pin_memory
        lr = batch["lr"].to(env.device, non_blocking=nb)
        hr = batch["hr"].to(env.device, non_blocking=nb)

        if hr.shape[-2:] != torch.Size(hr_size):
            hr = F.interpolate(hr.float(), size=hr_size, mode="bicubic", align_corners=False).clamp_min(0.0)

        lr_n = normalize(lr, stats)
        hr_n = normalize(hr, stats)

        with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=env.use_amp):
            fake_n = generator(lr_n, hr_size)

        fake_n = fake_n.float()
        hr_n = hr_n.float()

        l1_total += F.l1_loss(fake_n, hr_n, reduction="sum").item()
        psnr_total += _psnr_norm(fake_n, hr_n)
        fake_raw = denormalize(fake_n, stats)
        hr_raw = denormalize(hr_n, stats)
        response = (fake_raw.sum(dim=(1, 2, 3)) / hr_raw.sum(dim=(1, 2, 3)).clamp_min(1e-6)).mean().item()
        response_total += response
        pixel_total += hr_n.numel()
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break

    n = max(num_batches, 1)
    return {
        "l1": l1_total / max(pixel_total, 1),
        "psnr_norm": psnr_total / n,
        "response": response_total / n,
    }


# ---------------------------------------------------------------------------
# Sample saving
# ---------------------------------------------------------------------------

def save_sample_batch(
    generator: torch.nn.Module,
    batch: dict[str, Tensor],
    stats: ChannelStats,
    env,
    hr_size: tuple[int, int],
    path: Path,
) -> None:
    generator.eval()
    with torch.no_grad():
        nb = env.pin_memory
        lr = batch["lr"].to(env.device, non_blocking=nb)
        hr = batch["hr"].to(env.device, non_blocking=nb)
        if hr.shape[-2:] != torch.Size(hr_size):
            hr = F.interpolate(hr.float(), size=hr_size, mode="bicubic", align_corners=False).clamp_min(0.0)
        lr_n = normalize(lr, stats)
        hr_n = normalize(hr, stats)

        with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=env.use_amp):
            fake_n = generator(lr_n, hr_size)

        lr_img = denormalize(lr_n.float().cpu(), stats)
        fake_img = denormalize(fake_n.float().cpu(), stats)
        hr_img = denormalize(hr_n.float().cpu(), stats)
        np.savez_compressed(path, lr=lr_img.numpy(), fake=fake_img.numpy(), hr=hr_img.numpy())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    env = resolve_env()
    print(f"[env] {env}")

    data_dir = Path(args.dataset_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    hr_size: tuple[int, int] = tuple(args.hr_size)  # type: ignore[assignment]
    stats_cache = out_dir / "normalization.json"

    # Build train loader (also computes / loads stats).
    train_loader, stats = get_dataloader(
        dataset_type=args.dataset_type,
        path=data_dir,
        split="train",
        env=env,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        scale_factor=args.scale_factor,
        hr_size=hr_size if args.dataset_type == "hdf5" else None,
        reservoir_size=args.batch_buffer_size * args.batch_size,
        stats_batch_size=args.stats_batch_size,
        max_stats_batches=args.max_stats_batches,
        stats_cache_path=stats_cache,
        skip_stats_cache=args.skip_stats_cache,
    )
    val_loader, _ = get_dataloader(
        dataset_type=args.dataset_type,
        path=data_dir,
        split="val",
        env=env,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        scale_factor=args.scale_factor,
        hr_size=hr_size if args.dataset_type == "hdf5" else None,
        stats_cache_path=stats_cache,
        skip_stats_cache=False,
    )

    stats.save(stats_cache)

    generator = Generator(base_channels=args.gen_channels, num_blocks=args.gen_blocks).to(env.device)
    discriminator = Discriminator().to(env.device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr * args.d_lr_ratio, betas=(0.5, 0.999))

    # GradScaler is a no-op on non-CUDA devices (enabled=False).
    _scaler_device = env.device.type if env.use_amp else "cpu"
    scaler_g = torch.amp.GradScaler(_scaler_device, enabled=env.use_amp)
    scaler_d = torch.amp.GradScaler(_scaler_device, enabled=env.use_amp)

    best_val = math.inf
    log_path = out_dir / "metrics.jsonl"

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            generator.train()
            discriminator.train()

            g_running = d_running = l1_running = phys_running = 0.0
            last_d_loss = 0.0
            seen = 0
            import time as _time
            epoch_t0 = _time.time()
            print(f"\n{'='*60}")
            print(f"[epoch {epoch:03d}/{args.epochs}] starting training loop")
            print(f"{'='*60}")

            for step, batch in enumerate(train_loader, start=1):
                nb = env.pin_memory
                lr = batch["lr"].to(env.device, non_blocking=nb)
                hr = batch["hr"].to(env.device, non_blocking=nb)

                # Resize HR to target resolution if it differs (e.g. parquet 125x125 with --hr-size 128 128).
                if hr.shape[-2:] != torch.Size(hr_size):
                    hr = F.interpolate(hr.float(), size=hr_size, mode="bicubic", align_corners=False).clamp_min(0.0)

                lr_n = normalize(lr, stats)
                hr_n = normalize(hr, stats)

                with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=env.use_amp):
                    fake = generator(lr_n, hr_size)

                fake_raw = denormalize(fake.float(), stats)
                hr_raw = denormalize(hr_n.float(), stats)
                pred_energy = fake_raw.sum(dim=(1, 2, 3))
                target_energy = hr_raw.sum(dim=(1, 2, 3))

                # Discriminator update (TTUR: every n_critic steps).
                if step % args.n_critic == 0:
                    with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=env.use_amp):
                        real_logits = discriminator(lr_n, hr_n)
                        fake_logits_d = discriminator(lr_n, fake.detach())
                        d_loss = 0.5 * ((real_logits - args.real_label) ** 2 + fake_logits_d ** 2).mean()

                    opt_d.zero_grad(set_to_none=True)
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(opt_d)
                    scaler_d.update()
                    last_d_loss = d_loss.item()

                # Generator update.
                with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=env.use_amp):
                    fake_logits_g = discriminator(lr_n, fake)
                    adv_loss = 0.5 * ((fake_logits_g - 1.0) ** 2).mean()
                    l1_loss = F.l1_loss(fake, hr_n)
                    response = pred_energy / target_energy.clamp_min(1e-6)
                    phys_loss = (response - 1.0).abs().mean()
                    g_loss = adv_loss + args.lambda_l1 * l1_loss + args.lambda_physics * phys_loss

                opt_g.zero_grad(set_to_none=True)
                scaler_g.scale(g_loss).backward()
                scaler_g.step(opt_g)
                scaler_g.update()

                bs = lr.shape[0]
                seen += bs
                g_running += g_loss.item() * bs
                d_running += last_d_loss * bs
                l1_running += l1_loss.item() * bs
                phys_running += phys_loss.item() * bs

                # Progress log every 50 steps
                if step % 50 == 0:
                    elapsed = _time.time() - epoch_t0
                    sps = seen / elapsed if elapsed > 0 else 0
                    print(
                        f"  [ep {epoch:03d} step {step:5d}] "
                        f"g={g_running/seen:.4f} d={d_running/seen:.4f} "
                        f"l1={l1_running/seen:.4f} phys={phys_running/seen:.4f} "
                        f"| {seen} samples | {sps:.0f} samp/s | {elapsed:.0f}s elapsed",
                        flush=True,
                    )

                if args.max_train_batches is not None and step >= args.max_train_batches:
                    break

            train_metrics: dict[str, object] = {
                "epoch": epoch,
                "train_g_loss": g_running / max(seen, 1),
                "train_d_loss": d_running / max(seen, 1),
                "train_l1": l1_running / max(seen, 1),
                "train_phys": phys_running / max(seen, 1),
            }

            val_metrics = evaluate(generator, val_loader, stats, env, hr_size, args.max_val_batches)
            train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            print(
                f"epoch {epoch:03d} "
                f"g={train_metrics['train_g_loss']:.5f} "
                f"d={train_metrics['train_d_loss']:.5f} "
                f"l1={train_metrics['train_l1']:.5f} "
                f"phys={train_metrics['train_phys']:.5f} "
                f"val_l1={train_metrics['val_l1']:.5f} "
                f"val_psnr={train_metrics['val_psnr_norm']:.2f} "
                f"val_resp={train_metrics['val_response']:.4f}"
            )
            log_file.write(json.dumps(train_metrics) + "\n")
            log_file.flush()

            ckpt = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": opt_g.state_dict(),
                "optimizer_d": opt_d.state_dict(),
                "stats": stats.to_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")

            # Save a visual sample from the first val batch.
            sample_batch = next(iter(val_loader))
            save_sample_batch(generator, sample_batch, stats, env, hr_size, sample_dir / f"epoch_{epoch:03d}.npz")

            if val_metrics["l1"] < best_val:
                best_val = val_metrics["l1"]
                torch.save(ckpt, ckpt_dir / "best.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SRGAN for jet image super-resolution")

    # --- original flags (names/defaults unchanged) ---
    parser.add_argument("--data-dir", type=str, default="datasets",
                        help="[deprecated alias] Directory with parquet files — use --dataset-path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to store checkpoints and logs (auto-named if omitted)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stats-batch-size", type=int, default=128)
    parser.add_argument("--max-stats-batches", type=int, default=None)
    parser.add_argument("--batch-buffer-size", type=int, default=8,
                        help="Reservoir size in units of batch_size (parquet only)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=50.0)
    parser.add_argument("--lambda-physics", type=float, default=10.0)
    parser.add_argument("--gen-channels", type=int, default=64)
    parser.add_argument("--gen-blocks", type=int, default=8)
    parser.add_argument("--d-lr-ratio", type=float, default=0.5)
    parser.add_argument("--n-critic", type=int, default=1)
    parser.add_argument("--real-label", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    # --- new flags ---
    parser.add_argument("--dataset-type", choices=["parquet", "hdf5"], default="parquet",
                        help="Dataset format")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to dataset directory (overrides --data-dir)")
    parser.add_argument("--hr-size", type=int, nargs=2, default=[125, 125], metavar=("H", "W"),
                        help="Target HR spatial size (H W)")
    parser.add_argument("--scale-factor", type=float, default=2.0,
                        help="Downscale factor for synthetic LR generation (HDF5 only)")
    parser.add_argument("--skip-stats-cache", action="store_true",
                        help="Recompute normalization stats even if cache exists")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --dataset-path takes precedence over legacy --data-dir.
    if args.dataset_path is None:
        args.dataset_path = args.data_dir

    # Auto-name output dir to avoid collisions across resolution/dataset experiments.
    if args.output_dir is None:
        h, w = args.hr_size
        args.output_dir = f"outputs/srgan_{args.dataset_type}_x{args.scale_factor}_{h}x{w}"

    train(args)


if __name__ == "__main__":
    main()
