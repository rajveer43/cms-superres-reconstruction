from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm
from torch.utils.data import IterableDataset


IMAGE_COLUMNS = ("X_jets_LR", "X_jets")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files


def split_files(files: Sequence[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    if len(files) == 1:
        return list(files), list(files)
    n_val = max(1, int(round(len(files) * val_ratio)))
    n_val = min(n_val, len(files) - 1)
    return list(files[:-n_val]), list(files[-n_val:])


def _ensure_chw(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 3:
        raise ValueError(f"Expected 3D image sample, got shape {sample.shape}")
    if sample.shape[0] == 3:
        return sample
    if sample.shape[-1] == 3:
        return np.transpose(sample, (2, 0, 1))
    raise ValueError(f"Cannot infer channel dimension from shape {sample.shape}")


def batch_to_tensor(batch_col) -> Tensor:
    samples = batch_col.to_pylist()
    arr = np.stack([_ensure_chw(np.asarray(sample, dtype=np.float32)) for sample in samples], axis=0)
    return torch.from_numpy(arr)


def log1p_clip(x: Tensor) -> Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


@dataclass
class ChannelStats:
    mean: Tensor
    std: Tensor

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def stream_channel_stats(files: Sequence[Path], batch_size: int, max_batches: int | None = None) -> ChannelStats:
    total_sum = None
    total_sumsq = None
    total_count = 0

    batches_seen = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(IMAGE_COLUMNS), use_threads=True):
            for col in (0, 1):
                x = batch_to_tensor(batch.column(col)).to(torch.float64)
                x = log1p_clip(x)
                summed = x.sum(dim=(0, 2, 3))
                summed_sq = (x * x).sum(dim=(0, 2, 3))
                count = x.shape[0] * x.shape[2] * x.shape[3]

                if total_sum is None:
                    total_sum = summed
                    total_sumsq = summed_sq
                else:
                    total_sum += summed
                    total_sumsq += summed_sq
                total_count += count
            batches_seen += 1
            if max_batches is not None and batches_seen >= max_batches:
                break
        if max_batches is not None and batches_seen >= max_batches:
            break

    if total_sum is None or total_sumsq is None or total_count == 0:
        raise RuntimeError("Could not compute normalization statistics")

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return ChannelStats(mean=mean.to(torch.float32), std=std.to(torch.float32))


def normalize(x: Tensor, stats: ChannelStats) -> Tensor:
    x = log1p_clip(x)
    mean = stats.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = stats.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / std


def denormalize(x: Tensor, stats: ChannelStats) -> Tensor:
    mean = stats.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = stats.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return torch.expm1(x * std + mean).clamp_min(0.0)


class ParquetJetSRDataset(IterableDataset):
    def __init__(
        self,
        files: Sequence[Path],
        batch_size: int,
        shuffle_files: bool = True,
        shuffle_batches: bool = True,
        batch_buffer_size: int = 8,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.shuffle_batches = shuffle_batches
        self.batch_buffer_size = max(1, batch_buffer_size)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        files = self.files[:]
        if self.shuffle_files:
            random.shuffle(files)

        for path in files:
            parquet = pq.ParquetFile(path)
            buffer: deque[dict[str, Tensor]] = deque()
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=list(IMAGE_COLUMNS), use_threads=True):
                item = {
                    "lr": batch_to_tensor(batch.column(0)),
                    "hr": batch_to_tensor(batch.column(1)),
                }
                buffer.append(item)
                if len(buffer) >= self.batch_buffer_size:
                    if self.shuffle_batches:
                        shuffled = list(buffer)
                        random.shuffle(shuffled)
                        buffer = deque(shuffled)
                    while buffer:
                        yield buffer.popleft()
            if self.shuffle_batches:
                shuffled = list(buffer)
                random.shuffle(shuffled)
                buffer = deque(shuffled)
            while buffer:
                yield buffer.popleft()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.1 * self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_blocks: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_channels))
        layers.extend(
            [
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(base_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, lr: Tensor, target_size: tuple[int, int]) -> Tensor:
        lr_up = F.interpolate(lr, size=target_size, mode="bicubic", align_corners=False)
        return lr_up + self.net(lr_up)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, lr: Tensor, hr: Tensor) -> Tensor:
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        x = torch.cat([lr_up, hr], dim=1)
        return self.net(x)


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    dataset: ParquetJetSRDataset,
    stats: ChannelStats,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    generator.eval()
    l1_total = 0.0
    pixel_total = 0
    num_batches = 0

    for batch in dataset:
        lr = normalize(batch["lr"].to(device), stats)
        hr = normalize(batch["hr"].to(device), stats)
        fake = generator(lr, hr.shape[-2:])
        l1 = F.l1_loss(fake, hr, reduction="sum").item()
        l1_total += l1
        pixel_total += hr.numel()
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break

    return {"l1": l1_total / max(pixel_total, 1)}


def save_sample_batch(
    generator: nn.Module,
    batch: dict[str, Tensor],
    stats: ChannelStats,
    device: torch.device,
    path: Path,
) -> None:
    generator.eval()
    with torch.no_grad():
        lr = normalize(batch["lr"].to(device), stats)
        hr = normalize(batch["hr"].to(device), stats)
        fake = generator(lr, hr.shape[-2:])
        lr_img = denormalize(lr.cpu(), stats)
        fake_img = denormalize(fake.cpu(), stats)
        hr_img = denormalize(hr.cpu(), stats)
        np.savez_compressed(
            path,
            lr=lr_img.numpy(),
            fake=fake_img.numpy(),
            hr=hr_img.numpy(),
        )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    files = discover_parquet_files(data_dir)
    train_files, val_files = split_files(files, args.val_ratio)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    stats = stream_channel_stats(train_files, batch_size=args.stats_batch_size, max_batches=args.max_stats_batches)
    with (out_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    train_dataset = ParquetJetSRDataset(
        train_files,
        batch_size=args.batch_size,
        shuffle_files=True,
        shuffle_batches=True,
        batch_buffer_size=args.batch_buffer_size,
    )
    val_dataset = ParquetJetSRDataset(
        val_files,
        batch_size=args.batch_size,
        shuffle_files=False,
        shuffle_batches=False,
        batch_buffer_size=1,
    )

    best_val = math.inf
    log_path = out_dir / "metrics.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            generator.train()
            discriminator.train()

            g_running = 0.0
            d_running = 0.0
            l1_running = 0.0
            phys_running = 0.0
            seen = 0

            for step, batch in enumerate(train_dataset, start=1):
                lr = normalize(batch["lr"].to(device), stats)
                hr = normalize(batch["hr"].to(device), stats)

                fake = generator(lr, hr.shape[-2:])
                fake_raw = denormalize(fake, stats)
                hr_raw = denormalize(hr, stats)
                pred_energy = fake_raw.sum(dim=(1, 2, 3))
                target_energy = hr_raw.sum(dim=(1, 2, 3))

                # Update discriminator.
                real_logits = discriminator(lr, hr)
                fake_logits = discriminator(lr, fake.detach())
                d_loss = 0.5 * ((real_logits - 1.0) ** 2 + fake_logits**2).mean()

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                # Update generator.
                fake_logits = discriminator(lr, fake)
                adv_loss = 0.5 * ((fake_logits - 1.0) ** 2).mean()
                l1_loss = F.l1_loss(fake, hr)
                phys_loss = F.l1_loss(torch.log1p(pred_energy), torch.log1p(target_energy))
                g_loss = adv_loss + args.lambda_l1 * l1_loss + args.lambda_physics * phys_loss

                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_g.step()

                batch_size = lr.shape[0]
                seen += batch_size
                g_running += g_loss.item() * batch_size
                d_running += d_loss.item() * batch_size
                l1_running += l1_loss.item() * batch_size
                phys_running += phys_loss.item() * batch_size

                if args.max_train_batches is not None and step >= args.max_train_batches:
                    break

            train_metrics = {
                "epoch": epoch,
                "train_g_loss": g_running / max(seen, 1),
                "train_d_loss": d_running / max(seen, 1),
                "train_l1": l1_running / max(seen, 1),
                "train_phys": phys_running / max(seen, 1),
            }

            val_metrics = evaluate(
                generator,
                val_dataset,
                stats,
                device,
                max_batches=args.max_val_batches,
            )
            train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            print(
                f"epoch {epoch:03d} "
                f"g={train_metrics['train_g_loss']:.5f} "
                f"d={train_metrics['train_d_loss']:.5f} "
                f"l1={train_metrics['train_l1']:.5f} "
                f"phys={train_metrics['train_phys']:.5f} "
                f"val_l1={train_metrics['val_l1']:.5f}"
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

            sample_batch = next(iter(val_dataset))
            save_sample_batch(generator, sample_batch, stats, device, sample_dir / f"epoch_{epoch:03d}.npz")

            if val_metrics["l1"] < best_val:
                best_val = val_metrics["l1"]
                torch.save(ckpt, ckpt_dir / "best.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GAN for jet image super-resolution")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Directory with parquet files")
    parser.add_argument("--output-dir", type=str, default="outputs/srgan", help="Where to store checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stats-batch-size", type=int, default=128, help="Batch size for normalization statistics")
    parser.add_argument("--max-stats-batches", type=int, default=None, help="Limit batches used to estimate normalization statistics")
    parser.add_argument("--batch-buffer-size", type=int, default=8, help="In-memory batch shuffle buffer")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=50.0, help="Weight on pixel reconstruction loss")
    parser.add_argument("--lambda-physics", type=float, default=10.0, help="Weight on total-energy response loss")
    parser.add_argument("--val-ratio", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit training batches per epoch for debugging")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches for debugging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
