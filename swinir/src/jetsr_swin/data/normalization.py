from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

IMAGE_COLUMNS = ("X_jets_LR", "X_jets")


def _ensure_chw(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {sample.shape}")
    if sample.shape[0] == 3:
        return sample
    if sample.shape[-1] == 3:
        return np.transpose(sample, (2, 0, 1))
    raise ValueError(f"Cannot infer channel dim from shape {sample.shape}")


def batch_to_tensor(batch_col) -> Tensor:
    samples = batch_col.to_pylist()
    arr = np.stack([_ensure_chw(np.asarray(s, dtype=np.float32)) for s in samples], axis=0)
    return torch.from_numpy(arr)


def log1p_clip(x: Tensor) -> Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


@dataclass
class ChannelStats:
    mean: Tensor
    std: Tensor

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStats":
        return cls(
            mean=torch.tensor(d["mean"], dtype=torch.float32),
            std=torch.tensor(d["std"], dtype=torch.float32),
        )


def stream_channel_stats(
    files: Sequence[Path],
    batch_size: int = 128,
    max_batches: int | None = None,
) -> ChannelStats:
    """Compute log1p channel-wise mean/std streaming from parquet (no full load)."""
    total_sum: Tensor | None = None
    total_sumsq: Tensor | None = None
    total_count = 0

    batches_seen = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=list(IMAGE_COLUMNS), use_threads=True
        ):
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

    if total_sum is None or total_count == 0:
        raise RuntimeError("Could not compute normalization stats")

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
