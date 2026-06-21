"""Normalization utilities shared by all dataset loaders.

Pipeline: raw energy → log1p_clip → channel-wise z-score.
Both parquet and HDF5 loaders apply this exact sequence so that
metrics are computed on the same scale (see docs/DATASET_COMPAT.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor


IMAGE_COLUMNS = ("X_jets_LR", "X_jets")


@dataclass
class ChannelStats:
    mean: Tensor
    std: Tensor

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStats":
        return cls(mean=torch.tensor(d["mean"]), std=torch.tensor(d["std"]))

    @classmethod
    def load(cls, path: Path) -> "ChannelStats":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def log1p_clip(x: Tensor) -> Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def normalize(x: Tensor, stats: ChannelStats) -> Tensor:
    x = log1p_clip(x)
    mean = stats.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = stats.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / std


def denormalize(x: Tensor, stats: ChannelStats) -> Tensor:
    mean = stats.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = stats.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return torch.expm1(x * std + mean).clamp_min(0.0)


def _ensure_chw(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 3:
        raise ValueError(f"Expected 3D image sample, got shape {sample.shape}")
    if sample.shape[0] == 3:
        return sample
    if sample.shape[-1] == 3:
        return np.transpose(sample, (2, 0, 1))
    raise ValueError(f"Cannot infer channel dimension from shape {sample.shape}")


def batch_to_tensor(batch_col) -> Tensor:
    # Fast path: flatten 3-level nested list column to a contiguous buffer in
    # one shot (50× faster than to_pylist()+per-sample np.stack).
    # Data is stored as double; cast to float32 after reshape.
    try:
        col = batch_col
        flat = col.flatten().flatten().flatten()  # list<list<list<double>>> → flat double
        buf = flat.buffers()[1]
        n = len(col)
        total = len(flat)
        raw = np.frombuffer(buf, dtype=np.float64, count=total).reshape(n, -1)
        # Infer per-sample shape from first element to handle LR vs HR sizes.
        first = np.asarray(col[0].as_py(), dtype=np.float32)
        arr = raw.reshape((n, *first.shape)).astype(np.float32)
        return torch.from_numpy(arr)
    except Exception:
        pass
    # Fallback for unexpected column types.
    samples = batch_col.to_pylist()
    arr = np.stack([_ensure_chw(np.asarray(s, dtype=np.float32)) for s in samples], axis=0)
    return torch.from_numpy(arr)


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


def stream_channel_stats(
    source: Iterable[Tensor],
    max_batches: int | None = None,
) -> ChannelStats:
    """Compute per-channel mean/std by streaming batches of (B,C,H,W) tensors.

    Works for both parquet and HDF5 — callers provide a batch iterator.
    Accumulates in float64 to avoid numerical drift over large datasets.
    """
    total_sum: Tensor | None = None
    total_sumsq: Tensor | None = None
    total_count = 0
    batches_seen = 0

    for batch in source:
        x = log1p_clip(batch.to(torch.float64))
        s = x.sum(dim=(0, 2, 3))
        ssq = (x * x).sum(dim=(0, 2, 3))
        count = x.shape[0] * x.shape[2] * x.shape[3]

        if total_sum is None:
            total_sum = s
            total_sumsq = ssq
        else:
            total_sum += s
            total_sumsq += ssq
        total_count += count
        batches_seen += 1
        if max_batches is not None and batches_seen >= max_batches:
            break

    if total_sum is None or total_count == 0:
        raise RuntimeError("No data found while computing normalization statistics")

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean * mean  # type: ignore[operator]
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return ChannelStats(mean=mean.float(), std=std.float())


def stream_channel_stats_parquet(
    files: Sequence[Path],
    batch_size: int,
    max_batches: int | None = None,
) -> ChannelStats:
    """Convenience wrapper: stream parquet files directly (no DataLoader needed)."""

    def _iter() -> Iterable[Tensor]:
        seen = 0
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=list(IMAGE_COLUMNS),
                use_threads=True,
            ):
                for col_idx in (0, 1):
                    yield batch_to_tensor(batch.column(col_idx))
                    seen += 1
                    if max_batches is not None and seen >= max_batches:
                        return

    return stream_channel_stats(_iter(), max_batches=None)
