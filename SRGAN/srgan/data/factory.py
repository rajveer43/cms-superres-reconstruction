"""DataLoader factory — single entry point for all dataset types.

get_dataloader() returns a torch.utils.data.DataLoader regardless of whether
the underlying format is parquet or HDF5.  Resource settings (num_workers,
pin_memory, persistent_workers, prefetch_factor) come from EnvConfig so that
no caller has to know which platform it runs on.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from ..utils.env import EnvConfig, prefetch_generator
from .hdf5_dataset import build_hdf5_dataset
from .normalization import (
    ChannelStats,
    discover_parquet_files,
    split_files,
    stream_channel_stats,
    stream_channel_stats_parquet,
)
from .parquet_dataset import ParquetJetSRDataset


def _parquet_loader(
    path: Path,
    split: str,
    env: EnvConfig,
    batch_size: int,
    val_ratio: float,
    reservoir_size: int,
) -> DataLoader:
    files = discover_parquet_files(path)
    train_files, val_files = split_files(files, val_ratio)
    chosen = train_files if split == "train" else val_files

    dataset = ParquetJetSRDataset(
        files=chosen,
        batch_size=batch_size,
        shuffle_files=(split == "train"),
        reservoir_size=reservoir_size if split == "train" else 1,
    )
    nw = env.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=env.pin_memory,
        prefetch_factor=env.prefetch_factor if nw > 0 else None,
        persistent_workers=env.persistent_workers and nw > 0,
        drop_last=(split == "train"),
    )


def _hdf5_loader(
    path: Path,
    split: str,
    env: EnvConfig,
    batch_size: int,
    val_ratio: float,
    scale_factor: float,
    hr_size: tuple[int, int] | None,
) -> DataLoader:
    dataset = build_hdf5_dataset(path, scale_factor=scale_factor, hr_size=hr_size)
    n = len(dataset)  # type: ignore[arg-type]
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val
    if split == "train":
        indices = list(range(n_train))
        sampler = RandomSampler(torch.utils.data.Subset(dataset, indices))
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        indices = list(range(n_train, n))
        sampler = SequentialSampler(torch.utils.data.Subset(dataset, indices))
        subset = torch.utils.data.Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=env.num_workers,
        pin_memory=env.pin_memory,
        persistent_workers=env.persistent_workers and env.num_workers > 0,
        prefetch_factor=env.prefetch_factor if env.num_workers > 0 else None,
        drop_last=(split == "train"),
    )



def get_dataloader(
    dataset_type: str,
    path: Path,
    split: str,
    env: EnvConfig,
    batch_size: int,
    val_ratio: float = 0.33,
    scale_factor: float = 2.0,
    hr_size: tuple[int, int] | None = None,
    reservoir_size: int = 512,
    stats_batch_size: int = 128,
    max_stats_batches: int | None = None,
    stats_cache_path: Path | None = None,
    skip_stats_cache: bool = False,
) -> tuple[DataLoader, ChannelStats]:
    """Return (dataloader, channel_stats) for the requested split.

    Channel stats are computed once by streaming the training split.
    On subsequent calls the cached normalization.json is reused unless
    skip_stats_cache=True is passed.
    """
    if dataset_type not in ("parquet", "hdf5"):
        raise ValueError(f"Unknown dataset_type {dataset_type!r}. Choose 'parquet' or 'hdf5'.")

    # --- normalization stats (cached) ---
    stats: ChannelStats | None = None
    if stats_cache_path is not None and stats_cache_path.exists() and not skip_stats_cache:
        stats = ChannelStats.load(stats_cache_path)
        print(f"[data] loaded stats from cache: {stats_cache_path}")
    else:
        print("[data] computing normalization statistics…")
        if dataset_type == "parquet":
            files = discover_parquet_files(path)
            train_files, _ = split_files(files, val_ratio)
            stats = stream_channel_stats_parquet(
                train_files,
                batch_size=stats_batch_size,
                max_batches=max_stats_batches,
            )
        else:
            # For HDF5: stream via a temporary DataLoader (no fork, workers=0).
            tmp_ds = build_hdf5_dataset(path, scale_factor=scale_factor, hr_size=hr_size)
            n = len(tmp_ds)  # type: ignore[arg-type]
            n_train = n - max(1, int(round(n * val_ratio)))
            n_batches_total = (n_train + stats_batch_size - 1) // stats_batch_size
            cap = max_stats_batches if max_stats_batches is not None else n_batches_total
            print(f"[data] HDF5 stats: {n_train:,} train showers, "
                  f"batch_size={stats_batch_size}, "
                  f"scanning {min(cap, n_batches_total)} / {n_batches_total} batches…")
            tmp_subset = torch.utils.data.Subset(tmp_ds, list(range(n_train)))
            tmp_dl = DataLoader(tmp_subset, batch_size=stats_batch_size, num_workers=0)

            def _hr_iter():
                seen = 0
                for batch in tmp_dl:
                    yield batch["hr"]
                    seen += 1
                    if seen % 20 == 0:
                        print(f"  [stats] {seen}/{min(cap, n_batches_total)} batches done…", flush=True)
                    if max_stats_batches is not None and seen >= max_stats_batches:
                        break

            stats = stream_channel_stats(_hr_iter())
            print(f"[data] stats computed: mean={stats.mean} std={stats.std}")

        if stats_cache_path is not None:
            stats_cache_path.parent.mkdir(parents=True, exist_ok=True)
            stats.save(stats_cache_path)
            print(f"[data] stats saved to {stats_cache_path}")

    # --- dataloader ---
    if dataset_type == "parquet":
        loader = _parquet_loader(path, split, env, batch_size, val_ratio, reservoir_size)
        if split == "train":
            loader = _WrappedPrefetchLoader(loader, env)
    else:
        loader = _hdf5_loader(path, split, env, batch_size, val_ratio, scale_factor, hr_size)

    return loader, stats


class _WrappedPrefetchLoader:
    """Thin wrapper that adds one-ahead prefetch to a DataLoader.

    Used for the parquet IterableDataset path where num_workers=0.
    Exposes the same iteration interface as DataLoader.
    """

    def __init__(self, loader: DataLoader, env: EnvConfig) -> None:
        self._loader = loader
        self._env = env

    def __iter__(self):
        return iter(prefetch_generator(self._loader))

    def __len__(self):
        return len(self._loader)  # type: ignore[arg-type]
