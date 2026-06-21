"""Streaming parquet dataset with worker-safe sharding and reservoir shuffle.

Design decisions:
- IterableDataset: streams one row-group at a time, never loads a full file.
- Worker sharding: each DataLoader worker handles a disjoint file subset, so
  num_workers > 0 is safe as long as pyarrow use_threads=True is kept (it is).
  In practice we use num_workers=0 and wrap with prefetch_generator instead,
  because fork+pyarrow is unreliable on macOS / some Linux kernels.
- Reservoir shuffle: bounded memory, uniform per-sample distribution within a
  buffer window. Avoids the deque pattern that produced correlated batches.
- Yields individual (C,H,W) samples — DataLoader collates into batches. This
  enables pin_memory and prefetch_factor to work correctly.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from .normalization import IMAGE_COLUMNS, batch_to_tensor


class ParquetJetSRDataset(IterableDataset):
    """Lazy streaming dataset over a list of parquet files.

    Each item yielded is a dict with keys:
        lr   : Tensor (3, H_lr, W_lr)  — raw energy, float32
        hr   : Tensor (3, H_hr, W_hr)  — raw energy, float32
        pt   : Tensor scalar            — transverse momentum
        m0   : Tensor scalar            — invariant mass
        y    : Tensor scalar (int64)    — class label
    Normalization is applied in the training loop, not here.
    """

    _PHYS_COLUMNS = ("X_jets_LR", "X_jets", "pt", "m0", "y")

    def __init__(
        self,
        files: Sequence[Path],
        batch_size: int,
        shuffle_files: bool = True,
        reservoir_size: int = 512,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.reservoir_size = max(1, reservoir_size)

    def _shard_files(self) -> list[Path]:
        files = self.files[:]
        if self.shuffle_files:
            random.shuffle(files)
        info = get_worker_info()
        if info is not None:
            # Each worker processes a disjoint slice: worker 0 gets files[0::N], etc.
            files = files[info.id :: info.num_workers]
        return files

    def _iter_samples(self, files: list[Path]) -> Iterator[dict[str, Tensor]]:
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=self.batch_size,
                columns=list(self._PHYS_COLUMNS),
                use_threads=True,
            ):
                lr_batch = batch_to_tensor(batch.column(0))  # (B,3,H,W)
                hr_batch = batch_to_tensor(batch.column(1))
                pt_list = batch.column(2).to_pylist()
                m0_list = batch.column(3).to_pylist()
                y_list = batch.column(4).to_pylist()
                for i in range(lr_batch.shape[0]):
                    yield {
                        "lr": lr_batch[i],
                        "hr": hr_batch[i],
                        "pt": torch.tensor(float(pt_list[i]), dtype=torch.float32),
                        "m0": torch.tensor(float(m0_list[i]), dtype=torch.float32),
                        "y": torch.tensor(int(y_list[i]), dtype=torch.int64),
                    }

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        files = self._shard_files()
        reservoir: list[dict[str, Tensor]] = []

        for sample in self._iter_samples(files):
            if len(reservoir) < self.reservoir_size:
                reservoir.append(sample)
            else:
                # Randomly eject one item and yield it; insert the new sample.
                eject_idx = random.randrange(len(reservoir))
                yield reservoir[eject_idx]
                reservoir[eject_idx] = sample

        # Flush remaining items in random order.
        random.shuffle(reservoir)
        yield from reservoir
