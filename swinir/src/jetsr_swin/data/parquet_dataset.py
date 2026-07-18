from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow.parquet as pq
from torch import Tensor
from torch.utils.data import IterableDataset

from .normalization import batch_to_tensor

IMAGE_COLUMNS = ("X_jets_LR", "X_jets")
META_COLUMNS = ("pt", "m0", "y")


class ParquetJetSRDataset(IterableDataset):
    """Stream LR/HR (and optionally metadata) batches from parquet files.

    Yields dicts with keys: lr, hr, and (if include_meta) pt, m0, y.
    """

    def __init__(
        self,
        files: Sequence[Path],
        batch_size: int,
        shuffle_files: bool = True,
        shuffle_batches: bool = True,
        batch_buffer_size: int = 8,
        include_meta: bool = False,
        max_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.shuffle_batches = shuffle_batches
        self.batch_buffer_size = max(1, batch_buffer_size)
        self.include_meta = include_meta
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        import torch  # local import keeps module light

        files = self.files[:]
        if self.shuffle_files:
            random.shuffle(files)

        columns = list(IMAGE_COLUMNS) + (list(META_COLUMNS) if self.include_meta else [])
        emitted = 0
        for path in files:
            parquet = pq.ParquetFile(path)
            buffer: deque[dict[str, Tensor]] = deque()
            for batch in parquet.iter_batches(
                batch_size=self.batch_size, columns=columns, use_threads=True
            ):
                item: dict[str, Tensor] = {
                    "lr": batch_to_tensor(batch.column(0)),
                    "hr": batch_to_tensor(batch.column(1)),
                }
                if self.include_meta:
                    item["pt"] = torch.tensor(batch.column(2).to_pylist(), dtype=torch.float32)
                    item["m0"] = torch.tensor(batch.column(3).to_pylist(), dtype=torch.float32)
                    item["y"] = torch.tensor(batch.column(4).to_pylist(), dtype=torch.long)
                buffer.append(item)
                if len(buffer) >= self.batch_buffer_size:
                    if self.shuffle_batches:
                        shuffled = list(buffer)
                        random.shuffle(shuffled)
                        buffer = deque(shuffled)
                    while buffer:
                        yield buffer.popleft()
                        emitted += 1
                        if self.max_batches is not None and emitted >= self.max_batches:
                            return
            if self.shuffle_batches:
                shuffled = list(buffer)
                random.shuffle(shuffled)
                buffer = deque(shuffled)
            while buffer:
                yield buffer.popleft()
                emitted += 1
                if self.max_batches is not None and emitted >= self.max_batches:
                    return
