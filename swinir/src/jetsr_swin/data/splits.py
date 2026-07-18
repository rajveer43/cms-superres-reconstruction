from __future__ import annotations

from pathlib import Path
from typing import Sequence


def discover_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files


def split_files(
    files: Sequence[Path],
    val_ratio: float = 0.33,
    test_ratio: float = 0.0,
) -> tuple[list[Path], list[Path], list[Path]]:
    """File-level train/val/test split.

    With 3 files and default ratios (val=0.33, test=0.0), this yields 2 train / 1 val / 0 test.
    Set test_ratio > 0 to carve a test set from the tail. With 3 files and
    val_ratio=test_ratio=0.33, you get 1 train / 1 val / 1 test (good for development).
    """
    files = list(files)
    n = len(files)
    if n == 1:
        return files, files, []
    n_test = max(0, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1 - n_test)
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Bad split: train={n_train}, val={n_val}, test={n_test} from {n} files")
    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]
    return train, val, test
