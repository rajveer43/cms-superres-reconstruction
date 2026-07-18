from pathlib import Path

import pytest

from jetsr_swin.data import discover_parquet_files, ParquetJetSRDataset

DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not present")
def test_stream_one_batch():
    files = discover_parquet_files(DATA_DIR)
    ds = ParquetJetSRDataset(files[:1], batch_size=2, shuffle_files=False, shuffle_batches=False, max_batches=1)
    batch = next(iter(ds))
    assert batch["lr"].shape[1:] == (3, 64, 64)
    assert batch["hr"].shape[1:] == (3, 125, 125)
    assert batch["lr"].shape[0] == batch["hr"].shape[0]


@pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not present")
def test_stream_with_meta():
    files = discover_parquet_files(DATA_DIR)
    ds = ParquetJetSRDataset(files[:1], batch_size=2, include_meta=True, shuffle_files=False, shuffle_batches=False, max_batches=1)
    batch = next(iter(ds))
    for k in ("lr", "hr", "pt", "m0", "y"):
        assert k in batch
    assert batch["pt"].shape[0] == batch["lr"].shape[0]
