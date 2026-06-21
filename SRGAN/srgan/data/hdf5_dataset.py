"""Lazy map-style HDF5 dataset for CaloChallenge Dataset 2.

CaloChallenge Dataset 2 geometry (documented at calochallenge.github.io):
    showers : (N, 6480) = (N, 45_r * 144_phi)  — single calorimeter layer
    incident_energies : (N, 1)                  — GeV, used as physics proxy

Since the parquet dataset has 3 channels (ECAL, HCAL, Tracks), we replicate
the single calorimeter layer to 3 channels so the model interface is unchanged.

Worker safety:
    HDF5 file handles cannot be shared across forked processes. We store only
    the file path and re-open it in __getitem__ if the current PID changed.
    This is the standard pattern for h5py + DataLoader multiprocessing.

LR generation:
    CaloChallenge has no explicit low-resolution field. We synthesise LR by
    downsampling the HR shower image with bicubic interpolation (antialias=True)
    using the requested scale_factor. This mirrors what the parquet loader
    already has as its native LR/HR split.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

# CaloChallenge Dataset 2: 45 radial bins x 144 azimuthal bins = 6480 cells.
_CALO_R = 45
_CALO_PHI = 144
_N_CHANNELS = 3  # replicate single layer to match parquet's (ECAL, HCAL, Tracks)


class HDF5JetDataset(Dataset):
    """Map-style dataset over one HDF5 file (CaloChallenge Dataset 2 format).

    Each item is a dict with keys matching ParquetJetSRDataset:
        lr   : Tensor (3, H_lr, W_lr)
        hr   : Tensor (3, _CALO_R, _CALO_PHI) = (3, 45, 144)
        pt   : Tensor scalar  — incident_energy proxy (zero if absent)
        m0   : Tensor scalar  — zeros (not present in this dataset)
        y    : Tensor scalar  — zeros (no class label in this dataset)
    """

    def __init__(
        self,
        path: Path,
        scale_factor: float = 2.0,
        hr_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.scale_factor = scale_factor
        self.hr_size = hr_size  # if set, resize HR to this before downsampling

        with h5py.File(self.path, "r", swmr=True) as f:
            self._length = int(f["showers"].shape[0])
            self._has_energies = "incident_energies" in f

        # Opened lazily in __getitem__; re-opened if PID changes after fork.
        self._file: h5py.File | None = None
        self._pid: int = -1

    def _get_file(self) -> h5py.File:
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
            self._file = h5py.File(self.path, "r", swmr=True)
            self._pid = pid
        return self._file

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        f = self._get_file()

        # Read single sample — O(1) RAM, no full-file load.
        shower = f["showers"][idx]  # (6480,) float64
        hr_np = shower.reshape(_CALO_R, _CALO_PHI).astype(np.float32)
        hr = torch.from_numpy(hr_np).unsqueeze(0).expand(_N_CHANNELS, -1, -1).clone()
        # (3, 45, 144)

        # Optional: resize HR to a different spatial size before downsampling.
        if self.hr_size is not None:
            hr = F.interpolate(
                hr.unsqueeze(0).float(),
                size=self.hr_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0).clamp_min(0.0)

        # Synthesise LR by downsampling HR.
        h, w = hr.shape[-2], hr.shape[-1]
        lr_h = max(1, round(h / self.scale_factor))
        lr_w = max(1, round(w / self.scale_factor))
        lr = F.interpolate(
            hr.unsqueeze(0).float(),
            size=(lr_h, lr_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp_min(0.0)

        if self._has_energies:
            pt_val = float(f["incident_energies"][idx].flat[0])
        else:
            pt_val = 0.0

        return {
            "lr": lr,
            "hr": hr.float(),
            "pt": torch.tensor(pt_val, dtype=torch.float32),
            "m0": torch.zeros((), dtype=torch.float32),
            "y": torch.zeros((), dtype=torch.int64),
        }

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def build_hdf5_dataset(
    data_dir: Path,
    scale_factor: float = 2.0,
    hr_size: tuple[int, int] | None = None,
) -> ConcatDataset:
    """Load all dataset_2_*.hdf5 files in data_dir as a single ConcatDataset."""
    files = sorted(data_dir.glob("dataset_2_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No dataset_2_*.hdf5 files found in {data_dir}")
    datasets = [HDF5JetDataset(f, scale_factor=scale_factor, hr_size=hr_size) for f in files]
    return ConcatDataset(datasets)
