# Dataset Compatibility & Normalization

This document defines the exact preprocessing each loader applies so that
GAN outputs trained on parquet vs. HDF5 are directly comparable (apple-to-apple).

## Normalization pipeline (identical for both loaders)

```
raw pixel value
    → clamp(x, min=0)          # clip negative values (both datasets can have small negatives)
    → log1p(x)                 # log-scale transform: compresses the long tail
    → (x - channel_mean) / channel_std   # per-channel z-score
```

Inverse (for visualisation and physics metrics):
```
z-score tensor
    → x * std + mean
    → expm1(x)
    → clamp(x, min=0)
```

Channel stats (mean, std) are computed by streaming the **training split only**
and saved to `outputs/<run>/normalization.json`. They are reused on subsequent
runs unless `--skip-stats-cache` is passed.

## Parquet loader (`srgan/data/parquet_dataset.py`)

| Property | Value |
|---|---|
| Format | Apache Parquet, pyarrow row-group streaming |
| LR field | `X_jets_LR` column, shape `(3, 64, 64)`, native |
| HR field | `X_jets` column, shape `(3, 125, 125)`, native |
| Physics | `pt`, `m0`, `y` columns read alongside images |
| RAM usage | One row-group at a time (never full file) |
| Shuffle | Reservoir shuffle, buffer = `batch_size * batch_buffer_size` |
| Workers | num_workers=0 in DataLoader; background prefetch thread |

## HDF5 loader (`srgan/data/hdf5_dataset.py`)

Source: CaloChallenge Dataset 2 (`dataset_2_1.hdf5`, `dataset_2_2.hdf5`)

| Property | Value |
|---|---|
| Format | HDF5 (SWMR mode), map-style Dataset |
| `showers` key | `(N, 6480)` = `(N, 45_r × 144_phi)`, float64 |
| `incident_energies` key | `(N, 1)`, float64, used as `pt` proxy |
| HR construction | Reshape `(6480,)` → `(45, 144)` → replicate to `(3, 45, 144)` |
| LR construction | Bicubic downsample of HR by `scale_factor` (antialias=True) |
| Physics | `pt` = incident energy; `m0`, `y` = zeros (not present) |
| RAM usage | Single-index h5py slice per `__getitem__` call |
| Workers | Supports `num_workers > 0`; handle re-opened per-PID after fork |

## Apple-to-apple comparison rules

1. **Same normalization pipeline** — log1p + channel z-score applied identically.
2. **Same metrics** — `val_l1`, `val_psnr_norm` (max_val=1.0), `val_response` (Σpred/Σtarget).
3. **Different native resolution** — parquet HR is `(3, 125, 125)`; HDF5 HR is `(3, 45, 144)`.
   Use `--hr-size H W` to resize HDF5 images to match parquet before training.
   Example: `--hr-size 125 125 --scale-factor 1.953` approximates the parquet scale ratio.
4. **Channel meaning differs** — parquet channels are ECAL/HCAL/Tracks; HDF5 replicates a
   single calorimeter layer to 3 channels. Cross-dataset channel comparisons are not meaningful;
   only aggregate (sum) energy metrics are comparable.
