# SRGAN Codebase Documentation

**Project:** CMS Jet Super-Resolution — GAN Baseline  
**Package root:** `task/SRGAN/`  
**Last updated:** 2026-06-11

---

## Directory structure

```
SRGAN/
├── train_srgan.py          # Entry point — training loop
├── analyze_dataset.py      # EDA script for parquet files
├── analyze_results.py      # Post-training physics analysis
├── visualize_gan_output.py # Render LR/GAN/HR comparison figures
├── save_experiment.py      # Generate all experiment artifacts
├── Makefile                # smoke-parquet / smoke-hdf5 / train-final targets
├── requirements.txt        # numpy, torch, pyarrow, matplotlib, pillow, h5py
│
├── srgan/                  # Python package — all reusable code lives here
│   ├── __init__.py         # Re-exports all public symbols
│   ├── data/
│   │   ├── normalization.py    # ChannelStats, log1p+z-score pipeline
│   │   ├── parquet_dataset.py  # Lazy streaming IterableDataset
│   │   ├── hdf5_dataset.py     # Lazy map-style Dataset (CaloChallenge)
│   │   └── factory.py          # get_dataloader() — unified entry point
│   ├── models/
│   │   ├── generator.py        # Generator, ResidualBlock
│   │   └── discriminator.py    # PatchGAN Discriminator
│   └── utils/
│       ├── env.py              # Platform detection, EnvConfig
│       └── seed.py             # seed_everything()
│
├── experiments/
│   └── 2026-06-11/
│       ├── RUN_LOG.md          # Daily run log with all metrics
│       ├── hdf5_run_01/        # CaloChallenge run
│       └── parquet_run_01/     # QCDToGGQQ run
│
├── docs/
│   ├── CODEBASE.md             # This file
│   └── DATASET_COMPAT.md       # Normalization pipeline per dataset
│
├── outputs/                # Legacy training runs (pre-refactor)
├── reports/                # Legacy EDA reports
└── notebooks/              # Jupyter notebooks for EDA and visualization
```

---

## Module reference

### `srgan/utils/env.py`

**Purpose:** Single-call platform detection. All resource decisions (device, workers, AMP) come from here — no scattered `torch.cuda.is_available()` calls elsewhere.

#### `resolve_env() → EnvConfig`

Detects the runtime and returns a frozen config object. Call once at the start of `train()`.

Detection priority:
1. CUDA available → `device=cuda`, `pin_memory=True`, `use_amp=True`, `num_workers=min(4, cpu_count)` (2 on Kaggle/Colab)
2. MPS available (Apple Silicon) → `device=mps`, `num_workers=0` (fork+MPS deadlocks), `use_amp=False`
3. CPU fallback → `num_workers=min(2, cpu_count)`, `use_amp=False`

```python
from srgan.utils.env import resolve_env
env = resolve_env()
# EnvConfig(device=mps, num_workers=0, pin_memory=False, use_amp=False, ...)
```

#### `EnvConfig` fields

| Field | Type | Description |
|---|---|---|
| `device` | `torch.device` | cuda / mps / cpu |
| `num_workers` | `int` | DataLoader workers |
| `pin_memory` | `bool` | True only on CUDA |
| `persistent_workers` | `bool` | True when num_workers > 0 and not MPS |
| `prefetch_factor` | `int\|None` | None when num_workers == 0 |
| `use_amp` | `bool` | True only on CUDA |
| `dtype` | `torch.dtype` | float16 with AMP, else float32 |
| `is_kaggle` | `bool` | Detected via `/kaggle/input` |
| `is_colab` | `bool` | Detected via `COLAB_GPU` env var |

#### `prefetch_generator(iterable, maxsize=2) → _PrefetchIterator`

Wraps any iterable with one-ahead background prefetch on a daemon thread using a bounded `queue.Queue`. Used on the parquet path where `num_workers=0` in DataLoader.

---

### `srgan/data/normalization.py`

**Purpose:** The shared normalization pipeline applied identically to both parquet and HDF5 data. This is what makes cross-dataset metric comparisons valid.

#### Pipeline

```
raw energy → clamp(x, min=0) → log1p(x) → (x - channel_mean) / channel_std
```

Inverse (for visualization and physics metrics):
```
z_score → x*std + mean → expm1(x) → clamp(min=0)
```

#### `ChannelStats`

Dataclass holding per-channel `mean` and `std` tensors (shape `(C,)`).

```python
stats = ChannelStats(mean=torch.tensor([...]), std=torch.tensor([...]))
stats.save(path / "normalization.json")   # serialize
stats = ChannelStats.load(path / "normalization.json")  # deserialize
```

#### `normalize(x, stats) → Tensor`

Applies log1p + z-score in-place-safe. Moves `stats` tensors to the same device as `x`.

#### `denormalize(x, stats) → Tensor`

Inverse: returns raw energy values (always ≥ 0 after `clamp_min`).

#### `stream_channel_stats_parquet(files, batch_size, max_batches) → ChannelStats`

Streams parquet row-groups, accumulating mean/std in float64 to avoid numerical drift. Never loads a full file into RAM.

#### `discover_parquet_files(data_dir) → list[Path]`

Sorted glob for `*.parquet` in `data_dir`. Raises `FileNotFoundError` if none found.

#### `split_files(files, val_ratio) → (train_files, val_files)`

Last `ceil(N * val_ratio)` files go to val. Minimum 1 val file, minimum 1 train file.

---

### `srgan/data/parquet_dataset.py`

**Purpose:** Lazy streaming `IterableDataset` over parquet files. Never loads a full file; streams one row-group at a time.

#### `ParquetJetSRDataset`

```python
dataset = ParquetJetSRDataset(
    files=[Path("run0.parquet"), Path("run1.parquet")],
    batch_size=64,          # pyarrow row-group read size
    shuffle_files=True,     # randomise file order per epoch
    reservoir_size=512,     # shuffle buffer (samples, not batches)
)
```

Each `__iter__` call yields individual sample dicts:

```python
{
    "lr": Tensor(3, 64, 64),     # raw energy float32
    "hr": Tensor(3, 125, 125),   # raw energy float32
    "pt": Tensor scalar,          # transverse momentum (GeV)
    "m0": Tensor scalar,          # invariant mass (GeV)
    "y":  Tensor scalar int64,    # class label {0, 1}
}
```

**Normalization is NOT applied here** — done in the training loop so that raw energy is always available for physics loss computation.

#### Design decisions

| Decision | Reason |
|---|---|
| `IterableDataset` not map-style | Parquet files have no random-access index; streaming row-groups is the only RAM-safe option |
| `num_workers=0` in DataLoader | pyarrow's `use_threads=True` + multiprocessing fork = unreliable on macOS/some Linux kernels |
| Reservoir shuffle | Bounded memory, uniform per-sample distribution — better than the deque buffer it replaced |
| Worker sharding via `get_worker_info()` | If someone does use `num_workers>0`, each worker processes a disjoint file subset to avoid duplicates |
| Yields individual samples | Lets DataLoader's `pin_memory` and standard collation work correctly |

---

### `srgan/data/hdf5_dataset.py`

**Purpose:** Lazy map-style `Dataset` for CaloChallenge Dataset 2.

#### CaloChallenge Dataset 2 geometry

```
showers: (N, 6480) = (N, 45_r × 144_phi)   — single calorimeter layer
incident_energies: (N, 1)                    — GeV, used as pt proxy
```

The single layer is reshaped to `(45, 144)` and replicated to 3 channels to match the parquet interface.

#### `HDF5JetDataset`

```python
ds = HDF5JetDataset(
    path=Path("dataset_2_1.hdf5"),
    scale_factor=2.0,      # LR = bicubic downsample of HR by this factor
    hr_size=None,          # optional: resize HR before downsampling
)
item = ds[0]
# {"lr": Tensor(3,22,72), "hr": Tensor(3,45,144), "pt": ..., "m0": zeros, "y": zeros}
```

#### Worker-safety (fork-safe HDF5)

HDF5 file handles cannot be shared across forked processes. The handle is re-opened in `__getitem__` when the current PID differs from the PID at open time:

```python
def _get_file(self):
    if self._file is None or self._pid != os.getpid():
        self._file = h5py.File(self.path, "r", swmr=True)
        self._pid = os.getpid()
    return self._file
```

`swmr=True` (Single Writer Multiple Reader) allows safe concurrent reads from DataLoader workers without file corruption.

#### `build_hdf5_dataset(data_dir, scale_factor, hr_size) → ConcatDataset`

Loads all `dataset_2_*.hdf5` files in `data_dir` and concatenates them.

---

### `srgan/data/factory.py`

**Purpose:** Single entry point that returns a `(DataLoader, ChannelStats)` tuple regardless of dataset type.

#### `get_dataloader(...) → (DataLoader, ChannelStats)`

```python
train_loader, stats = get_dataloader(
    dataset_type="parquet",        # or "hdf5"
    path=Path("../datasets"),
    split="train",                 # or "val"
    env=env,                       # EnvConfig from resolve_env()
    batch_size=32,
    val_ratio=0.33,
    stats_cache_path=out_dir / "normalization.json",
    skip_stats_cache=False,        # set True to force recompute
)
```

**Stats caching:** On first run, stats are computed by streaming the training split and saved to `stats_cache_path`. On subsequent runs the JSON is loaded directly — critical on Kaggle/Colab where recomputing wastes 10–20 minutes.

**Parquet path:** `num_workers=0`, wrapped with `_WrappedPrefetchLoader` (one-ahead background thread).  
**HDF5 path:** `num_workers=env.num_workers`, standard DataLoader with `pin_memory`, `persistent_workers`, `prefetch_factor`.

---

### `srgan/models/generator.py`

#### `Generator`

Bicubic-residual architecture:

```
lr_input
  → F.interpolate(bicubic) to target_size    # skip connection
  → 7×7 conv → ReLU
  → N × ResidualBlock(base_channels)
  → 3×3 conv → InstanceNorm → ReLU
  → 7×7 conv
  → + skip connection
  → output
```

`target_size` is passed at forward time — the same checkpoint works for any output resolution.

```python
gen = Generator(in_channels=3, base_channels=64, num_blocks=8)
fake = gen(lr_normalized, target_size=(125, 125))
```

#### `ResidualBlock`

Standard residual with scaled output (`0.1 * block(x)`). Instance norm instead of batch norm for stability at small batch sizes.

---

### `srgan/models/discriminator.py`

#### `Discriminator`

PatchGAN with spectral normalization on all conv layers.

```
concat([bicubic_upsample(lr), hr])   # (B, 6, H, W)
  → 4 × strided SpectralNorm Conv + InstanceNorm + LeakyReLU
  → SpectralNorm Conv → patch logits
```

Input spatial size is fully dynamic (derived from `hr.shape[-2:]`) — any resolution works without architecture changes.

LSGAN loss (MSE on logits, not BCE):
- Real: `(logits - 0.9)² / 2` (one-sided label smoothing, `real_label=0.9`)
- Fake: `logits² / 2`
- Generator: `(logits - 1.0)² / 2`

---

### `train_srgan.py`

Entry point. Key flags:

| Flag | Default | Description |
|---|---|---|
| `--dataset-type` | `parquet` | `parquet` or `hdf5` |
| `--dataset-path` | `datasets/` | Path to dataset directory |
| `--hr-size H W` | `125 125` | Target HR spatial size |
| `--scale-factor` | `2.0` | Synthetic LR scale (HDF5 only) |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `2e-4` | Generator learning rate |
| `--d-lr-ratio` | `0.5` | Discriminator LR = lr × ratio (TTUR) |
| `--n-critic` | `1` | Update D once every N G steps |
| `--lambda-l1` | `50.0` | Pixel reconstruction loss weight |
| `--lambda-physics` | `10.0` | Energy response loss weight |
| `--real-label` | `0.9` | One-sided label smoothing for real samples |
| `--skip-stats-cache` | off | Force recompute normalization stats |
| `--output-dir` | auto-named | Where to write checkpoints and logs |

#### Loss composition

```
G_loss = adv_loss + λ_l1 * L1(fake_n, hr_n) + λ_phys * mean(|ΣE_pred/ΣE_true - 1|)
D_loss = 0.5 * ((real_logits - real_label)² + fake_logits²).mean()
```

The physics loss operates on **denormalized** (raw energy) tensors to avoid the log-compression masking energy errors.

#### AMP / mixed precision

Wrapped with `torch.autocast` on the forward pass and `torch.amp.GradScaler` on backward. Both are no-ops when `env.use_amp=False` (MPS, CPU).

#### Checkpoint format

```python
{
    "epoch": int,
    "generator": state_dict,
    "discriminator": state_dict,
    "optimizer_g": state_dict,
    "optimizer_d": state_dict,
    "stats": {"mean": [...], "std": [...]},
    "args": vars(args),   # full CLI args — enables reproducible re-runs
}
```

---

### `save_experiment.py`

Standalone post-training artifact generator. Loads checkpoint, runs inference on val split, produces all figures and JSON metrics. Re-runnable at any time without retraining.

```bash
python save_experiment.py \
    --run-dir experiments/2026-06-11/parquet_run_01 \
    --dataset-path ../datasets \
    --dataset-type parquet \
    --hr-size 125 125 \
    --max-val-samples 500   # cap for fast iteration; omit for full val split
```

#### Outputs per run

| Path | Content |
|---|---|
| `figures/training/loss_curves.png` | G/D/L1 loss vs epoch |
| `figures/training/physics_response_curve.png` | Σ_GAN/Σ_HR vs epoch |
| `figures/training/psnr_curve.png` | val PSNR vs epoch |
| `figures/reconstruction/sample_grid_{0-3}.png` | LR/GAN/HR side-by-side (log1p) |
| `figures/reconstruction/mean_shower_comparison.png` | Per-channel mean shower |
| `figures/reconstruction/residual_map.png` | (GAN−HR) residual, RdBu_r |
| `figures/physics/energy_response_hist.png` | GAN vs bicubic response distribution |
| `figures/physics/energy_scatter.png` | E_pred vs E_true (log-log) |
| `figures/physics/energy_scatter_residual.png` | Relative error vs E_true |
| `figures/physics/sampling_fraction_hist.png` | E_dep/E_pt distributions |
| `figures/physics/response_by_class.png` | *(parquet only)* GAN response per QCD class |
| `figures/physics/response_vs_pt.png` | *(parquet only)* Response vs pT scatter |
| `figures/physics/response_vs_m0.png` | *(parquet only)* Response vs m0 scatter |
| `figures/correlations/radial_profile.png` | Mean energy vs row bin |
| `figures/correlations/azimuthal_profile.png` | Mean energy vs column/phi bin |
| `figures/correlations/pixel_correlation.png` | HR vs GAN pixel scatter (log1p, Pearson r) |
| `figures/correlations/ssim_hist.png` | Per-sample SSIM histogram |
| `figures/correlations/sparsity_comparison.png` | LR/GAN/HR sparsity bar chart |
| `figures/correlations/channel_correlation_heatmap.png` | *(parquet only)* Inter-channel Pearson r |
| `physics_metrics.json` | Response, sampling fraction, relative error stats |
| `correlation_metrics.json` | Pearson r, SSIM, profile MSEs, sparsity |
| `EXPERIMENT_REPORT.md` | Full markdown report with all tables and figure links |

---

## Reproducing a run

```bash
cd task/SRGAN

# Parquet (QCDToGGQQ)
python train_srgan.py \
    --dataset-type parquet --dataset-path ../datasets \
    --epochs 5 --batch-size 32 --max-train-batches 80 \
    --output-dir experiments/2026-06-11/parquet_run_01

python save_experiment.py \
    --run-dir experiments/2026-06-11/parquet_run_01 \
    --dataset-path ../datasets --dataset-type parquet \
    --hr-size 125 125 --max-val-samples 500

# HDF5 (CaloChallenge)
python train_srgan.py \
    --dataset-type hdf5 --dataset-path ../datasets/calochallenge_dataset2 \
    --epochs 5 --batch-size 32 --max-train-batches 80 \
    --hr-size 45 144 --output-dir experiments/2026-06-11/hdf5_run_01

python save_experiment.py \
    --run-dir experiments/2026-06-11/hdf5_run_01 \
    --dataset-path ../datasets/calochallenge_dataset2 --dataset-type hdf5 \
    --hr-size 45 144 --max-val-samples 500
```

Random seed is fixed at 42 by default (`--seed 42`). Normalization stats are cached in `<output-dir>/normalization.json` and reloaded on repeat runs automatically.
