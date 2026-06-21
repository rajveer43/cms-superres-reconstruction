# parquet_run_01 — Experiment Notes

**Date:** 2026-06-13
**Dataset:** QCDToGGQQ CMS jet images (parquet, native LR/HR pairs)
**Status:** Complete — 20 epochs, converging, best parquet result so far

## Run configuration

| Parameter | Value |
|---|---|
| epochs | 20 |
| batch_size | 64 |
| lr | 2e-4 |
| d_lr_ratio | 0.5 |
| n_critic | 1 |
| lambda_l1 | 50 |
| lambda_physics | 15 |
| gen_channels | 64 |
| gen_blocks | 8 |
| hr_size | 125 × 125 |
| scale_factor | 2.0 |
| val_ratio | 0.15 |
| max_train_batches | 400 |
| seed | 42 |

## Data coverage

- Total dataset: **139,306 jet images** (3 parquet files: 36,272 + 47,540 + 55,494)
- Split is **file-level** (`val_ratio=0.15` → 1 of 3 files held out):
  - Train split: **83,812 images** (files run0 + run1, 60%)
  - Val split: **55,494 images** (file run2, 40%)
- Batches per epoch: 400 (capped by `--max-train-batches`)
- Images seen per epoch: 400 × 64 = **25,600 (~31% of train split)**
- Total images processed: 20 × 25,600 = **512,000 passes** (same ~25.6K seen 20×)

## Per-epoch metrics

| Epoch | train_g | train_d | train_l1 | train_phys | val_l1 | val_psnr | val_response |
|---|---|---|---|---|---|---|---|
| 1 | 6.874 | 0.270 | 0.106 | 0.091 | 0.0990 | 2.30 | 0.9461 |
| 2 | 6.051 | 0.081 | 0.095 | 0.060 | 0.0968 | 2.19 | 1.0206 |
| 3 | 5.770 | 0.032 | 0.091 | 0.051 | 0.0938 | 2.17 | 0.9906 |
| 4 | 5.536 | 0.022 | 0.087 | 0.047 | 0.0919 | 2.32 | 0.9494 |
| 5 | 5.414 | 0.015 | 0.085 | 0.044 | 0.0908 | 2.22 | 1.0088 |
| 6 | 5.401 | 0.014 | 0.086 | 0.040 | 0.0907 | 2.33 | 0.9406 |
| 7 | 5.189 | 0.009 | 0.082 | 0.038 | 0.0893 | 2.34 | 0.9691 |
| 8 | 5.130 | 0.009 | 0.082 | 0.034 | 0.0871 | 2.38 | 0.9930 |
| 9 | 5.008 | 0.012 | 0.082 | 0.028 | 0.0867 | 2.43 | 1.0156 |
| 10 | 4.933 | 0.007 | 0.080 | 0.029 | 0.0853 | 2.44 | 1.0132 |
| 11 | 4.886 | 0.007 | 0.079 | 0.028 | 0.0844 | 2.40 | 0.9946 |
| 12 | 4.879 | 0.005 | 0.080 | 0.026 | 0.0837 | 2.42 | 1.0149 |
| 13 | 4.693 | 0.005 | 0.076 | 0.025 | 0.0822 | 2.49 | 0.9886 |
| 14 | 4.833 | 0.005 | 0.079 | 0.026 | 0.0819 | 2.46 | 1.0024 |
| 15 | 4.722 | 0.005 | 0.077 | 0.025 | 0.0820 | 2.43 | 1.0025 |
| 16 | 4.769 | 0.005 | 0.078 | 0.026 | 0.0816 | 2.42 | 1.0145 |
| 17 | 4.773 | 0.013 | 0.078 | 0.025 | 0.0811 | 2.47 | 1.0154 |
| 18 | 4.540 | 0.002 | 0.074 | 0.023 | 0.0815 | 2.43 | 0.9939 |
| 19 | 4.496 | 0.013 | 0.072 | 0.026 | 0.0797 | 2.48 | 0.9794 |
| **20** | **4.491** | **0.003** | **0.073** | **0.023** | **0.0774** | **2.57** | **0.9865** |

**Best epoch:** 20 (val_l1 = 0.0774, val_response = 0.9865)

## Key observations

### Improvements over the 5-epoch parquet baseline (2026-06-11/parquet_run_01)

| Metric | baseline (5 ep) | this run (20 ep) | Change |
|---|---|---|---|
| val_l1 | 0.1025 | 0.0774 | −24% |
| val_psnr | 2.14 dB | 2.57 dB | +0.43 dB |
| val_response | 0.925 | 0.987 | Much closer to 1.0 |
| D loss | 0.151 (unstable) | 0.003 (very confident) | D dominates |

### What worked

- **20 epochs steadily reduced val_l1** from 0.099 → 0.0774, still trending down at epoch 20 (no plateau yet)
- **`--lambda-physics 15` kept val_response tight around 1.0** — range 0.94–1.02 across all 20 epochs, best 0.9865 at epoch 20. The energy-conservation constraint is well satisfied.
- **Native LR/HR pairs** (parquet has real low-res counterparts, no synthetic downsampling) → far lower val_l1 than the HDF5 run (0.077 vs 0.190)
- Normalization cache reused — distinct channel stats (mean [0.00352, 0.00289, 0.00247], std [0.0588, 0.0444, 0.0286]) confirm ECAL/HCAL/Tracks are normalized independently; stats step skipped at startup

### What still needs fixing

| Issue | Symptom | Fix |
|---|---|---|
| D over-confident / collapsing | train_d ~0.003–0.013, very low | Raise `--n-critic` to 2, or add R1 gradient penalty |
| Low PSNR_norm (~2.5 dB) | sparse jet images penalize MSE-based PSNR | Expected for sparse calo data; track val_l1 as primary instead |
| Only ~31% data/epoch | max_train_batches=400 | Remove cap (or raise), run on GPU for full 83.8K train images |
| File-level val split | val = 1 whole file (run2) | Acceptable, but row-level shuffle split would de-correlate train/val |
| val_l1 still descending | 0.0816 → 0.0774 ep 16–20 | Train longer / more data — model not yet saturated |

## Progress across parquet runs

```
2026-06-11 run_01:  5 epochs, 400 batches/ep (~31% data) → val_l1=0.1025, response=0.925
2026-06-13 run_01: 20 epochs, 400 batches/ep (~31% data) → val_l1=0.0774, response=0.987
target:            20 epochs, full 83.8K train (100% data) → val_l1<0.06,  response=0.98–1.02
```

## Suggested next run

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_srgan.py \
    --dataset-type parquet \
    --dataset-path ../datasets \
    --epochs 20 \
    --batch-size 64 \
    --n-critic 2 \
    --lambda-l1 50 --lambda-physics 15 \
    --hr-size 125 125 --scale-factor 2.0 \
    --val-ratio 0.15 \
    --seed 42 \
    --output-dir experiments/2026-06-14/parquet_run_02
```

Remove `--max-train-batches` for full-data coverage, and `--n-critic 2` to curb the over-confident discriminator. Best results on a CUDA GPU (Kaggle T4) — the MPS batch-of-1 data path makes full-data runs slow on Mac.
