# hdf5_run_02 — Experiment Notes

**Date:** 2026-06-12
**Dataset:** CaloChallenge Dataset 2 (HDF5)
**Status:** Complete — 20 epochs, converging, best result so far on HDF5

## Run configuration

| Parameter | Value |
|---|---|
| epochs | 20 |
| batch_size | 32 |
| lr | 2e-4 |
| d_lr_ratio | 0.5 |
| n_critic | 2 |
| lambda_l1 | 50 |
| lambda_physics | 15 |
| gen_channels | 64 |
| gen_blocks | 8 |
| hr_size | 45 × 144 |
| scale_factor | 2.0 |
| val_ratio | 0.15 |
| max_train_batches | 500 |
| seed | 42 |

## Data coverage

- Total dataset: 200,000 showers (2 HDF5 files × 100K)
- Train split: 170,000 showers (85%), Val split: 30,000 showers (15%)
- Batches per epoch: 500 (capped by `--max-train-batches`)
- Showers seen per epoch: 500 × 32 = **16,000 (8% of train split)**
- Total showers processed: 20 × 16,000 = **320,000 passes** (same 16K seen 20×)

## Per-epoch metrics

| Epoch | train_g | train_d | train_l1 | train_phys | val_l1 | val_psnr | val_response |
|---|---|---|---|---|---|---|---|
| 1 | 16.399 | 0.155 | 0.287 | 0.112 | 0.2505 | 6.49 | 0.9934 |
| 2 | 13.802 | 0.037 | 0.247 | 0.064 | 0.2369 | 6.73 | 1.0355 |
| 3 | 12.941 | 0.023 | 0.233 | 0.053 | 0.2248 | 6.81 | 1.0208 |
| 4 | 12.381 | 0.025 | 0.224 | 0.048 | 0.2274 | 6.72 | 0.9150 |
| 5 | 11.992 | 0.024 | 0.217 | 0.044 | 0.2192 | 6.93 | 1.0854 |
| 6 | 11.695 | 0.022 | 0.212 | 0.041 | 0.2099 | 6.89 | 1.0651 |
| 7 | 11.379 | 0.024 | 0.207 | 0.037 | 0.2050 | 6.92 | 0.9596 |
| 8 | 11.210 | 0.030 | 0.204 | 0.035 | 0.2037 | 6.95 | 1.0529 |
| 9 | 11.164 | 0.027 | 0.204 | 0.033 | 0.1988 | 6.95 | 1.0039 |
| 10 | 10.880 | 0.032 | 0.199 | 0.032 | 0.2016 | 6.84 | 0.9800 |
| 11 | 10.826 | 0.032 | 0.199 | 0.030 | 0.1959 | 6.95 | 0.9625 |
| 12 | 10.786 | 0.034 | 0.198 | 0.028 | 0.2037 | 6.83 | 0.9416 |
| 13 | 10.682 | 0.037 | 0.196 | 0.027 | 0.1950 | 7.06 | 0.9736 |
| 14 | 10.538 | 0.050 | 0.194 | 0.027 | 0.1941 | 7.01 | 1.0404 |
| 15 | 10.623 | 0.033 | 0.196 | 0.025 | 0.1949 | 7.03 | 1.0034 |
| 16 | 10.506 | 0.036 | 0.194 | 0.025 | 0.1937 | 7.01 | 1.0258 |
| 17 | 10.446 | 0.035 | 0.193 | 0.024 | 0.1937 | 6.92 | 0.9872 |
| 18 | 10.288 | 0.036 | 0.190 | 0.023 | 0.1907 | 7.01 | 0.9684 |
| 19 | 10.314 | 0.035 | 0.191 | 0.022 | 0.1920 | 6.92 | 0.9760 |
| **20** | **10.217** | **0.038** | **0.189** | **0.022** | **0.1898** | **7.01** | **0.9913** |

**Best epoch:** 20 (val_l1 = 0.1898, val_response = 0.9913)

## Key observations

### Improvements over hdf5_run_01

| Metric | run_01 (5 ep) | run_02 (20 ep) | Change |
|---|---|---|---|
| val_l1 | 0.2537 | 0.1898 | −25% |
| val_psnr | 6.44 dB | 7.01 dB | +0.57 dB |
| val_response | 1.091 | 0.991 | Much closer to 1.0 |
| D loss | 0.037 (collapsed) | 0.038 (stable) | Stable with n_critic=2 |

### What worked

- `--n-critic 2` fixed discriminator collapse — D loss stayed stable at ~0.02–0.05 across all 20 epochs (vs collapsing to 0.037 by epoch 5 in run_01)
- `--lambda-physics 15` brought response from 1.091 (run_01) to 0.991 at epoch 20
- val_response oscillates around 1.0 (range 0.915–1.085) — generator is learning the physics constraint but not yet tightly converged
- Normalization cache reused from run_01 — stats step skipped entirely, training started immediately

### What still needs fixing

| Issue | Symptom | Fix |
|---|---|---|
| Only 8% data/epoch | max_train_batches=500 | Remove cap, run on Kaggle GPU |
| Response oscillation | ±8% swing each epoch | Increase lambda_physics to 20, or add LR scheduler |
| val_l1 plateauing | 0.193–0.190 from ep 16–20 | More data per epoch needed |
| D still weak | D loss ~0.03 | Try R1 gradient penalty |

## Progress across all HDF5 runs

```
run_01: 5 epochs,  80 batches/ep  (~3%  data) → val_l1=0.2537, response=1.091
run_02: 20 epochs, 500 batches/ep (~8%  data) → val_l1=0.1898, response=0.991
target: 20 epochs, full data      (100% data) → val_l1<0.15,   response=0.98–1.02
```

## Suggested next run

```bash
python train_srgan.py \
    --dataset-type hdf5 \
    --dataset-path ../datasets/calochallenge_dataset2 \
    --epochs 20 \
    --batch-size 32 \
    --n-critic 2 \
    --lambda-l1 50 --lambda-physics 20 \
    --hr-size 45 144 --scale-factor 2.0 \
    --val-ratio 0.15 \
    --seed 42 \
    --output-dir experiments/2026-06-12/hdf5_run_03
```

Remove `--max-train-batches` for full data — run on Kaggle T4 GPU for best results.
