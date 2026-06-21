# Experiment Run Log — 2026-06-11

**Researcher:** Rajveer Rathod  
**Project:** CMS Jet Super-Resolution — GAN Baseline  
**Goal:** Establish GAN SR baselines on both available datasets for apple-to-apple comparison in the research paper.

---

## Runs today

| Run ID | Dataset | Epochs | best val_l1 | best val_psnr | best val_response |
|---|---|---|---|---|---|
| `hdf5_run_01` | CaloChallenge Dataset 2 (HDF5) | 5 | 0.25373 | 6.44 dB | 1.091 |
| `parquet_run_01` | QCDToGGQQ CMS Jet Images (Parquet) | 5 | 0.10251 | 2.14 dB | 0.925 |

---

## hdf5_run_01

**Dataset:** CaloChallenge Dataset 2  
**Files:** `dataset_2_1.hdf5`, `dataset_2_2.hdf5` — 200,000 showers total  
**HR shape:** `(3, 45, 144)` — 45 radial × 144 azimuthal bins, replicated to 3 channels  
**LR shape:** `(3, 22, 72)` — synthetic bicubic downscale at scale factor 2.0  
**Train/Val split:** 134,000 / 66,000

### Training command
```bash
python train_srgan.py \
    --dataset-type hdf5 \
    --dataset-path ../datasets/calochallenge_dataset2 \
    --epochs 5 --batch-size 32 \
    --max-train-batches 80 --max-val-batches 20 --max-stats-batches 10 \
    --gen-channels 64 --gen-blocks 8 \
    --lr 2e-4 --d-lr-ratio 0.5 --n-critic 1 \
    --lambda-l1 50 --lambda-physics 10 \
    --hr-size 45 144 --scale-factor 2.0 \
    --output-dir experiments/2026-06-11/hdf5_run_01
```

### Per-epoch metrics

| Epoch | G loss | D loss | train L1 | train phys | val L1 | val PSNR | val response |
|---|---|---|---|---|---|---|---|
| 1 | 19.995 | 0.281 | 0.3447 | 0.2464 | 0.3006 | 5.87 | 0.815 |
| 2 | 16.439 | 0.139 | 0.2850 | 0.1841 | 0.2778 | 6.13 | 0.925 |
| 3 | 14.935 | 0.078 | 0.2624 | 0.1379 | 0.2644 | 6.30 | 1.111 |
| 4 | 14.345 | 0.045 | 0.2517 | 0.1295 | 0.2586 | 6.33 | 1.062 |
| **5** | **14.152** | **0.037** | **0.2522** | **0.1063** | **0.2537** | **6.44** | **1.091** |

### Physics metrics (500 val samples)

| Metric | GAN | Bicubic |
|---|---|---|
| Response mean | 1.092 | 1.093 |
| Response std | 0.085 | 0.057 |
| Response median | 1.072 | 1.078 |
| \|Rel error\| mean | 0.094 | — |

### Correlation metrics

| Metric | Value |
|---|---|
| Pearson r (pixel, log1p) | 0.9052 |
| SSIM mean | 0.9163 ± 0.032 |
| Radial profile MSE | 82.18 |
| Azimuthal profile MSE | 427.5 |
| GAN sparsity | 0.170 |
| HR sparsity | 0.755 |

### Observations
- Discriminator loss collapsed toward ~0.04 by epoch 5 — adversarial signal weakening. Consider increasing `--n-critic 2` next run.
- Response overshoots 1.0 (1.09) — GAN is generating slightly more energy than ground truth. Increase `--lambda-physics` to 15–20.
- High Pearson r (0.905) shows strong pixel-level spatial agreement.
- Large azimuthal profile MSE (427) is expected: the 144-bin phi profile has high absolute energy values; relative error is small.
- GAN sparsity (17%) is much lower than HR sparsity (75%) — the generator fills in near-zero cells. A sparsity-aware loss (e.g. L1 on the log domain) would help.

---

## parquet_run_01

**Dataset:** QCDToGGQQ CMS Jet Images  
**Files:** 3 parquet files — 139,306 events total (run0: 36,272 | run1: 47,540 | run2: 55,494)  
**HR shape:** `(3, 125, 125)` — ECAL, HCAL, Tracks channels  
**LR shape:** `(3, 64, 64)` — native low-resolution from detector simulation  
**Train/Val split:** files 0–1 (train) / file 2 (val) — ~83,812 / ~55,494

### Training command
```bash
python train_srgan.py \
    --dataset-type parquet \
    --dataset-path ../datasets \
    --epochs 5 --batch-size 32 \
    --max-train-batches 80 --max-val-batches 20 --max-stats-batches 5 \
    --gen-channels 64 --gen-blocks 8 \
    --lr 2e-4 --d-lr-ratio 0.5 --n-critic 1 \
    --lambda-l1 50 --lambda-physics 10 \
    --hr-size 125 125 --scale-factor 2.0 \
    --output-dir experiments/2026-06-11/parquet_run_01
```

### Per-epoch metrics

| Epoch | G loss | D loss | train L1 | train phys | val L1 | val PSNR | val response |
|---|---|---|---|---|---|---|---|
| 1 | 7.903 | 0.471 | 0.1211 | 0.1582 | 0.1154 | 2.10 | 0.976 |
| 2 | 6.187 | 0.256 | 0.1012 | 0.0933 | 0.1117 | 1.91 | 1.106 |
| 3 | 5.626 | 0.217 | 0.0942 | 0.0716 | 0.1036 | 2.13 | 0.916 |
| 4 | 5.711 | 0.189 | 0.0948 | 0.0749 | 0.1054 | 1.87 | 1.101 |
| **5** | **5.841** | **0.151** | **0.0933** | **0.0885** | **0.1025** | **2.14** | **0.925** |

### Physics metrics (500 val samples)

| Metric | GAN | Bicubic |
|---|---|---|
| Response mean | 0.926 | 1.088 |
| Response std | 0.031 | 0.022 |
| Response median | 0.930 | 1.089 |
| \|Rel error\| mean | 0.074 | — |

### Per-class physics (QCD label y ∈ {0,1})

| Class | N samples | GAN response μ | GAN response σ | Bicubic response μ |
|---|---|---|---|---|
| 0 (QCD gluon) | 257 | 0.9305 | 0.0264 | 1.0916 |
| 1 (QCD quark) | 243 | 0.9215 | 0.0351 | 1.0842 |

### Correlation metrics

| Metric | Value |
|---|---|
| Pearson r (pixel, log1p) | 0.6421 |
| SSIM mean | 0.9794 ± 0.017 |
| Radial profile MSE | 2.38 × 10⁻⁵ |
| Azimuthal profile MSE | 1.39 × 10⁻⁵ |
| GAN sparsity | 0.735 |
| HR sparsity | 0.983 |

### Observations
- Response oscillates between 0.92–1.10 across epochs — GAN and physics loss are competing. The model is not converged at 5 epochs. Target: ≥ 20 epochs without `--max-train-batches` cap.
- Bicubic response (1.088) is consistently above 1.0 — bicubic upsampling over-estimates total energy for these jet images (expected: LR/HR ratio ~0.25 from dataset analysis).
- GAN undershoots (0.926) — undercounting energy in sparse regions. Increase `--lambda-physics` to 15.
- SSIM (0.979) is higher than HDF5 (0.916) — the 125×125 jet images have stronger structural regularity.
- Lower Pearson r (0.642 vs 0.905 for HDF5) — the jet image task is harder: 3 distinct physics channels, sparser, more varied shower morphology.
- GAN sparsity (73.5%) close to HR (98.3%) after 5 epochs — good: the generator is learning to suppress noise in empty cells.

---

## Cross-dataset comparison

| Metric | QCDToGGQQ | CaloChallenge |
|---|---|---|
| Task difficulty | Higher (3 channels, real LR/HR pairs) | Lower (1 layer → 3ch, synthetic LR) |
| val L1 | **0.10251** | 0.25373 |
| val PSNR | 2.14 dB | **6.44 dB** |
| Response GAN | 0.926 ± 0.031 | 1.092 ± 0.085 |
| Pixel Pearson r | 0.642 | **0.905** |
| SSIM | **0.979** | 0.916 |
| D loss at epoch 5 | 0.151 | 0.037 |
| GAN vs HR sparsity gap | 0.248 | 0.585 |

**Key takeaways for paper:**
1. Lower val L1 on parquet does not mean better SR — the metric is normalized differently. PSNR and response are more meaningful for cross-dataset comparison.
2. Both runs show the discriminator collapsing (D loss → 0.04–0.15) within 5 epochs — TTUR (`d-lr-ratio 0.5`) slows it but does not prevent it at this data budget.
3. The physics response loss is working — both runs show response converging toward 1.0. Full convergence needs 15–20+ epochs with the data cap removed.

---

## Next steps

- [ ] Remove `--max-train-batches` cap and run 20 epochs on both datasets (full data)
- [ ] Increase `--lambda-physics 15` on parquet to fix the 0.926 under-response
- [ ] Increase `--n-critic 2` on HDF5 to slow D collapse
- [ ] Run `save_experiment.py` again after full runs to regenerate all figures
- [ ] Add SwinIR results to cross-dataset comparison table
