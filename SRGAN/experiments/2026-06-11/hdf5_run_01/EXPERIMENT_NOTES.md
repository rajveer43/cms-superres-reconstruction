# hdf5_run_01 — Experiment Notes

**Date:** 2026-06-11  
**Dataset:** CaloChallenge Dataset 2 (HDF5)  
**Status:** Complete — 5 epochs, not converged, baseline established

## What this run is

First GAN baseline on the CaloChallenge Dataset 2. Purpose: verify the HDF5 data pipeline works end-to-end and establish a starting point for response and SSIM metrics to compare against future longer runs and against SwinIR.

## What worked

- HDF5 lazy loading (SWMR + per-PID handle re-open) worked correctly on macOS MPS
- Normalization stats computed from training split only, cached — no data leakage
- Physics response improved every epoch (0.815 → 1.091)
- Pixel Pearson r = 0.905 after only 5 epochs — strong spatial agreement

## What needs fixing for next run

| Issue | Symptom | Fix |
|---|---|---|
| D collapse | D loss 0.28 → 0.037 in 5 epochs | `--n-critic 2` |
| Response overshoot | 1.09 (should be 1.00) | `--lambda-physics 15` |
| GAN fills sparse cells | GAN sparsity 17% vs HR 75% | Add log-domain L1 or sparsity penalty |
| Not converged | val_l1 still falling at epoch 5 | Remove `--max-train-batches`, run 20 epochs |

## Suggested next run command

```bash
python train_srgan.py \
    --dataset-type hdf5 \
    --dataset-path ../datasets/calochallenge_dataset2 \
    --epochs 20 \
    --batch-size 32 \
    --n-critic 2 \
    --lambda-l1 50 --lambda-physics 15 \
    --hr-size 45 144 --scale-factor 2.0 \
    --output-dir experiments/2026-06-11/hdf5_run_02
```

## Figures generated

18 figures in `figures/` — see `EXPERIMENT_REPORT.md` for full list with inline images.

Key figures to check first:
- `figures/training/physics_response_curve.png` — response oscillation pattern
- `figures/reconstruction/sample_grid_0.png` — visual quality after 5 epochs
- `figures/correlations/radial_profile.png` — GAN vs HR radial energy profile
- `figures/physics/energy_response_hist.png` — response distribution shape
