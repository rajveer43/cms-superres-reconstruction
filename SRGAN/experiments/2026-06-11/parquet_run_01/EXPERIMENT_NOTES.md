# parquet_run_01 — Experiment Notes

**Date:** 2026-06-11  
**Dataset:** QCDToGGQQ CMS Jet Images (Parquet)  
**Status:** Complete — 5 epochs, not converged, baseline established

## What this run is

First GAN baseline on the native CMS jet image dataset (the primary dataset for this GSoC project). Uses real LR/HR pairs from the detector simulation — no synthetic downsampling. Purpose: match the previous `final_run` / `long_run` experiments but with the new production-grade pipeline.

## Comparison with previous best run (`long_run`)

The previous best result (logged in project memory) was `long_run` val_l1 = 0.0828, run without physics loss, still improving at epoch 6.

This run (`parquet_run_01`) reached val_l1 = 0.1025 at epoch 5 — higher L1 than `long_run` because:
1. Physics loss (`λ_phys=10`) adds regularization that slightly increases L1
2. Only 80 batches/epoch vs full data in `long_run` — severely undertrained
3. Different normalization (stats computed fresh, cache = `normalization.json`)

Target for full run: val_l1 < 0.09 with physics response within 2% of 1.0.

## Per-class analysis (novel vs prior work)

The per-class physics breakdown is new — prior runs did not split by QCD label:

| Class | Description | GAN response |
|---|---|---|
| 0 | QCD gluon jet | 0.9305 ± 0.026 |
| 1 | QCD quark jet | 0.9215 ± 0.035 |

Both classes under-respond (< 1.0). Quark jets (class 1) show slightly more variance — consistent with their broader shower morphology. This per-class breakdown should appear in the paper.

## What worked

- Parquet streaming with reservoir shuffle — no RAM spikes observed
- Stats cache hit on val loader (computed once, reused)
- Per-class response, pt/m0 scatter plots generated cleanly
- Channel correlation heatmap shows ECAL/HCAL/Tracks inter-channel structure

## What needs fixing

| Issue | Symptom | Fix |
|---|---|---|
| Under-response | 0.926 (should be 1.00) | `--lambda-physics 15` |
| Not converged | val_l1 still falling at epoch 5 | Full data, 20 epochs |
| D losing signal | D loss 0.471 → 0.151 | Acceptable — not collapsed yet |
| GAN sparsity gap | GAN 73.5% vs HR 98.3% | Physics loss helps; more epochs needed |

## Suggested next run command

```bash
python train_srgan.py \
    --dataset-type parquet \
    --dataset-path ../datasets \
    --epochs 20 \
    --batch-size 32 \
    --lambda-l1 50 --lambda-physics 15 \
    --hr-size 125 125 \
    --output-dir experiments/2026-06-11/parquet_run_02
```

## Figures generated

22 figures in `figures/` — 4 more than the HDF5 run because parquet has real `pt`, `m0`, `y` columns.

Extra parquet-only figures:
- `figures/physics/response_by_class.png` — GAN response split by QCD label
- `figures/physics/response_vs_pt.png` — response as a function of jet pT
- `figures/physics/response_vs_m0.png` — response as a function of jet mass
- `figures/correlations/channel_correlation_heatmap.png` — ECAL/HCAL/Tracks Pearson r matrix

Key figures to check first:
- `figures/physics/response_by_class.png` — for paper: class-level physics fidelity
- `figures/reconstruction/mean_shower_comparison.png` — per-channel mean shower (3 rows)
- `figures/correlations/channel_correlation_heatmap.png` — inter-channel structure preserved?
