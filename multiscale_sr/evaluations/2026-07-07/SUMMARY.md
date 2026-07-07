# Cross-Run Evaluation Summary — 2026-07-07

Evaluated 8 checkpoint(s); 8 succeeded. Same parquet, same downsampling, same fixed HR tagger, seed=42.

## Master table (fixed HR tagger)

| Run | Scale | Cfg ep | Reached | AUC_SR | Efficiency | Recovery | Status |
|---|---|---|---|---|---|---|---|
| 2026-06-23_datasets_16x_baseline | 16x | 5 | 5 | 0.539 | 77.3% | -21.9% | undertrained (recovery<0: SR worse than bicubic) |
| 2026-07-03_datasets_16x_local_16x_cached_5ep | 16x | 5 | 5 | 0.598 | 85.7% | 23.3% | undertrained (few epochs) |
| 2026-07-01_datasets_16x_local_16x_30ep | 16x | 30 | 23 | 0.591 | 84.7% | 17.9% | partial (stopped before configured epochs) |
| 2026-06-23_datasets_32x_baseline | 32x | 5 | 5 | 0.518 | 74.2% | -4.4% | undertrained (recovery<0: SR worse than bicubic) |
| 2026-06-24_datasets_32x_baseline | 32x | 20 | 20 | 0.563 | 80.7% | 21.8% | trained |
| 2026-07-01_datasets_32x_local_32x_30ep | 32x | 30 | 30 | 0.587 | 84.1% | 35.6% | trained |
| 2026-06-23_datasets_64x_baseline | 64x | 20 | 20 | 0.698 | 100.0% | 100.2% | trained |
| 2026-06-28_datasets_64x_local_64x_50ep | 64x | 50 | 50 | 0.696 | 99.7% | 98.6% | trained |

## Best checkpoint per scale (by tagging efficiency)

- **16x:** `2026-07-03_datasets_16x_local_16x_cached_5ep` — efficiency 85.7%, recovery 23.3% (undertrained (few epochs))
- **32x:** `2026-07-01_datasets_32x_local_32x_30ep` — efficiency 84.1%, recovery 35.6% (trained)
- **64x:** `2026-06-23_datasets_64x_baseline` — efficiency 100.0%, recovery 100.2% (trained)

## Sanity: AUC_HR consistency across runs

- AUC_HR range across runs: 0.698–0.698 (spread 0.000) — OK. A fixed HR tagger should give ~identical AUC_HR regardless of the SR model.

## Epoch effect (does more training help?)

- **16x recovery vs epochs:** 5ep:-22% → 5ep:23% → 30ep:18%  → still improving.
- **32x recovery vs epochs:** 5ep:-4% → 20ep:22% → 30ep:36%  → still improving.
- **64x recovery vs epochs:** 20ep:100% → 50ep:99%  → plateaued.

## Bottom line

Best-performing: 2026-06-23_datasets_64x_baseline (100%), 2026-06-28_datasets_64x_local_64x_50ep (99%). Needs retraining (recovery<0): 2026-06-23_datasets_16x_baseline, 2026-06-23_datasets_32x_baseline.
