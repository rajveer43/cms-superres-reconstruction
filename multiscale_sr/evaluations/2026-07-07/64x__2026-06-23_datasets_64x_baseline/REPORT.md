# Classification-Eval Report — 2026-06-23_datasets_64x_baseline

- **Scale:** 64x  |  **Config epochs:** 20  |  **Reached epoch:** 20
- **Checkpoint:** `/Users/rajveerrathod/Work/Google_Summer_of_code/task/multiscale_sr/experiments/2026-06-23_datasets_64x_baseline/checkpoints/best.pt`
- **Status:** trained
- **Tagger:** fixed HR-trained, 2822 train / 1210 test


## Headline metrics (fixed HR tagger)

| Source | AUC | 1/εB @ 50% |
|---|---|---|
| HR (truth) | 0.698 | 4.26 |
| LR (bicubic) | 0.543 | 2.19 |
| **SR (model)** | **0.698** | **4.55** |

- **Tagging efficiency (AUC_SR/AUC_HR):** 100.0%
- **Recovery (LR→HR gap closed):** 100.2%
- **SR↔HR score agreement:** Pearson r=0.845, mean|Δ|=0.100
- **Calibration ECE:** HR=0.095 / LR=0.255 / SR=0.086

## Interpretation

- SR AUC (0.698) is **above the bicubic LR floor** (0.543) — the model adds taggable information.
- Recovery is **positive** (100.2%): SR closes the LR→HR gap.
- SR fools the HR tagger like HR does (per-sample r=0.845).

## Figures

- [roc_overlay.png](classification/roc_overlay.png)
- [auc_summary_bar.png](classification/auc_summary_bar.png)
- [score_distributions.png](classification/score_distributions.png)
- [confusion_matrices.png](classification/confusion_matrices.png)
- [efficiency_vs_threshold.png](classification/efficiency_vs_threshold.png)
- [calibration.png](classification/calibration.png)
- [score_agreement.png](classification/score_agreement.png)
- [energy_correlation.png](classification/energy_correlation.png)
- [pt_correlation.png](classification/pt_correlation.png)

## Caveat
Absolute AUC is capped by the small tagger (few epochs, ~4k samples); the **efficiency ratio** and **recovery** are the headline, not absolute AUC.
