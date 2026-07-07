# Classification-Eval Report — 2026-06-28_datasets_64x_local_64x_50ep

- **Scale:** 64x  |  **Config epochs:** 50  |  **Reached epoch:** 50
- **Checkpoint:** `/Users/rajveerrathod/Work/Google_Summer_of_code/task/multiscale_sr/experiments/2026-06-28_datasets_64x_local_64x_50ep/checkpoints/best.pt`
- **Status:** trained
- **Tagger:** fixed HR-trained, 2822 train / 1210 test


## Headline metrics (fixed HR tagger)

| Source | AUC | 1/εB @ 50% |
|---|---|---|
| HR (truth) | 0.698 | 4.26 |
| LR (bicubic) | 0.543 | 2.19 |
| **SR (model)** | **0.696** | **4.69** |

- **Tagging efficiency (AUC_SR/AUC_HR):** 99.7%
- **Recovery (LR→HR gap closed):** 98.6%
- **SR↔HR score agreement:** Pearson r=0.893, mean|Δ|=0.082
- **Calibration ECE:** HR=0.095 / LR=0.255 / SR=0.106

## Interpretation

- SR AUC (0.696) is **above the bicubic LR floor** (0.543) — the model adds taggable information.
- Recovery is **positive** (98.6%): SR closes the LR→HR gap.
- SR fools the HR tagger like HR does (per-sample r=0.893).

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
