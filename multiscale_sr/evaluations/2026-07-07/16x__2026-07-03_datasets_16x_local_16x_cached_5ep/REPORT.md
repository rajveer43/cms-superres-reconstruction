# Classification-Eval Report — 2026-07-03_datasets_16x_local_16x_cached_5ep

- **Scale:** 16x  |  **Config epochs:** 5  |  **Reached epoch:** 5
- **Checkpoint:** `/Users/rajveerrathod/Work/Google_Summer_of_code/task/multiscale_sr/experiments/2026-07-03_datasets_16x_local_16x_cached_5ep/checkpoints/best.pt`
- **Status:** undertrained (few epochs)
- **Tagger:** fixed HR-trained, 2822 train / 1210 test


## Headline metrics (fixed HR tagger)

| Source | AUC | 1/εB @ 50% |
|---|---|---|
| HR (truth) | 0.698 | 4.26 |
| LR (bicubic) | 0.568 | 2.73 |
| **SR (model)** | **0.598** | **3.34** |

- **Tagging efficiency (AUC_SR/AUC_HR):** 85.7%
- **Recovery (LR→HR gap closed):** 23.3%
- **SR↔HR score agreement:** Pearson r=0.246, mean|Δ|=0.212
- **Calibration ECE:** HR=0.095 / LR=0.212 / SR=0.122

## Interpretation

- SR AUC (0.598) is **above the bicubic LR floor** (0.568) — the model adds taggable information.
- Recovery is **positive** (23.3%): SR closes the LR→HR gap.
- Per-sample SR↔HR agreement is weak (r=0.246).

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
