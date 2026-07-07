# Classification-Eval Report — 2026-06-24_datasets_32x_baseline

- **Scale:** 32x  |  **Config epochs:** 20  |  **Reached epoch:** 20
- **Checkpoint:** `/Users/rajveerrathod/Work/Google_Summer_of_code/task/multiscale_sr/experiments/2026-06-24_datasets_32x_baseline/checkpoints/best.pt`
- **Status:** trained
- **Tagger:** fixed HR-trained, 2822 train / 1210 test


## Headline metrics (fixed HR tagger)

| Source | AUC | 1/εB @ 50% |
|---|---|---|
| HR (truth) | 0.698 | 4.26 |
| LR (bicubic) | 0.526 | 2.18 |
| **SR (model)** | **0.563** | **2.54** |

- **Tagging efficiency (AUC_SR/AUC_HR):** 80.7%
- **Recovery (LR→HR gap closed):** 21.8%
- **SR↔HR score agreement:** Pearson r=0.268, mean|Δ|=0.215
- **Calibration ECE:** HR=0.095 / LR=0.227 / SR=0.211

## Interpretation

- SR AUC (0.563) is **above the bicubic LR floor** (0.526) — the model adds taggable information.
- Recovery is **positive** (21.8%): SR closes the LR→HR gap.
- Per-sample SR↔HR agreement is weak (r=0.268).

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
