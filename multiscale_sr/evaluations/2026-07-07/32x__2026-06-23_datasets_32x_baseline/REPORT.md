# Classification-Eval Report — 2026-06-23_datasets_32x_baseline

- **Scale:** 32x  |  **Config epochs:** 5  |  **Reached epoch:** 5
- **Checkpoint:** `/Users/rajveerrathod/Work/Google_Summer_of_code/task/multiscale_sr/experiments/2026-06-23_datasets_32x_baseline/checkpoints/best.pt`
- **Status:** undertrained (recovery<0: SR worse than bicubic)
- **Tagger:** fixed HR-trained, 2822 train / 1210 test


## Headline metrics (fixed HR tagger)

| Source | AUC | 1/εB @ 50% |
|---|---|---|
| HR (truth) | 0.698 | 4.26 |
| LR (bicubic) | 0.526 | 2.18 |
| **SR (model)** | **0.518** | **2.14** |

- **Tagging efficiency (AUC_SR/AUC_HR):** 74.2%
- **Recovery (LR→HR gap closed):** -4.4%
- **SR↔HR score agreement:** Pearson r=0.074, mean|Δ|=0.236
- **Calibration ECE:** HR=0.095 / LR=0.227 / SR=0.224

## Interpretation

- ⚠️ SR AUC (0.518) is **below the bicubic LR floor** (0.526) — SR degrades taggability. Undertrained / not converged.
- ⚠️ Recovery is **negative** (-4.4%): needs more epochs, not a model fault.
- Per-sample SR↔HR agreement is weak (r=0.074).

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
