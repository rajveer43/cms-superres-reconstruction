# Classification-Based Evaluation

Pixel metrics (PSNR/SSIM/L1) answer *"does it look right?"*. This pipeline answers
the question that matters for physics: **does super-resolution preserve the
class-discriminative information a jet tagger needs?** A jet tagger (small CNN) is
trained on the real class label, then evaluated on three image sources at HR
resolution — HR (ground truth, the ceiling), LR (bicubic upsample, the floor), and
SR (generator output, the recovery).

## Headline numbers (scale=64x, n_test=1210)

| source | AUC (fixed HR tagger) |
|--------|----------------------:|
| HR     | 0.6980 |
| LR     | 0.5433 |
| SR     | 0.6982 |

- **Tagging efficiency (AUC_SR / AUC_HR) = 100.0%** — how close SR taggability
  gets to the HR ceiling. Close to 100% means SR images are essentially as
  taggable as real HR.
- **LR→HR gap recovered = 100.2%** — of the taggability lost by downscaling,
  how much SR restores. 0% = no better than LR; 100% = fully restored to HR.
- **Background rejection 1/eps_B @ 50% signal eff** —
  HR=4.3, LR=2.2,
  SR=4.5. The HEP-standard operating point.

## Figures

| file | what it shows |
|------|---------------|
| `roc_overlay.png` | ROC curves for HR/LR/SR. SR sitting near HR (above LR) is the win condition. Left: one fixed HR-trained tagger on all sources. Right: an independent tagger per source. |
| `auc_summary_bar.png` | The three AUCs as bars with the efficiency / recovery headline. |
| `score_distributions.png` | Tagger score histograms split by true class. Wider class separation = more taggable. SR should resemble HR, not LR. |
| `confusion_matrices.png` | Confusion matrix + accuracy/F1 at the Youden-J optimal threshold, per source. |
| `efficiency_vs_threshold.png` | Left: signal/background efficiency vs threshold. Right: background rejection vs signal efficiency (log scale) — the curve physicists read, marked at 50%. |
| `calibration.png` | Reliability curves + ECE: does the tagger's confidence stay calibrated on SR images, or does SR shift it? |
| `score_agreement.png` | Per-sample scatter of the HR tagger's score on SR (and LR) vs its score on HR. Points on the diagonal mean SR fools the tagger *the same way* HR does — a stricter test than matching aggregate AUC. |

## Method notes

- **Fixed-HR tagger (primary):** one tagger trained on HR, evaluated frozen on all
  three sources. Answers *"do SR images look like HR to a tagger trained on real
  data?"* This is the headline.
- **Per-source taggers (secondary):** an independent tagger trained on each source,
  measuring the class information each source contains regardless of HR
  compatibility.
- All sources share HR spatial size (LR is bicubic-upsampled) so one tagger
  architecture consumes any of them and comparisons are fair.
- Parquet only — the HDF5/CaloChallenge data has no class label.
