# Multi-Scale Super-Resolution (Option A)

Independent GAN per downsampling scale. The high-resolution (HR) calorimeter
image is the fixed ground truth; the low-resolution (LR) input is produced by
**area-downsampling HR** to a target scale (e.g. 64, 32, 16). A separate model
is trained for each scale to quantify how reconstruction quality and physics
fidelity (energy response) degrade as input resolution drops.

This reuses the SRGAN bicubic-residual architecture unchanged — only the input
resolution differs between runs.

## Layout

```
multiscale_sr/
├── multiscale_sr/            # package
│   ├── models/               # Generator, Discriminator (same as SRGAN)
│   ├── data/                 # parquet + HDF5 loaders, multi-scale collate
│   ├── engine.py             # losses, metrics, eval, figures
│   ├── experiment.py         # run-dir layout + YAML config
│   └── wandb_logger.py       # W&B wrapper (graceful no-op if disabled)
├── train.py                  # training entrypoint
├── evaluate.py               # standalone eval for a checkpoint
├── configs/                  # scale_64 / scale_32 / scale_16 YAML
└── experiments/              # run outputs (gitignored)
```

Each run is written to:

```
experiments/{YYYY-MM-DD}_{dataset}_{scale}x_{run_name}/
├── checkpoints/   best.pt, latest.pt
├── figures/       sample_epoch_{N}.png, metrics.png
├── config.yaml    frozen run config
├── eval.json      final metrics
└── metrics.jsonl  per-epoch log
```

## Datasets

Both formats are auto-detected from `--data-dir`:

- **Parquet** (CMS jets): `*.parquet` with `X_jets_LR`, `X_jets`, `pt`, `m0`, `y`.
  At scale 64 you may pass `--use-native-lr` to feed the detector LR instead of
  a downsampled HR.
- **HDF5** (CaloChallenge Dataset 2): `dataset_2_*.hdf5`. HR is resized to a
  square `--hr-size` (default 125) so both datasets share one HR resolution.

Normalization (`log1p` + channel-wise z-score) statistics are computed once on
**HR** and cached to `normalization.json` in the run dir — HR is the common
reference scale for every model regardless of input resolution.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add WANDB_API_KEY (optional)
```

## Train

```bash
# Parquet, scale 32, via config:
python train.py --config configs/scale_32.yaml --data-dir ../datasets --run-name baseline

# HDF5, scale 16, pure CLI:
python train.py --data-dir ../datasets/calochallenge_dataset2 --scale 16 --epochs 50

# Quick smoke test (tiny, no W&B):
python train.py --data-dir ../datasets --scale 32 --epochs 1 \
    --max-train-batches 3 --max-val-batches 2 --max-stats-batches 2 --no-wandb
```

CLI flags override values from `--config`.

## Evaluate

```bash
python evaluate.py --checkpoint experiments/<run>/checkpoints/best.pt \
    --data-dir ../datasets --save-grid /tmp/grid.png
```

## Classification-based evaluation (tagging efficiency)

Pixel metrics (PSNR/SSIM/L1) measure whether SR *looks* right. They say nothing
about whether SR preserves the **physics-classification** information a downstream
jet tagger needs. `classification_eval.py` makes taggability the headline metric:
it trains a jet tagger and measures how taggable HR / LR / SR images are.

```bash
python classification_eval.py --checkpoint experiments/<run>/checkpoints/best.pt \
    --data-dir ../datasets
```

Headline = **tagging efficiency = AUC_SR / AUC_HR** (fixed HR-trained tagger).
Parquet only — HDF5/CaloChallenge has no class label.

Writes to `experiments/<run>/figures/classification/`:

- `roc_overlay.png` — ROC for HR/LR/SR (fixed HR tagger + per-source taggers)
- `auc_summary_bar.png` — AUC bars with the efficiency / recovery headline
- `score_distributions.png` — tagger score histograms per class, per source
- `confusion_matrices.png` — confusion at the Youden-J optimal threshold
- `efficiency_vs_threshold.png` — signal/bkg efficiency + background rejection vs
  signal efficiency (the HEP ROC50 view, `1/eps_B` at 50% signal eff)
- `calibration.png` — reliability curves + ECE per source
- `score_agreement.png` — per-sample HR-vs-SR score scatter (does SR fool the HR
  tagger the same way HR does?)
- `classification_eval.json` — every metric, machine-readable
- `EXPLANATION.md` — what each figure/metric means

`tag_efficiency.py` is the lighter sibling (ROC + AUC bar only); use
`classification_eval.py` for the full diagnostic suite.

## Metrics

- `val_l1` — L1 on normalized tensors (primary selection metric)
- `val_psnr_norm` — PSNR with fixed `max_val=1.0` (comparable across scales)
- `val_energy_response` — `sum(E_pred)/sum(E_true)` on denormalized energy; 1.0 is unbiased
- **tagging efficiency** — `AUC_SR / AUC_HR` from `classification_eval.py` (above)

## Loss

```
L_G   = L_adv + 50·L_l1 + 10·L_phys
L_adv = 0.5·mean[(D(fake) − 1)²]                  (LSGAN)
L_D   = 0.5·mean[(D(real) − 0.9)² + D(fake)²]
L_l1  = mean|G(lr) − hr|                           (normalized)
L_phys= mean|sum(E_pred)/sum(E_true) − 1|         (denormalized)
```
