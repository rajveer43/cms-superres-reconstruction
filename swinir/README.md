# SwinIR Super-Resolution for CMS Jet Images

Transformer-based super-resolution generator for CMS calorimeter jet images, built as a
non-adversarial comparative study against the residual-CNN GAN baseline in `../train_srgan.py`.

## What this package implements

- **SwinIR base** (`models/swinir.py`): patch-embed + 4 Swin Transformer blocks with
  shifted-window attention and an η-φ relative positional bias, ConvTranspose upsample
  64→128, center-crop to 125, refinement conv.
- **FiLM conditioning** (`models/layers/film.py`, Phase 2): per-block γ/β modulation
  from `(pt, m0, y)` metadata.
- **W&B logging** (`logging/`): scalars, sample panels, per-layer activation and
  gradient norms via forward/backward hooks, attention-map visualizations, and
  the learned relative-positional-bias matrix.

## Layout

```
swinir/
├── src/jetsr_swin/        # importable package
│   ├── data/              # parquet streaming, normalization, splits
│   ├── models/            # SwinIR + layers
│   ├── losses/            # L1 + physics
│   ├── training/          # trainer, callbacks, seed
│   ├── evaluation/        # metrics, per-class, plots
│   ├── logging/           # wandb logger, hooks, attention viz
│   └── utils/             # io, device
├── configs/               # YAML configs
├── scripts/               # CLI entrypoints
├── notebooks/             # EDA, architecture sanity, results comparison
├── tests/                 # pytest
├── outputs/               # run artifacts (gitignored)
└── reports/               # publication-ready summaries
```

## Setup

```bash
cd swinir
pip install -e ".[dev]"
export WANDB_API_KEY=...        # your key
export WANDB_PROJECT=jetsr-swin
```

## Train

```bash
# Phase 1: SwinIR base (no conditioning, no GAN)
python scripts/train_swinir.py --config configs/swinir_base.yaml

# Phase 2: + FiLM conditioning
python scripts/train_swinir.py --config configs/swinir_film.yaml

# Quick smoke test (no W&B, 1 epoch, tiny budget)
python scripts/train_swinir.py --config configs/swinir_base.yaml --smoke
```

## Evaluate / compare

```bash
python scripts/evaluate.py \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --data-dir ../datasets \
  --out reports/swinir/eval_<run>.json
```

## Device

Auto-detects in order: CUDA → MPS (Apple Silicon) → CPU. AMP is enabled only on CUDA.

## Reproducibility

Every run logs: seed, git SHA, `torch.__version__`, device, dataset file list, full config.
