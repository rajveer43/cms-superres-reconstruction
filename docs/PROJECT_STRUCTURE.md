# Project Structure

This repo is organized as an end-to-end workflow:

```text
task/
  datasets/                 # parquet inputs
  docs/                     # project notes and structure
  outputs/                  # experiment runs and checkpoints
  reports/                  # publication-ready summaries
  train_srgan.py            # GAN training entry point
  analyze_dataset.py        # dataset profiling
  analyze_results.py        # training-run analysis
  Makefile                  # one-command workflow shortcuts
  README.md                 # project landing page
  PRESENTATION.md           # slide-friendly summary
```

## Layering

### 1. Data layer

- `datasets/` contains the parquet files with paired low-resolution and high-resolution jet images.
- The dataset analysis script summarizes class balance, sparsity, image size, and energy response.

### 2. Experiment layer

- `outputs/<run_name>/` stores checkpoints, metrics, generated samples, and plots for each run.
- The current best run is `outputs/final_run/`.

### 3. Reporting layer

- `reports/` stores the publication-facing material.
- `reports/dataset_analysis_sample/` contains the dataset summary used in the presentation.
- `reports/PRESENTATION.md` mirrors the slide-ready project summary.

### 4. Code layer

- `train_srgan.py` trains the model.
- `analyze_dataset.py` profiles the raw data.
- `analyze_results.py` turns a completed run into plots, tables, and physics-proxy metrics.

## Workflow

1. Run dataset analysis.
2. Train the model.
3. Analyze the best checkpoint.
4. Build the presentation from the final results.

## Why this layout works

- It keeps raw data separate from generated artifacts.
- It makes the best run obvious.
- It supports a presentation without mixing source code and outputs.
- It keeps the repo easy to extend with notebooks, additional baselines, or ablation studies later.
