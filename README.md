# Jet Image Super-Resolution GAN

End-to-end super-resolution for calorimeter jet images.

The project learns a mapping from low-resolution jet images to high-resolution jet images using a conditional GAN with an explicit physics-aware loss term.

Dataset columns:

- `X_jets_LR`: low-resolution calorimeter image
- `X_jets`: high-resolution target image
- `pt`, `m0`, `y`: event metadata and class label

## End-to-end workflow

1. Inspect the dataset and generate summary statistics:
   ```bash
   make dataset-analysis
   ```
2. Train the final model:
   ```bash
   make train-final
   ```
3. Generate plots, physics metrics, and the run report:
   ```bash
   make analyze-final
   ```
4. Copy the presentation summary into the reports folder:
   ```bash
   make presentation
   ```

For a single-reference presentation of the best result, use:

- `outputs/final_run/ANALYSIS.md`
- `outputs/final_run/plots/summary_panel.png`
- `reports/dataset_analysis_sample/DATASET_ANALYSIS.md`
- `PRESENTATION.md`

## Recommended repository structure

See `docs/PROJECT_STRUCTURE.md` for the full layout. The repo is organized around four layers:

- `datasets/` for the parquet inputs
- `outputs/` for experiment runs
- `reports/` for publication-ready summaries
- `docs/` for project-level notes, data analysis, and structure

For a direct comparison of every major run and the reasoning behind the final checkpoint, see `docs/OUTPUT_COMPARISON.md`.

## What the dataset looks like

The sample dataset analysis shows:

- class balance is close to even: `1042` vs `1006`
- low-resolution images are sparse and smaller than the high-resolution targets
- the mean LR/HR intensity ratio is about `0.25`

That means the LR input contains roughly one quarter of the total target intensity, so the model must both upsample and recover missing energy.

## Best result so far

The final run is the current presentation checkpoint.

| Run | Val L1 | GAN response | Raw L1 | Raw PSNR | Comment |
| --- | ---: | ---: | ---: | ---: | --- |
| Final run | 0.09725 | 1.0098 | 0.00698 | 14.32 dB | Best balance |
| Adjusted run | 0.09907 | 1.0217 | 0.00709 | 14.29 dB | Slight response overshoot |
| Bicubic baseline | - | 0.9523 | 0.00846 | 14.13 dB | Smooth but under-energized |

Interpretation:

- the GAN improves raw-space reconstruction over bicubic
- the physics response is close to 1.0
- the result is strong enough for a progress presentation
- normalized L1 is still not a universal win, so the limitations should be stated clearly

## Model choice

The LR tensors are smaller than the HR targets, so the model is built as an "upsample then refine" pipeline rather than a pure image translator. That matches the task: the network must recover fine structure while also restoring missing energy. The final architecture was chosen because it was the best compromise across reconstruction quality, physics response, and training stability.

- **Generator**: residual CNN with bicubic upsampling, instance normalization, and a global skip connection.
  - The skip connection lets the model learn corrections instead of recreating the full image from scratch.
  - Instance normalization works well on sparse calorimeter images.
- **Discriminator**: PatchGAN-style CNN with spectral normalization.
  - Patch-based discrimination focuses on local deposit structure.
  - Spectral normalization stabilizes adversarial training.

## Optimization choices

Training uses a hybrid objective because no single loss captures the full problem. L1 anchors the output to the target image, the adversarial term improves local sharpness, and the physics term keeps the reconstructed total intensity close to the target. In practice, this combination gave the most balanced checkpoint: better raw-space reconstruction than bicubic, with response close to 1.0.

- **Loss**: least-squares adversarial loss + L1 reconstruction loss + physics loss.
  - L1 keeps the output close to the target image.
  - The adversarial term sharpens local structure.
  - The physics loss keeps total reconstructed intensity close to the target.
- **Normalization**:
  - `log1p` compresses the heavy energy tail.
  - channel-wise normalization is computed from the training split only.
- **Split strategy**:
  - parquet files are split at file level to reduce leakage.
- **Optimizer**:
  - Adam with `lr=2e-4`, `betas=(0.5, 0.999)`.

## Reproduce the best run

```bash
python train_srgan.py \
  --data-dir datasets \
  --output-dir outputs/final_run \
  --epochs 18 \
  --batch-size 8 \
  --stats-batch-size 16 \
  --max-stats-batches 10 \
  --max-train-batches 20 \
  --max-val-batches 5 \
  --lambda-l1 50 \
  --lambda-physics 12

python analyze_results.py --run-dir outputs/final_run --data-dir datasets
```

## Files of interest

- `train_srgan.py`: training loop, generator, discriminator, checkpointing
- `analyze_dataset.py`: dataset profiling and publication-ready summary
- `analyze_results.py`: run analysis, plots, and physics-proxy metrics
- `Makefile`: one-command shortcuts for the full workflow
- `docs/`: repository structure and dataset notes
- `reports/`: publication layer for the presentation assets

## Notes

- The scripts use `pyarrow` and stream parquet batches; the full dataset is not loaded into memory.
- Validation reports normalized L1; saved samples and physics metrics are denormalized back to physical scale.
- The final checkpoint is `outputs/final_run/checkpoints/best.pt`.
