# Jet Image Super-Resolution GAN

## Problem

Train a generative adversarial network to reconstruct high-resolution calorimeter jet images from low-resolution jet images.

Dataset:

- `X_jets_LR`: low-resolution input
- `X_jets`: high-resolution target
- `pt`, `m0`, `y`: event metadata and class label

## Model

The model follows an upsample-then-refine design because the LR input is smaller than the HR target and the task requires both structure recovery and energy recovery. The final architecture was chosen as the best compromise across reconstruction quality, physics response, and training stability.

- Generator: residual CNN with bicubic upsampling, skip connection, and instance normalization.
- Discriminator: PatchGAN-style CNN with spectral normalization.
- Loss:
  - least-squares adversarial loss
  - L1 reconstruction loss
  - physics loss on total reconstructed energy

## Why this setup

No single loss captures the full problem, so the training objective combines three terms. L1 anchors the output to the target image, the adversarial term sharpens local calorimeter structure, and the physics term keeps total reconstructed energy close to the target. In practice, this combination gave the most balanced checkpoint: better raw-space reconstruction than bicubic, with response close to 1.0.

## Final training run

Configuration:

- `lambda_l1 = 50`
- `lambda_physics = 12`
- `epochs = 18`
- capped batches for a Mac M4 friendly run

## Comparison of runs

| Run | Val L1 | GAN response | Raw L1 | Comment |
| --- | ---: | ---: | ---: | --- |
| Adjusted run | 0.09907 | 1.0217 | 0.00709 | Slight response overshoot |
| Final run | 0.09725 | 1.0098 | 0.00698 | Best balance overall |
| Bicubic baseline | - | 0.9523 | 0.00846 | Smooth but underestimates intensity |

## Final metrics

Best checkpoint: `outputs/final_run/checkpoints/best.pt`

- Validation L1: `0.09725`
- GAN response: `1.0098`
- Bicubic response: `0.9523`
- Raw-space L1: GAN `0.00698` vs bicubic `0.00846`
- Raw-space PSNR: GAN `14.32 dB` vs bicubic `14.13 dB`

## Interpretation

- The final run is the best checkpoint so far.
- It matches total image intensity very closely.
- It improves raw-space reconstruction over bicubic.
- It is slightly worse than bicubic on normalized L1, so the result is not uniformly better across every metric.

## Figures

- Loss curves: `outputs/final_run/plots/loss_curves.png`
- Side-by-side example: `outputs/final_run/plots/epoch_018_side_by_side.png`
- Summary panel: `outputs/final_run/plots/summary_panel.png`

## Dataset behavior

- The low-resolution inputs are materially smaller than the targets, so upsampling is required before refinement.
- Event classes (`y`) do not show a large response gap in the final run.
- The physics response is close to 1.0 for both classes:
  - class 0: `1.0143`
  - class 1: `1.0052`

## Conclusion

The final run is the best presentation checkpoint:

- stable training
- response close to target
- improved raw-space reconstruction
- clear visual improvement over low-resolution input

Main limitation:

- normalized L1 is still slightly worse than bicubic, so the model is not a universal win across all metrics.
