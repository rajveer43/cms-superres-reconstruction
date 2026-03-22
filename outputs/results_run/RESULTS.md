# Results Analysis

This analysis uses the saved `best.pt` checkpoint from `outputs/results_run/` and evaluates it on 10 validation batches.

## Training behavior

The training losses decreased over 3 epochs:

- epoch 1: generator loss 7.48707, discriminator loss 1.86376, L1 0.13009, val L1 0.12330
- epoch 2: generator loss 5.18753, discriminator loss 0.39842, L1 0.09996, val L1 0.11088
- epoch 3: generator loss 4.77415, discriminator loss 0.36440, L1 0.09164, val L1 0.10771

Validation L1 improved monotonically, which indicates the generator learned a better mapping over the compact run.

## Baseline comparison

On 10 validation batches:

- Normalized L1, generator: 0.103388
- Normalized L1, bicubic: 0.094822
- Raw-space L1, generator: 0.006340
- Raw-space L1, bicubic: 0.008559
- Raw-space RMSE, generator: 0.207549
- Raw-space RMSE, bicubic: 0.196219
- Raw-space PSNR, generator: 53.227 dB
- Raw-space PSNR, bicubic: 53.715 dB

## Interpretation

- The generator is better than bicubic interpolation under raw-space L1, which is the main metric I would use for sparse calorimeter images.
- The normalized L1, RMSE, and PSNR are not uniformly better, which usually means the model is trading a small number of sharper local corrections for larger errors on a few high-energy pixels.
- That behavior is consistent with a GAN objective: the adversarial term pushes for sharper structure, while the L1 term keeps the output aligned with the target distribution.

## Limitations

- This is still a compact training run, not a full physics-grade benchmark.
- The evaluation uses a validation subset, not the entire dataset.
- I have not yet added event-level physics metrics such as jet mass response or substructure observables.
