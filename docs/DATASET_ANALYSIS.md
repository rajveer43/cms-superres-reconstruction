# Dataset Analysis Notes

The sampled dataset analysis gives a concise view of the task before training.

## Observed properties

- Total analyzed events: `2048`
- Class balance:
  - class 0: `1042`
  - class 1: `1006`
- Image shapes:
  - low-resolution: `(3, 64, 64)`
  - high-resolution: `(3, 125, 125)`
- Mean `pt`: `116.8989`
- Mean `m0`: `21.4095`
- Mean LR energy: `61.0800`
- Mean HR energy: `244.3201`
- Mean LR/HR response: `0.2500`
- Mean LR nonzero fraction: `0.0292`
- Mean HR nonzero fraction: `0.0175`

## Why this matters

- The low-resolution image contains about one quarter of the total intensity of the high-resolution target.
- The images are sparse, so a linear pixel-loss alone tends to underfit the structure.
- The class balance is close enough that a single shared model is reasonable.
- The strong sparsity justifies `log1p` compression and a physics-aware loss term.

## Implications for modeling

- The generator should upsample before refinement because the target grid is larger.
- The loss should include both reconstruction and energy-consistency terms.
- Visual quality alone is not enough; the total response must stay close to the target.

## Practical conclusion

The dataset analysis supports the final training design:

- conditional image-to-image GAN
- residual upsampling generator
- PatchGAN discriminator
- L1 reconstruction loss
- physics loss on total intensity
