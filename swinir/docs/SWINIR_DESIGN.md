# SwinIR design notes

This document tracks the *why* behind the architectural choices in
`src/jetsr_swin/models/swinir.py`. Update it whenever you change the model.

## Why SwinIR for HEP jet images

Calorimeter jets have global substructure (prong separation at large angular
distances). A 3×3-kernel CNN has to stack many layers to integrate global
context. Window-attention with shifted windows lets a single block see the
full window of η-φ relationships at O(N) cost, and the shift cycles cover
cross-window dependencies after two blocks. The hypothesis: this yields better
global energy consistency than the residual-CNN baseline.

## Architectural choices

### Patch embedding (stride 1)

We use a stride-1 3×3 Conv → 96 channels. Unlike ViT's stride-P patch embed,
this keeps the token grid at 64×64 (same as input) — every pixel becomes a
token. That preserves the very-sparse calorimeter signal (typical nonzero
fraction ~2–3%) which a stride-P embed would average away.

### Window size 4

The proposal pins window_size=4. Tokens per window = 16; for a 64×64 grid we
have 256 windows. Tiny windows + many windows is cheap and matches the local
clustering of energy deposits.

### η-φ relative positional bias

The learnable bias table is indexed by (Δη, Δφ) within a window. φ wraps at
±π in the detector, but a 4×4 window covers a small angular range so we treat
it as planar. A future cyclic variant can swap in via the same interface.

### Upsample head: ConvT + center crop

64 × 2 = 128, but the HR target is 125×125. We chose ConvTranspose2d(2×) then
center-crop 128→125 (loses 1.5 pixels on each side, ~1.2% of width). Alternatives:
- bilinear/bicubic up + conv: smoother but doesn't learn the upsample kernel
- pixel shuffle: requires more channels and an in-channel rearrangement

Center-crop preserves the field of view at the cost of a tiny boundary loss,
which is acceptable since jets are compact and centered.

### Global skip

`out = swinir(x) + bicubic_up(x)`. This matches the baseline pattern: the
network learns a correction over a strong prior (bicubic), instead of building
the image from scratch. Stabilizes training, improves PSNR on sparse inputs.

### Body conv + residual

After the Swin blocks we have a 3×3 conv with a long residual back to the
patch-embed output. This is the standard SwinIR pattern — gives the high-freq
residual path a chance to refine token features before upsampling.

### FiLM conditioning (Phase 2)

Per Swin block, after LN: `x = (1 + γ(meta)) * x + β(meta)` with γ, β derived
from `(pt, m0, y_onehot)` via a 64→128 MLP. We use `(1 + γ)` instead of `γ` so
that zero-initialized FiLM weights make the layer the identity — the base model
behavior is preserved at training start.

## Loss design

`L = λ_l1 * L1(pred_norm, target_norm) + λ_phys * L1(log1p(Σ pred_raw), log1p(Σ target_raw))`

- **No adversarial term.** Whole point of the ablation: does global attention
  alone close the gap to the GAN?
- λ values mirror the GAN's defaults (50 / 12) for apples-to-apples comparison.

## Compute

- 64×64 grid × 96 channels × 4 blocks × MLP ratio 4 → ~3M parameters
- Single forward pass ~10 ms on a small GPU at batch 8
- AMP enabled on CUDA only; MPS autocast is too experimental to rely on

## Known limitations

- Window size 4 is small; a 4-block depth gives an effective receptive field
  of ~8 tokens. For wide-prong substructure we may need either larger windows
  (8 or 16) or deeper stacks (6–8 blocks).
- The η-φ bias treats φ as Euclidean — true cyclic bias would be more
  physically motivated but for window=4 this is a negligible effect.
- ConvTranspose can introduce checkerboarding; we use the refinement conv
  to suppress it, but worth checking visually each run.
