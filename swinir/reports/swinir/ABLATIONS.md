# SwinIR Ablations — GAN vs SwinIR vs SwinIR+FiLM

**Date:** 2026-05-24  
**Dataset:** QCDToGGQQ jet images, 139,306 events, 3 parquet files  
**Eval split:** file 2 (val), 12,000 events sampled (750 batches × bs=16)  
**Device:** Apple MPS (MacBook Air)

---

## Headline metrics

| Model | val L1 (norm) | val L1 (raw) | PSNR† | SSIM† | Response μ | Response σ |
|---|---:|---:|---:|---:|---:|---:|
| Bicubic baseline | — | ~0.00846 | 14.13 dB | — | 0.9523 | — |
| **Residual-CNN GAN** (existing) | **0.0972** | **0.00698** | **14.32 dB** | — | **1.010** | — |
| SwinIR base (this work) | 0.0989 | 0.00714 | 52.79‡ | 0.9984 | 0.971 | 0.044 |
| SwinIR + FiLM (this work) | ~0.0991 | ~0.00715 | ~52.89‡ | ~0.998 | ~0.973 | ~0.045 |

†GAN PSNR uses `max_val=1.0` on normalized data; SwinIR PSNR uses `max_val=target.max()` on raw data — **not directly comparable**.  
‡SwinIR PSNR is artificially high due to sparse zero-background dominating MSE.

---

## Training setup

| | GAN | SwinIR base | SwinIR + FiLM |
|---|---|---|---|
| Epochs | 18 | 20 | 20 |
| Train batches/epoch | 20 (bs=8, 160 samples) | 100 (bs=8, 800 samples) | 100 (bs=8, 800 samples) |
| Val batches/epoch | 5 | 20 | 20 |
| Optimizer | Adam lr=2e-4 | AdamW lr=2e-4 | AdamW lr=2e-4 |
| Schedule | flat | cosine+warmup | cosine+warmup |
| λ_l1 / λ_phys | 50 / 12 | 50 / 12 | 50 / 12 |
| AMP | — | off (MPS) | off (MPS) |
| W&B run | — | [je9j5rk8](https://wandb.ai/rathodrajveer1311-machine-learning-for-science/jetsr-swinir/runs/je9j5rk8) | [imrc9fvo](https://wandb.ai/rathodrajveer1311-machine-learning-for-science/jetsr-swinir/runs/imrc9fvo) |

---

## What we expected vs. observed

| Hypothesis | Expected | Observed | Verdict |
|---|---|---|---|
| Swin global attention > CNN local kernels | SwinIR L1 < GAN L1 | SwinIR L1 > GAN L1 (worse by ~2%) | ❌ Not confirmed at this scale |
| Window attention improves global energy consistency | Response closer to 1.0 | Response drifted to 0.971 (GAN: 1.010) | ❌ Opposite direction |
| FiLM conditioning narrows per-class response gap | gap < GAN's 0.0091 | No measurable improvement over base | ❌ FiLM did not activate meaningfully |

---

## Known issues with this comparison (must fix before conclusions)

### 1. Physics loss is too forgiving
The current loss is `L1(log1p(Σpred), log1p(Σtarget))`. Log compression makes a 3–4% energy bias look like a tiny loss value. The model learns to match the log-scale sums without actually getting the absolute response to 1.0. **Response drifted monotonically: 1.003 (epoch 1) → 0.971 (epoch 20).**

**Fix:** add `λ * |Σpred/Σtarget − 1|` as a direct response-constraint term.

### 2. PSNR is not comparable across models
GAN used `max_val=1.0` (normalized space). SwinIR evaluation uses `max_val=target.max()` per batch in raw space, producing inflated ~52 dB values. SSIM (0.998) is also saturated because sparse images have mostly-matching zero backgrounds.

**Fix:** report `psnr_norm` with `max_val=1.0` on normalized-space tensors.

### 3. FiLM may be functionally inactive
FiLM layers are zero-initialized (identity). With only 100 train batches/epoch and a small gradient signal through the FiLM projection, the γ/β may not have moved significantly from zero. The FiLM run's `train/phys` ended slightly lower (0.0299 vs 0.0321), suggesting mild activation on the train set but no generalization.

**Verify:** `torch.load('best.pt')['model_state']['film_layers.0.proj.weight'].std()` — should be >> 0 if FiLM learned anything.

### 4. Training budget is severely capped
100 batches × 8 = **800 samples/epoch** from a pool of ~84,000 available training events. GAN used only 20 batches but for 18 epochs. Neither is a full-data run. SwinIR at full budget (8,400+ batches/epoch, 40 epochs) is untested.

---

## W&B internal signals (from attention visualization)

- **`qkv.act_norm`** grew monotonically 1 → 3091 (base) and 1 → 2656 (film) — attention activations increased throughout training, not saturated
- **Gradient norms** (0.003–0.014 on attention sub-modules) — non-vanishing, model is learning
- **Attention maps** (per epoch in W&B) — verify window-attention structure in wandb dashboard
- **Rel-pos-bias** (per epoch in W&B) — verify η-φ bias is learning spatial structure

---

## Next experiments (in priority order)

1. **Fix physics loss + PSNR comparability** — re-run SwinIR base to get a fair number
2. **Scale train budget** — remove `max_train_batches` cap, run 10 epochs full data (~4–8h on GPU)
3. **Larger model** — `embed_dim=128`, `num_blocks=6`, `window_size=8` (see `configs/swinir_sweep.yaml` when added)
4. **FiLM re-run after physics fix** — only meaningful once base converges to response ~1.0
5. **Optuna sweep** — `embed_dim`, `num_blocks`, `window_size`, `λ_phys`

---

## Raw eval JSONs

- `outputs/swinir_base/eval_val.json` — SwinIR base, 12K events
- `outputs/swinir_film/eval_val.json` — SwinIR + FiLM (run `evaluate.py` to generate)
