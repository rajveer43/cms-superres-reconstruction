# Experiment Log

## 2026-05-24 — First full SwinIR runs

### What was built
- Complete `swinir/` package from scratch: data streaming, SwinIR model with η-φ
  relative positional bias, FiLM conditioning, W&B logging with attention maps and
  per-layer activation/gradient hooks, training loop (AdamW + cosine schedule, AMP),
  CLI scripts, configs, unit tests (12/12 passing)
- W&B project: `jetsr-swinir` under `rathodrajveer1311-machine-learning-for-science`

### Runs completed

| Run | W&B | Epochs | Best val L1 | Response |
|---|---|---|---|---|
| swinir_base (no W&B) | — | 20 | 0.0992 | 0.973 |
| swinir_base (with W&B) | [je9j5rk8](https://wandb.ai/rathodrajveer1311-machine-learning-for-science/jetsr-swinir/runs/je9j5rk8) | 20 | 0.0992 | 0.973 |
| swinir_film (with W&B) | [imrc9fvo](https://wandb.ai/rathodrajveer1311-machine-learning-for-science/jetsr-swinir/runs/imrc9fvo) | 20 | 0.0991 | 0.973 |

### Key findings

1. **SwinIR did not beat the GAN** (L1: 0.0989 vs 0.0972). With 5× more training
   data per epoch, this is a meaningful signal, not noise. At this scale (4 blocks,
   window=4, 800 samples/epoch), global attention provides no advantage.

2. **Energy response drifted wrong way** (1.003 → 0.971). The physics loss uses
   `L1(log1p(Σpred), log1p(Σtarget))`. Log compression is too forgiving — the model
   reduces the log-sum difference while the actual response falls below 1.0. GAN's
   response was 1.010. This is the most damaging difference for physics papers.

3. **FiLM gave no improvement** over base (Δval_L1 = 0.0001, Δresponse = 0.0003 —
   both within noise). Likely causes: zero-init gradient signal too weak, or
   hardcoded pt/m0 normalisation constants inaccurate. Need to verify
   `film_layers.0.proj.weight.std()` from the checkpoint.

4. **PSNR is non-comparable** between GAN (14.32 dB, max_val=1 normalized) and
   SwinIR (52.79 dB, max_val=target.max() raw). SSIM saturates at 0.998 due to
   sparse zero background. Neither metric is informative for this dataset without
   careful calibration.

5. **W&B internals**: `qkv.act_norm` grew 1 → 3091 over training. Gradient norms
   are small (0.003–0.014) but non-vanishing. Model is learning — just not converging
   to better physics.

### Issues discovered and fixed during the session

| Issue | Fix | File |
|---|---|---|
| `torch.cuda.amp.GradScaler` deprecated | Use `torch.amp.GradScaler(device_type, ...)` | `training/trainer.py` |
| YAML serialization failed on `torch.__version__` | Added `_sanitize()` to convert all values to plain Python types | `training/trainer.py` |
| W&B silently disabled when package missing | Now raises `RuntimeError` if `wandb_enabled=true` but package absent or API key missing | `logging/wandb_logger.py` |
| `.env` not loaded automatically | Added `load_dotenv_from_repo()` called at top of both CLI scripts | `utils/env.py`, `scripts/train_swinir.py` |
| `evaluate.py` hung silently on full val file | Added `\r` progress line every 20 batches | `scripts/evaluate.py` |
| `einops`, `tqdm` in requirements but unused | Removed from `requirements.txt` and `pyproject.toml` | — |

### Open tasks for next session

- [ ] Fix physics loss: add direct response constraint `λ * |Σpred/Σtarget − 1|`
- [ ] Fix PSNR: add `psnr_norm` metric (max_val=1 on normalized tensors)
- [ ] Add `include_meta: true` to `swinir_base.yaml` so per-class gap is tracked
- [ ] Scale to full training budget (remove `max_train_batches` cap)
- [ ] Verify FiLM activation: inspect `film_layers.0.proj.weight.std()` from checkpoint
- [ ] Run SwinIR+FiLM eval on 12K events (same as base)
