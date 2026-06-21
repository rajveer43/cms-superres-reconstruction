# Task List — CMS Jet SR / SRGAN

**Last updated:** 2026-06-11  
**Context:** Both baseline runs (hdf5_run_01, parquet_run_01) confirmed the pipeline works but only saw ~3% of data per epoch over 5 epochs. All tasks below are needed before paper-quality results.

---

## PRIORITY 1 — Full training (blocker for everything else)

- [ ] **P1.1 — Run full parquet training on GPU (Kaggle/Colab)**
  - Remove `--max-train-batches`, run 20 epochs, all 83K train events/epoch
  - Command: `python train_srgan.py --dataset-type parquet --dataset-path ../datasets --epochs 20 --batch-size 64 --lambda-l1 50 --lambda-physics 15 --output-dir experiments/full/parquet_run_01`
  - Expected time: ~1–2 hrs on T4/P100
  - Target: val_l1 < 0.09, response within ±2% of 1.0

- [ ] **P1.2 — Run full HDF5 training on GPU (Kaggle/Colab)**
  - Remove `--max-train-batches`, run 20 epochs, all 134K train showers/epoch
  - Command: `python train_srgan.py --dataset-type hdf5 --dataset-path ../datasets/calochallenge_dataset2 --epochs 20 --batch-size 64 --n-critic 2 --lambda-l1 50 --lambda-physics 15 --hr-size 45 144 --output-dir experiments/full/hdf5_run_01`
  - Expected time: ~1–2 hrs on T4/P100
  - Target: response within ±5% of 1.0, Pearson r > 0.95

- [ ] **P1.3 — Run save_experiment.py on full runs**
  - `python save_experiment.py --run-dir experiments/full/parquet_run_01 --dataset-path ../datasets --dataset-type parquet --hr-size 125 125`
  - `python save_experiment.py --run-dir experiments/full/hdf5_run_01 --dataset-path ../datasets/calochallenge_dataset2 --dataset-type hdf5 --hr-size 45 144`

---

## PRIORITY 2 — Physics-aware loss functions

Current loss: `G = adv + 50*L1 + 10*|ΣE_pred/ΣE_true - 1|`  
Problem: response still 0.926 on parquet (7.4% energy under-count) after 5 epochs.

- [ ] **P2.1 — Wasserstein distance on energy spectra (EMD loss)**
  - Add `λ_emd * EMD(hist(E_pred), hist(E_true))` where hist is a soft 50-bin histogram
  - Penalises shape mismatch in the energy distribution, not just the mean
  - Implement in `srgan/losses/emd_loss.py` using differentiable 1D Wasserstein (cumsum trick)

- [ ] **P2.2 — Sparsity-aware L1 (weighted by HR occupancy)**
  - Current L1 weights all pixels equally; sparse jet images have 98% zero cells that dominate
  - Implement: `L1_weighted = L1(fake, hr, weight=log1p(hr+1))` — penalises errors in high-energy cells more
  - Add `--lambda-sparsity` flag, default 0 (backward compatible)

- [ ] **P2.3 — Radial profile consistency loss**
  - `L_rad = MSE(radial_profile(E_pred), radial_profile(E_true))`
  - Radial profile = mean energy per r-bin (sum over phi, mean over batch)
  - Critical for CaloChallenge: azimuthal profile MSE is 427 vs radial MSE 82 — phi structure is not being learned
  - Add `--lambda-profile` flag, default 0

- [ ] **P2.4 — Per-class response conditioning (parquet only)**
  - Currently the same physics loss is applied to class 0 (gluon) and class 1 (quark)
  - Gluon jets are wider, quark jets are narrower — different SR difficulty
  - Add optional class-conditional physics loss: `λ_phys * mean_per_class(|response - 1|)`
  - Use the `y` label from the parquet batch (already loaded)

- [ ] **P2.5 — Tune λ_physics**
  - Current: 10 — gives 7.4% under-response on parquet
  - Try: 15, 20 on next full runs
  - Log response at every epoch to `metrics.jsonl` (already done) and plot convergence

---

## PRIORITY 3 — Resolution invariance

Current: generator accepts any `target_size` at forward time (already dynamic).  
Problem: trained only at one resolution — generalisation to other sizes not validated.

- [ ] **P3.1 — Multi-scale training (random crop augmentation)**
  - During training, randomly sample `target_size` from a set: `{(45,144), (90,288), (125,125)}`
  - Each batch uses a single target size (no mixed-size batches)
  - Add `--multi-scale` flag that enables this; default off
  - This forces the generator to learn resolution-agnostic features

- [ ] **P3.2 — Validate at multiple output resolutions**
  - After training at native resolution, run `save_experiment.py` with different `--hr-size` values
  - Check that SSIM and response do not degrade more than 5% vs native
  - Document in `docs/RESOLUTION_INVARIANCE.md`

- [ ] **P3.3 — Adaptive crop for parquet (125×125 → any target)**
  - Parquet HR is always 125×125; when `--hr-size` differs, the training loop already resizes via `F.interpolate`
  - Verify checkpoint trained at 125×125 generalises to 64×64 and 256×256 without fine-tuning

---

## PRIORITY 4 — Model improvements

- [ ] **P4.1 — Increase generator capacity for full runs**
  - Try `--gen-channels 96 --gen-blocks 16` (2× current capacity)
  - Current 64ch/8block model: ~2.1M params — likely underpowered for 125×125 jet images
  - Benchmark: train time per epoch should stay under 10 min on T4

- [ ] **P4.2 — Fix discriminator collapse**
  - D loss hits ~0.04 by epoch 5 (HDF5) — adversarial signal effectively dead
  - Try: `--n-critic 2` (update D twice per G step) AND `--d-lr-ratio 0.3`
  - Alternative: R1 gradient penalty on real samples (add `--r1-gamma` flag)

- [ ] **P4.3 — FiLM conditioning on physics variables (pt, m0)**
  - SwinIR already has a FiLM variant — port the idea to the GAN generator
  - Per-residual-block γ/β from `MLP([pt, m0, y] → 128d)`, zero-initialized
  - Hypothesis: conditioning on jet kinematics will improve per-class response
  - Implement as `--use-film` flag in `Generator`

---

## PRIORITY 5 — Comparison & paper

- [ ] **P5.1 — Add SwinIR results to cross-dataset comparison table**
  - SwinIR val L1 (parquet): 0.0989, response: 0.971 (from project memory)
  - Need to run SwinIR on HDF5 dataset for a fair comparison
  - Update `RUN_LOG.md` comparison table when done

- [ ] **P5.2 — Statistical significance**
  - Current val set: 500 samples (capped) — too small for error bars
  - Re-run `save_experiment.py` without `--max-val-samples` on the full val split
  - Report mean ± std on all physics metrics with N > 1000

- [ ] **P5.3 — Bicubic baseline is wrong for parquet**
  - Bicubic response = 1.088 on parquet — bicubic *over*-estimates energy
  - This is because LR images already hold ~25% of HR energy (from dataset EDA)
  - Bicubic upsampling of the LR doesn't recover the missing 75% — it just scales up what's there
  - Document this clearly in the paper: bicubic is not a valid upper bound for this dataset

- [ ] **P5.4 — Write paper section: "Physics-Aware Loss for Jet SR"**
  - Draft in `docs/PAPER_NOTES.md`
  - Sections needed: dataset description, model architecture, loss function derivation, results table, figures

---

## Kaggle / Colab notebook

- [x] **Notebook created:** `notebooks/kaggle_full_training.ipynb`
  - Handles dataset upload, full training, save_experiment, artifact download
  - See notebook for cell-by-cell instructions

---

## Quick reference — suggested next commands

```bash
# On Kaggle/Colab with GPU:
python train_srgan.py \
    --dataset-type parquet --dataset-path ../datasets \
    --epochs 20 --batch-size 64 \
    --lambda-l1 50 --lambda-physics 15 \
    --hr-size 125 125 \
    --output-dir experiments/full/parquet_run_01

python train_srgan.py \
    --dataset-type hdf5 --dataset-path ../datasets/calochallenge_dataset2 \
    --epochs 20 --batch-size 64 \
    --n-critic 2 --lambda-physics 15 \
    --hr-size 45 144 --scale-factor 2.0 \
    --output-dir experiments/full/hdf5_run_01
```
