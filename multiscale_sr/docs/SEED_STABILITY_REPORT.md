# Seed Stability Report — Phase A complete, SEMD instrumentation landed

**Date:** 2026-08-14, revised 2026-08-22
**Status:** Phase A code changes done and verified. The four-seed 64x sweep has
since been **run externally (Colab) and its results are in §0.1**, superseding the
"runs do not exist" finding that blocked A5. SEMD instrumentation is landed and
tested; the correlation verdict (S6) is **pending the retro-evaluation**.
**Scope:** `multiscale_sr/`, scales 16x / 32x / 64x.

---

## 0. Headline

The four-seed 64x experiment **now exists** — it was run in Colab, not in this
checkout, which is why the original audit (§1, retained below for provenance)
could not find it. With the frozen ruler from A1–A3 in place, its result is:

> **The physics metrics are blind to what the tagger measures.**
> Tagging efficiency varies by 10.2% (CV) across seeds while every logged physics
> quantity holds to under 1%. The metrics are not wrong; they measure the wrong
> thing.

Three findings, in order of how much they should change what we do:

**(a) The blindness is quantitative.** See the table in §0.1: tagging efficiency
CV 10.2%, against energy response 0.8%, peak ratio 0.4%, pT–E correlation 0.6%,
energy correlation 0.1%. No existing metric distinguishes the best seed from the
worst.

**(b) The blindness is structural, not incidental.** Reading `engine.py`:
`energy_response` is `sum(pred)/sum(target)`; `peak_metrics` is
`max(pred)/max(target)` plus a nonzero pixel count; `physics_loss` is
`|response - 1|`. **All three are invariant under arbitrary permutation of the
pixels** — shuffle the SR image and not one of them changes. They constrain how
much energy exists and how bright the peak is, never *where* the energy sits.
Jet tagging is almost entirely radiation geometry: prong count, subjet splitting,
angular spread. So a seed can match total energy and peak height while placing
substructure wrongly, and the dashboard calls it a good run. This is asserted
executably in `tests/test_spectral.py::test_geometry_discriminates_where_pixel_metrics_are_blind`.

**(c) The anti-correlation is the smoking gun.** Seed 456 has the *best* energy
fidelity (1.16% error) and the *worst* tagging (76.4%). Seed 123 has the *worst*
energy fidelity (5.82%) and the *best* tagging (97.9%). Optimizing the current
physics loss harder would have selected 456 over 123. That is why "train longer"
or "tune λ_physics" cannot fix this, and why the fix has to add a *geometric*
term rather than reweight the existing ones.

Also worth recording: PSNR has CV 200.7% with sign flips (+2.585, −6.423, +1.419,
−8.977) — two of four seeds are *worse than bicubic*. Together with (a) this is a
model that is not converging to a consistent solution, consistent with the
discriminator-freeze finding in §3.

---

## 0.1 The four-seed 64x result

Seeds 123/456/789/999, one frozen HR tagger, `--eval-seed 0`.

| Quantity | Value | CV |
|---|---|---|
| HR AUC | 0.6981 | — (fixed by construction) |
| LR/bicubic AUC | 0.5183 | — (fixed by construction) |
| SR AUC | 0.6098 ± 0.0624 | — |
| **Tagging efficiency** | **87.4% ± 8.9%** (76.4–97.9) | **10.2%** |
| **Recovery fraction** | **50.9% ± 34.7%** (8.6–91.8) | — |
| Energy response | 1.0176 ± 0.0078 | 0.8% |
| Peak ratio | 0.8855 ± 0.0033 | 0.4% |
| pT–E correlation | 0.7088 ± 0.0044 | 0.6% |
| Energy correlation | 0.9964 ± 0.0014 | 0.1% |
| Validation L1 | 0.0670 ± 0.0028 | 4.2% |
| PSNR (dB, vs LR) | −2.849 ± 5.717 | 200.7% |

Per-seed tagging efficiency: **123 → 97.9%, 456 → 76.4%, 789 → 89.7%, 999 → 85.4%.**

HR and LR AUC are identical across all four seeds, which is the A3 pass condition
holding in production: the differences above are the generator's, not the
evaluator's.

**The headline number is the ±34.7%, not the 50.9%.** A recovery fraction whose
standard deviation is two-thirds of its mean is not a result to quote as a point
estimate.

---

## 0.2 SEMD instrumentation (landed 2026-08-22)

To measure what the existing metrics cannot, the p=2 Spectral Energy Mover's
Distance of **arXiv:2410.05379v3** (Gambhir, Larkoski, Thaler — SPECTER) is now
implemented in `multiscale_sr/spectral.py`. SEMD represents an event as an
energy-weighted distribution of *pairwise angles* (Eq. 2.1) and compares two such
distributions in closed form (Eq. 2.19). Pixel permutation changes it — that is
precisely why it was chosen.

**Design decision: metrics first, loss second, and the loss is gated off.**

- SEMD is computed **unconditionally** as an evaluation metric
  (`classification_eval.py`) and every epoch during training (`val_semd`), so
  correlation data accumulates even from runs that do not optimize it.
- SEMD as a *generator loss* is implemented but **defaults to `--lambda-semd 0.0`
  (OFF)**, leaving the loss bit-for-bit what it was. It is guarded by an `if`, not
  multiplied by zero, so a disabled run pays none of its cost.

The reason for that ordering is finding (a): this is a *measurement* failure
before it is an optimization failure. Enabling a loss term against a quantity
never once observed on this pipeline — given finding (c), where the existing
objective is mildly anti-correlated with the goal — is a concrete way to make
things worse. **The falsifiable prediction is that SEMD ranks the four seeds in
tagging order where the existing physics metrics do not.**

### Honest statement of the adaptation

The paper operates on particle lists; we have images. Stated openly because each
of these is a place a wrong number could hide:

1. **Pixels as particles** — each pixel above threshold contributes its raw
   (denormalized) energy at its `(eta, phi)` centre; channels summed.
2. **Ground metric** — `omega_ij = dist_ij ** beta` on the pixel grid. Admissible
   because the paper requires only that `omega_ij` be symmetric with
   `omega_ii = 0`; it need not be a proper metric.
3. **Energy balancing is mandatory** — Eq. 2.7 assumes `E_tot^A == E_tot^B`, and
   SR/HR do not satisfy that by construction (it is exactly what
   `energy_response != 1` measures). The deficit is deposited at an explicit,
   logged scale `omega_R`, following the paper's prescription. Rescaling instead
   would erase the very mismatch we want the metric to notice.
4. **Top-K truncation** — 128×128 gives ~2.7e8 pairs, so only the `topk` (default
   128, matching the paper's own N=125 benchmark shapes) brightest pixels are
   kept. The discarded tail is folded into the balancing term, not dropped: energy
   is conserved (asserted in `test_extract_particles_conserves_energy`).

**Because of (4) this is labelled "SEMD (top-K pixel approximation)" everywhere
and is not the paper's exact observable.** `topk`, `omega_R`, `beta`, and
`threshold` are recorded in every results JSON, so no SEMD number is ever
unattributable to the K that produced it.

### What the tests establish

`tests/test_spectral.py` (31 tests, all passing) covers the metric axioms
(identity, symmetry, triangle inequality on `sqrt(SEMD)`), the invariances
(rotation, translation, particle permutation, inert zero-energy padding), energy
conservation and balancing, gradient finiteness through the sort *and* at exact
identity, and batching independence.

Two of these deserve mention because they corrected my own expectations:

- **Translation invariance is real and is a feature.** A rigid shift of the whole
  jet gives *exactly* zero distance, because SEMD depends only on pairwise
  distances. An early version of the ordering test compared translated images and
  therefore compared zero against zero; it now varies *internal* prong separation
  instead (`test_closer_geometry_gives_smaller_distance`), and the invariance is
  asserted separately.
- **"Zero" is only zero to the cancellation floor.** Eq. 2.19 subtracts three
  large, nearly-equal terms, so `SEMD(A,A)` lands ~1e-10 in float64, not 0.0. The
  identity test compares against the scale of the terms being cancelled rather
  than an absolute constant.

### Status: the verdict is not yet in

`semd_correlation.py` produces the decision table — every physics metric scored
against tagging efficiency, SEMD given no special treatment. It has been verified
on fixtures but **not yet run against the four real checkpoints**, so no claim
about whether SEMD correlates with tagging can be made here yet.

Two guards are built into that script, both of which matter at n=4:

- It reports the **exact permutation p-value**. With four points the smallest
  attainable value is 0.083, so *nothing* in this comparison can reach p<0.05.
- A **tie between SEMD and an existing metric is reported as inconclusive, not as
  a win.** Rank correlation saturates trivially on four points; several metrics
  reach |ρ|=1.0 by chance. Ties break on Pearson r and still require a margin.

If SEMD does not beat the existing metrics, **that is the finding and it gets
reported as a null result.** Sweeping `topk`/`omega_R`/`threshold` until the known
ranking reappears would be curve-fitting to four points, and would manufacture
exactly the kind of result this investigation exists to prevent.

---

## 1. Eval-seed audit (§1a) — the runs were missing *from this checkout*

> **Superseded 2026-08-22, retained for provenance.** The conclusion below —
> that the four seed runs do not exist — was correct *about this repository* and
> is why A5 was blocked. The sweep was subsequently run in Colab; its results are
> in §0.1. The audit's diagnostic reasoning still stands: §1.3's observation that
> a fixed eval seed pins HR/LR AUC is exactly what the frozen-tagger fix now
> guarantees, and it is why the §0.1 numbers can be trusted as measurements of
> the generator.

### 1.1 What was searched

| Evidence | Result |
|---|---|
| `experiments/*/config.yaml` (14 runs) | **all `seed: 42`** |
| W&B run configs, `experiments/*/wandb/*/files/config.yaml` (16 runs) | **all `seed: 42`** (`uniq -c` → `16 42`) |
| Checkpoints on disk | 13 runs × `best.pt`/`latest.pt`; none seed-named |
| Repo-wide grep for `123`/`456`/`789`/`999` near "seed" | **1 hit: the task doc itself** |
| `evaluations/*/batch_manifest.json` | `"seed": 42` for both batch dates |
| Sibling dirs, other branches, stashes | no seed-sweep artifacts |

### 1.2 The archived 64x numbers don't match the table

The one archived 64x classification eval
(`evaluations/2026-07-07/64x__2026-06-28_datasets_64x_local_64x_50ep/eval.log`):

```
HR AUC=0.6980   LR AUC=0.5433   SR AUC=0.6958   efficiency 99.7%   recovery 98.6%
```

This matches **no row** of the four-seed table (whose SR AUCs are 0.658 / 0.527 /
0.621 / 0.584). So the seed sweep was neither run from this checkout nor copied
back into it.

### 1.3 A corroborating detail

Across **all 16** archived classification evals — every scale, both batch dates —
`HR AUC` is **exactly 0.6980** and `LR AUC` is exactly constant per scale
(16x: 0.5679, 32x: 0.5258, 64x: 0.5433). That is precisely the behaviour §1a
predicts when the eval seed is held fixed: HR/LR don't depend on the generator, so
with a fixed `--seed 42` they are pinned. The 0.06 HR spread in the task's table is
therefore *diagnostic of a varying eval seed* in whatever environment produced it —
supporting the §1a hypothesis, but with data not present here.

**Conclusion:** the "GAN is seed-sensitive" conclusion is **not established**, and
cannot be evaluated from this repo. Retrieve the seed checkpoints (Colab/Kaggle/
W&B cloud) before Phase B is attempted.

---

## 2. Phase A code changes — done and verified

### A1. Eval seed decoupled from SR seed — `classification_eval.py`

- Added `--eval-seed` (default `0`), which now drives `seed_everything`,
  `_split_indices`, and **every** `train_tagger` call.
- `--seed` kept as a deprecated alias (default `None`); it warns, and errors if it
  contradicts an explicit `--eval-seed`.
- `eval_seed`, `test_frac`, `max_samples` are now recorded in
  `classification_eval.json`, so no future result is unattributable.

### A2. Frozen, reusable HR tagger

- Added `--tagger-checkpoint PATH`: loads if present, else trains and saves.
- The artifact stores weights, `width`, `in_channels`, the exact `test_idx`,
  `eval_seed`, `test_frac`, and `n`.
- **Leak guard:** on load, the cached `test_idx` is compared against the current
  split; a mismatch is a hard error, because reusing that tagger would score rows
  it was trained on. Verified firing (§2.1).
- `run_evaluations.py`: `--eval-seed` + `--tagger-dir`, using one frozen tagger per
  `(scale, val_ratio)` — the split depends on both, so a single shared tagger would
  trip the guard. Scale is a parameter; nothing is hardcoded to 64x.

### 2.1 A3 verification — the ruler is now fixed

Two **different** 32x SR checkpoints, same `--eval-seed 0`, same frozen tagger:

| source | `pad128_stabilized` | `pad128_progressive_13pct` | Δ |
|---|---|---|---|
| HR | 0.6474335857675202 | 0.6474335857675202 | **0.000e+00** |
| LR | 0.5337480466402211 | 0.5337480466402211 | **0.000e+00** |
| SR | 0.6207476860199543 | 0.5908462555595624 | 2.99e-02 |

HR and LR are **bitwise identical** while SR moves — exactly the A3 pass condition.
Guard check: rerunning with `--eval-seed 7` against the seed-0 tagger correctly
refuses rather than silently leaking.

---

## 3. Discriminator freeze (§1c) — confirmed, and worse than described

Measured from `experiments/*/metrics.jsonl` against each scale's **current config**
floor (16x: 0.02, 32x/64x: 0.05). `d_skip_frac` is absent from all archived runs
(it postdates them), so `train_d_loss` vs. floor is the proxy, as §1c anticipated.

| run | scale | floor | epochs | first < floor | frac below | mean last-5 | min |
|---|---|---|---|---|---|---|---|
| 2026-06-23_16x_baseline | 16 | 0.020 | 5 | 3 | 0.60 | 0.04182 | 0.00992 |
| 2026-07-01_16x_local_30ep | 16 | 0.020 | 23 | 3 | 0.91 | 0.00304 | 0.00265 |
| 2026-07-03_16x_cached_5ep | 16 | 0.020 | 5 | 2 | 0.80 | 0.01144 | 0.00239 |
| 2026-07-09_16x_pad128_5pct | 16 | 0.020 | 20 | 8 | 0.65 | 0.00756 | 0.00592 |
| 2026-06-23_32x_baseline | 32 | 0.050 | 5 | 2 | 0.80 | 0.04564 | 0.01046 |
| 2026-06-24_32x_baseline | 32 | 0.050 | 20 | 2 | 0.90 | 0.00347 | 0.00329 |
| 2026-07-01_32x_local_30ep | 32 | 0.050 | 30 | 2 | **0.97** | 0.00257 | 0.00215 |
| 2026-07-08_32x_leakage_fix | 32 | 0.050 | 6 | 1 | **1.00** | 0.00455 | 0.00166 |
| 2026-07-10_32x_pad128_13pct | 32 | 0.050 | 20 | 2 | 0.95 | 0.00388 | 0.00333 |
| 2026-07-11_32x_stabilized | 32 | 0.100¹ | 7 | 1 | 0.86 | 0.07559 | 0.00000 |
| 2026-06-23_64x_baseline | 64 | 0.050 | 20 | 2 | 0.50 | 0.11175 | 0.02028 |
| 2026-06-28_64x_local_50ep | 64 | 0.050 | 50 | **2** | 0.42 | 0.03689 | 0.02028 |

¹ the only run whose own config recorded `d_loss_floor`; the stabilization machinery
postdates the other runs, so they are measured against today's config values.

### 3.1 Correction to §1c

§1c states 64x's freeze onset is "epoch 39/50, last ~24%". That is **not what the
metrics show**. In `2026-06-28_64x_local_64x_50ep`, `train_d_loss` first drops below
0.05 at **epoch 2**, stays below through epoch 12, *recovers* above the floor for
epochs 13–38, then re-crosses at 39 and stays low. Total time below floor is **42%**
of the run, not 24%, and onset is epoch 2, not 39.

This matters for the plan: §1c's framing ("32x freezes immediately, 64x freezes
late, so the scales disentangle the freeze from the instance-noise anneal") **does
not hold**. Every scale first crosses its floor within epochs 1–3 — i.e. at or
immediately after `adv_warmup_epochs: 3`, when the adversarial term switches on.
The 64x mid-run recovery is a real and interesting difference, but it is not the
"late freeze" the cross-scale comparison in B4 was designed around.

### 3.2 Floors are mistuned at every scale

Comparing each scale's floor to where `d_loss` actually settles (mean of last 5):

- **16x** (floor 0.02): settles 0.003–0.04. The newer pad128 run settles **0.0076**,
  ~2.6x below its floor. The comment "16x settles at d~0.04" describes only the
  oldest 5-epoch run.
- **32x** (floor 0.05): settles 0.0026–0.0046 → **11–19x below floor**. Worst case.
- **64x** (floor 0.05): settles 0.037 → below floor, though closest to it.

---

## 4. Recommendation on `d_loss_floor` (deliverable 4)

Derived from each scale's measured settling point (§3), not copied between scales —
that copying is how the current mistuning arose. Using the prompt's ≈⅓-of-settled
heuristic, and preferring the newest run per scale (current data pipeline):

| scale | settles at | current floor | **recommended** |
|---|---|---|---|
| 16x | ~0.0076 | 0.02 | **0.0025** |
| 32x | ~0.0026 | 0.05 | **0.001** |
| 64x | ~0.037 | 0.05 | **0.012** |

**I recommend against landing these as-is.** A floor tuned to a single archived seed
per scale reintroduces exactly the failure being diagnosed. The better fix is an
**adaptive rule** — e.g. floor as a running quantile of recent `d_loss`, so it
tracks the settling point instead of being pinned to one run. Either way this
should be validated across seeds (B4 arm C) before adoption, and B4 cannot run
until the seed checkpoints exist.

Also: **`d_skip_frac` is logged** — `train.py` computes it, writes it into
`train_metrics`, prints it per epoch, and it reaches both `metrics.jsonl` and W&B.
No *archived* run has it (it postdates them), which is why every number in §3 is
still a proxy; every new run carries it directly. `val_semd` now rides along on the
same path, so future runs accumulate geometric data whether or not they optimize it.

---

## 5. Blocked items (revised 2026-08-22)

| Item | Status |
|---|---|
| A4 (eval noise floor, 8 eval seeds) | Runnable, **still not run** — bounds how much of §0.1's ±8.9% is eval noise |
| A5 (corrected four-seed table) | **Done** — §0.1, produced in Colab |
| S6 (SEMD correlation verdict) | **Unblocked, not yet run** — needs `semd_correlation.py` over the four checkpoints |
| B1–B4 (all of Phase B) | Unblocked in principle: the checkpoints now exist in Drive. Gated on S6 and on the standing "report back before Phase B" instruction |

Additional blocker found: **the archived 64x checkpoints can no longer be loaded.**
`experiments/2026-06-28_datasets_64x_local_64x_50ep/checkpoints/best.pt` has a
`net.*` generator `state_dict`, while the current `Generator` expects
`stem/body/up_stages/to_image` — `load_state_dict` raises. Only post-2026-07-09
checkpoints (16x pad128, 32x pad128 ×2) load with today's code. This is pre-existing
and unrelated to the Phase A edits, but it means even the 64x results already on
disk are not currently reproducible without a compatibility shim.

---

## 6. What can and cannot be claimed (deliverable 5)

**Can be claimed:**

1. **The evaluation was confounded, and is now fixed.** One `--seed` drove the
   generator-independent tagger split and init; `--eval-seed` now decouples them
   and HR/LR AUC are bitwise invariant across SR checkpoints (§2.1), including in
   the production sweep (§0.1).
2. **The 64x model is seed-unstable at 40 epochs.** Tagging efficiency spans
   76.4–97.9% over four seeds (87.4% ± 8.9%), recovery 50.9% ± 34.7%, under a
   fixed ruler. This is now measured, not inferred.
3. **The logged physics metrics cannot distinguish those seeds** (all CV <1%
   against tagging's 10.2%), and this follows structurally from their being
   pixel-permutation invariant, not from a tuning accident.
4. **The current physics objective is mildly anti-correlated with tagging** on
   these four points — the best-energy-fidelity seed is the worst tagger.
5. **The discriminator spends most of training below its update floor at every
   scale**, first crossing within epochs 1–3, 32x worst at 11–19x below floor.
6. The per-scale floors are mistuned relative to measured settling points.

**Cannot be claimed:**

- **Any single-seed recovery figure as a headline.** "78.6%" stays retired. The
  corrected figure is **50.9% ± 34.7%**, and the ±34.7% is the headline. The
  archived single-seed 64x "98.6%" is equally unusable — one draw under the old
  moving ruler, from a checkpoint that no longer loads (§5).
- **That SEMD is a better tagging surrogate.** Not yet measured. The correlation
  table is built and tested but has not been run against the four checkpoints.
- **That the D-freeze *causes* the seed spread.** Claims 5–6 rest on single-seed
  archived runs of differing epoch budgets and pipeline eras. They motivate the
  fix; they do not establish causation. That link is what B4 arm C would test.
- **Anything about seed stability at 16x or 32x.** The sweep was 64x only.

Every claim above is attached to: **seeds {123, 456, 789, 999}, `--eval-seed 0`,
scale 64x, 40 epochs, one frozen HR tagger.**

---

## 7. Suggested next step

**Run `semd_correlation.py` against the four 64x checkpoints.** Everything else
waits on that verdict:

- **If SEMD tracks tagging where the existing metrics do not** — we have a cheap
  geometric surrogate for an expensive downstream probe, and a follow-up sweep
  with `--lambda-semd > 0` is justified. Report back before starting it.
- **If it does not** — report the null result. Either the geometric hypothesis is
  wrong or the top-K approximation is too coarse to test it, and separating those
  needs *more seeds*, not more knobs.
- **If it ties** (likely at n=4) — that is inconclusive, not a win. Add seeds.

Still open independently of SEMD: A4 (eval noise floor over 8 eval seeds, never
run) would tell us how much of the ±8.9% spread is evaluation noise rather than
generator variance — worth doing, since it bounds everything else. The 64x
checkpoint incompatibility in §5 also still blocks re-evaluating the archived runs
with today's code.
