# Seed Stability Report — Phase A (partial)

**Date:** 2026-08-14
**Status:** Phase A code changes done and verified. Phase A *measurements* (A4, A5)
and all of Phase B are **blocked** — see §1.
**Scope:** `multiscale_sr/`, scales 16x / 32x / 64x.

---

## 0. Headline

**The four-seed experiment described in the task prompt does not exist in this
repository.** Every training run archived here used `seed: 42`. The per-seed table
in §0 of `SEED_STABILITY_INVESTIGATION.md` cannot be reproduced, corrected, or
audited from anything on disk, because the checkpoints it describes are not here.

That inverts the investigation's priority. The recommended first step (§1a: "find
out what `--seed` each of the four evaluation runs actually used") has an answer,
but not the expected one: **there were no four evaluation runs in this repo.**

What *was* possible, and is done:

- The §1a evaluation-confound claim was checked against the code and is **correct**.
  It is now fixed (A1–A3), and the fix is verified by direct experiment.
- The §1c discriminator-freeze claim was checked against the archived metrics and
  is **correct in substance but wrong in its 64x detail** — the freeze is *earlier
  and more universal* than stated. See §3.

---

## 1. Eval-seed audit (§1a) — the runs are missing

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

Also: **log `d_skip_frac`.** `train.py:327` computes it and `train.py:336` writes it
into `train_metrics`, but no archived run has it, so every number in §3 is a proxy.
Any new run must carry it.

---

## 5. Blocked items

| Item | Blocker |
|---|---|
| A4 (eval noise floor, 8 eval seeds) | Runnable, not run — see §6 |
| A5 (corrected four-seed table) | **Seed checkpoints do not exist** |
| B1–B4 (all of Phase B) | **Seed checkpoints do not exist** |

Additional blocker found: **the archived 64x checkpoints can no longer be loaded.**
`experiments/2026-06-28_datasets_64x_local_64x_50ep/checkpoints/best.pt` has a
`net.*` generator `state_dict`, while the current `Generator` expects
`stem/body/up_stages/to_image` — `load_state_dict` raises. Only post-2026-07-09
checkpoints (16x pad128, 32x pad128 ×2) load with today's code. This is pre-existing
and unrelated to the Phase A edits, but it means even the 64x results already on
disk are not currently reproducible without a compatibility shim.

---

## 6. What can and cannot be claimed (deliverable 5)

**Cannot be claimed, per scale:** nothing about seed stability, at any scale. The
multi-seed evidence is not in this repo, so the spread in §0 of the investigation
doc can be neither confirmed nor refuted here. No recovery figure should be quoted
as a headline — emphatically not "78.6%", and equally not the archived
single-seed 64x "98.6%", which is one draw evaluated under the old moving ruler and
comes from a checkpoint that no longer loads. Every archived number is **n=1 at
`seed: 42`**.

**Can be claimed:** (i) the evaluation was confounded — one `--seed` drove the
generator-independent tagger split and init, and it is now fixed, with HR/LR AUC
demonstrably invariant across SR checkpoints (§2.1); (ii) the discriminator spends
most of training below its update floor at **every** scale, first crossing within
epochs 1–3, with 32x worst at 11–19x below floor and up to 100% of epochs below it;
(iii) the current per-scale floors are mistuned relative to measured settling points.
Claims (ii) and (iii) rest on single-seed archived runs of differing epoch budgets
and pipeline eras, so they motivate the fix — they do not establish that the freeze
*causes* the seed spread. That link is exactly what B4 arm C would test.

---

## 7. Suggested next step

Retrieve the four seed checkpoints (or re-run the sweep with
`--eval-seed` fixed and `--tagger-checkpoint` set). Then A4/A5 are a few hours, and
their result determines whether Phase B is worth running at all. If the sweep must
be re-run, add `d_skip_frac` logging first — it turns §3's proxy into a direct
measurement at no cost.
