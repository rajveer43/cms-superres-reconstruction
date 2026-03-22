# Output Comparison and Final Run Justification

This document compares the major experiment outputs and explains why `outputs/final_run/` is the best checkpoint for presentation.

## What counts as an output

The project produces three kinds of outputs:

1. **Dataset analysis outputs**
   - summarize the raw parquet data
   - justify the model and loss design

2. **Training-run outputs**
   - checkpoints
   - metrics
   - saved samples
   - comparison plots

3. **Report outputs**
   - markdown summaries
   - plots for a presentation
   - final narrative for the repo

The most important training outputs are:

- `outputs/results_run/`
- `outputs/long_run/`
- `outputs/tuned_run/`
- `outputs/adjusted_run/`
- `outputs/balanced_run/`
- `outputs/final_run/`

## Dataset evidence that guides the whole project

The sampled dataset analysis shows:

- class balance is close to even: `1042` vs `1006`
- LR shape is `(3, 64, 64)` and HR shape is `(3, 125, 125)`
- mean LR/HR intensity ratio is `0.25`
- the images are sparse

Reasoning:

- the task is not a pure image translation problem; it is an energy recovery problem
- the model must preserve both structure and total intensity
- a physics-aware loss is justified because LR systematically under-represents HR energy

That dataset behavior is the main reason the final model uses an upsample-then-refine generator and a hybrid objective:

- upsampling before refinement
- L1 reconstruction
- adversarial sharpening
- a physics loss on total intensity

## Comparison criteria

I use five criteria to judge the outputs:

1. **Validation reconstruction**
   - lower validation L1 is better
   - tells us whether the model actually generalizes

2. **Physics response**
   - response = reconstructed total intensity / target total intensity
   - ideal value is close to `1.0`
   - too low means the model underestimates energy
   - too high means it overestimates energy

3. **Class consistency**
   - class 0 and class 1 should behave similarly
   - large class gaps would suggest bias or instability

4. **Baseline comparison**
   - compare against bicubic interpolation
   - the model should improve on the naive upsampling baseline

5. **Training stability**
   - losses should not explode
   - discriminator should not collapse
   - samples should improve over epochs

## Run-by-run comparison

### 1. `outputs/results_run/`

Purpose:

- compact baseline run
- quick test of the GAN pipeline

Observed behavior:

- validation L1 improved monotonically over 3 epochs
- raw-space L1 was better than bicubic
- normalized L1, RMSE, and PSNR were not consistently better than bicubic

Why it matters:

- this run proved the pipeline worked
- it showed the model could learn a meaningful mapping
- it also exposed the main weakness: reconstruction quality alone was not enough

Limitation:

- no physics loss yet
- no explicit control over total intensity

Conclusion:

- useful as a sanity check
- not strong enough for presentation

### 2. `outputs/long_run/`

Purpose:

- longer training without the later physics tuning

Observed behavior:

- validation L1 dropped to `0.082832`, the best reconstruction among the early runs
- physics response was only `0.5179`
- the model reconstructed about half of the target intensity on average

Why it matters:

- the model was learning structure
- but it was systematically under-energized
- that means the images may look sharper, but they are not physically correct

Conclusion:

- good reconstruction progress
- unacceptable physics behavior
- not suitable as the final checkpoint

### 3. `outputs/tuned_run/`

Purpose:

- first physics-aware tuning pass

Observed behavior:

- response moved to `0.9930`
- class-wise response was also close to `1.0`
- validation L1 was `0.108302`

Why it matters:

- this run fixed the major physics failure from `long_run`
- the model stopped underestimating intensity
- however, reconstruction quality was worse than the final run

Conclusion:

- important intermediate step
- not the best tradeoff because reconstruction lagged

### 4. `outputs/adjusted_run/`

Purpose:

- adjust the physics weight upward to pull response closer to one

Observed behavior:

- validation L1 improved to `0.099067`
- response overshot to `1.0217`
- class-wise response was also slightly above one

Why it matters:

- the model now tended to over-reconstruct instead of under-reconstruct
- that is better than the earlier under-response, but still not ideal

Conclusion:

- better than `tuned_run` on reconstruction
- still not the most balanced physics result

### 5. `outputs/balanced_run/`

Purpose:

- test a stronger physics loss balance

Observed behavior:

- response increased further to `1.0473`
- validation L1 worsened to `0.105303`

Why it matters:

- the physics term was too strong
- the model started to over-correct the total intensity
- reconstruction quality suffered

Conclusion:

- useful for understanding the loss tradeoff
- not a good final checkpoint because it overshoots energy

### 6. `outputs/final_run/`

Purpose:

- final tuned checkpoint intended for presentation

Observed behavior:

- validation L1: `0.097253`
- response: `1.0098`
- class 0 response: `1.0143`
- class 1 response: `1.0052`
- raw-space L1 beats bicubic: `0.00698` vs `0.00846`
- raw-space PSNR beats bicubic: `14.32 dB` vs `14.13 dB`

Why it matters:

- this run sits closest to the desired physics target without sacrificing reconstruction
- it improves on bicubic in the metric that matters most for sparse images: raw-space L1
- it keeps the response near one for both classes
- it is stable and has the cleanest balance of all runs

Conclusion:

- this is the best checkpoint for presentation

## Compact comparison table

| Run | Main goal | Val L1 | Response | Verdict |
| --- | --- | ---: | ---: | --- |
| `results_run` | pipeline sanity check | `0.10771` | not recorded | good proof of concept |
| `long_run` | longer optimization | `0.08283` | `0.5179` | best L1, bad physics |
| `tuned_run` | physics correction | `0.10830` | `0.9930` | physics fixed, L1 weaker |
| `adjusted_run` | tighter balance | `0.09907` | `1.0217` | close, but overshoots |
| `balanced_run` | stronger physics emphasis | `0.10530` | `1.0473` | too much physics weight |
| `final_run` | best tradeoff | `0.09725` | `1.0098` | best overall |

## Why `final_run` is the best

`final_run` wins because it is the only checkpoint that simultaneously satisfies the three things we actually need:

1. **Good reconstruction**
   - validation L1 is lower than the other physics-tuned runs
   - raw L1 is better than bicubic

2. **Correct physics response**
   - total intensity is within about `1%` of the target on average
   - both classes stay close to the target response

3. **Stable behavior**
   - no obvious divergence
   - no collapse
   - no major class-specific imbalance

In other words, `final_run` is not merely the lowest-loss model and not merely the most physics-correct model. It is the best compromise between both objectives, which is the right criterion for this task.
