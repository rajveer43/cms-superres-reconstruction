# Debugging prompt: suspected train/val data leakage in multiscale_sr

## Context

`multiscale_sr` trains a GAN-based super-resolution generator on CMS jet-image
parquet data (`X_jets_LR` / `X_jets`, columns `pt`, `m0`, `y`), then separately
evaluates it with a classification-based tagging-efficiency pipeline
(`classification_eval.py`) that trains a small jet tagger on HR/LR/SR images.

The split is **file-level**, not row-level: `split_files()` in
`multiscale_sr/data/normalization.py:100` divides the sorted list of parquet
files into train files and val files (the last `n_val` files become val). Both
the streaming `ParquetJetSRDataset` and the `CachedJetSRDataset` (decode cache)
call this same function, so at the row level there should be no overlap
*if and only if* every caller passes the same `val_ratio` and the same file
list, every time, for a given run.

That "if and only if" is not enforced anywhere. This prompt is about proving
or disproving three specific leak hypotheses, in priority order, using the
actual repo code — not modifying training logic until the root cause is
confirmed.

## Files to read first

- `multiscale_sr/multiscale_sr/data/normalization.py` — `split_files`,
  `ChannelStats`, `stream_channel_stats_parquet`
- `multiscale_sr/multiscale_sr/data/factory.py` — `get_dataloader`,
  `_parquet_loader`, `_cached_parquet_loader`
- `multiscale_sr/multiscale_sr/data/cache.py` — `CachedJetSRDataset`,
  `ensure_hr_cache`, `_cache_is_valid`
- `multiscale_sr/train.py` — how `val_ratio` and `stats_cache_path` are wired
  from CLI args
- `multiscale_sr/classification_eval.py` — `main()`, especially lines
  ~433-453 (checkpoint args, stats cache reuse, `val_loader` construction) and
  `_split_indices` (~92-96)
- `multiscale_sr/run_evaluations.py` — `run_one()` (~93-112): note it does
  **not** pass `--val-ratio`

## Hypothesis 1 (highest priority): `val_ratio` mismatch between training and evaluation

**Claim to verify:** `classification_eval.py` defaults `--val-ratio` to 0.33
(`classification_eval.py:78`). `run_evaluations.py` never forwards a
`--val-ratio` flag when it shells out to `classification_eval.py`
(`run_evaluations.py:98-107`). If any training run in `experiments/*/config.yaml`
used a `val_ratio` other than 0.33, the "val" file split used to *evaluate*
that checkpoint differs from the "val" file split it was *trained* against —
meaning some files scored as "val" in `classification_eval.py` were actually
in the generator's training set.

**How to check:**
1. For every run directory under `multiscale_sr/experiments/`, `cat
   config.yaml` and extract `val_ratio` (grep for `val_ratio:`).
2. Compare each value against 0.33 (the `classification_eval.py` default).
3. For any run where they differ, recompute `split_files(files, val_ratio)`
   for both ratios (a Python one-liner using
   `multiscale_sr.data.normalization.discover_parquet_files` +
   `split_files`) and diff the resulting file lists. If the val-file sets
   differ, that run's classification eval was scored partly on files the
   generator trained on — confirmed leak for that run.
4. Report which runs (if any) are affected, and whether `evaluate.py` (used
   for plain L1/PSNR val metrics) has the same mismatch — check its
   `--val-ratio` default too.

**Fix if confirmed:** `run_evaluations.py` should read `val_ratio` out of
each run's `config.yaml` and pass it explicitly to `classification_eval.py`
via `--val-ratio`, instead of relying on the flag's default.

## Hypothesis 2: stale `stats_cache_path` masks a split change

**Claim to verify:** `get_dataloader()` in `factory.py:207-210` loads
`ChannelStats` from `stats_cache_path` if the file exists, *unconditionally*
— it does not record or check which `val_ratio` or file manifest produced
those stats. If a stats JSON was computed once (say under `val_ratio=0.2`)
and is later reused for a run/eval with a different `val_ratio`, the
normalization statistics were computed over a train set that now includes
some of the "new" val files (or vice versa) — a subtler, distribution-level
leak (val images normalized using statistics partly derived from themselves).

**How to check:**
1. Find every `normalization.json` under `multiscale_sr/experiments/*/` and
   `multiscale_sr/evaluations/*/`.
2. For each, find what `val_ratio` was in effect when it was written (check
   the corresponding `config.yaml`, or `git log`/mtime ordering against
   other runs that share a stats file path).
3. Recompute `stream_channel_stats_parquet` on the *correct* train split for
   that ratio and diff mean/std against what's cached. A nontrivial
   difference (e.g. >1% relative) indicates the cached stats don't match the
   claimed split.

**Fix if confirmed:** Store the manifest (file list + val_ratio) alongside
`ChannelStats.save()`, and have `get_dataloader` refuse to reuse a cache
whose manifest doesn't match the current call (mirroring what
`cache.py::_cache_is_valid` already does for the HR decode cache — that part
of the codebase got this right, normalization.py did not).

## Hypothesis 3: decode-cache staleness across dataset changes

**Claim to verify:** `_cache_is_valid()` in `cache.py:52-66` keys the HR
decode cache on a manifest of (name, size, mtime, row count) per file. This
is correct *if* the underlying parquet files never change identity without
their mtime changing (e.g. no `cp -p` that preserves timestamps, no network
filesystem with unreliable mtime). Confirm this isn't silently stale by:
1. Checking whether any `.sr_cache/{train,val}/meta.json` currently on disk
   has a manifest that no longer matches `discover_parquet_files()` +
   `os.stat()` on the live parquet files (should self-heal via
   `ensure_hr_cache`, but confirm it actually rebuilds rather than erroring
   or silently reusing).
2. This is the lowest-priority hypothesis — the design here already looks
   sound — but rule it out so it's not revisited later.

## What "done" looks like

Produce a short report answering, per hypothesis: confirmed / not
confirmed, which specific run directories (if any) are affected, and the
one-line root cause. Do not apply fixes until the root cause is confirmed
against real files in this repo — several of these are "possible given the
code" but may not have actually triggered in the runs you have on disk.
