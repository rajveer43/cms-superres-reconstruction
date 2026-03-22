# Run Analysis

Run directory: `outputs/long_run`

## Training curves

- Final train G loss: 3.847604
- Final train D loss: 0.179096
- Final train L1: 0.071289
- Final val L1: 0.082832

## Physics proxies

### overall
- response: 0.517927 +/- 0.056737
- relative_error: -0.482073 +/- 0.056737
- energy_ratio_pt: 1.124930 +/- 0.246950
- energy_ratio_m0: 6.235737 +/- 2.183678
- energy_ratio_lr: 0.543870 +/- 0.059697

### class_0
- response: 0.519953 +/- 0.049447
- relative_error: -0.480047 +/- 0.049447
- energy_ratio_pt: 1.138811 +/- 0.244549
- energy_ratio_m0: 5.720568 +/- 1.689679
- energy_ratio_lr: 0.545829 +/- 0.052009

### class_1
- response: 0.515876 +/- 0.063203
- relative_error: -0.484124 +/- 0.063203
- energy_ratio_pt: 1.110875 +/- 0.248570
- energy_ratio_m0: 6.757385 +/- 2.482961
- energy_ratio_lr: 0.541887 +/- 0.066524

Notes:
- `response` is the ratio of generated total image intensity to the target HR intensity.
- `energy_ratio_pt` and `energy_ratio_m0` are image-intensity proxies normalized by event metadata.
- `class_0` / `class_1` are the label values in the parquet `y` column.