# Run Analysis

Run directory: `outputs/balanced_run`

## Training curves

- Final train G loss: 4.829534
- Final train D loss: 0.031346
- Final train L1: 0.089353
- Final val L1: 0.105303

## Results table

| Group | GAN response | Bicubic response | GAN relative error | Bicubic relative error |
| --- | ---: | ---: | ---: | ---: |
| Overall | 1.0473 | 0.9523 | 0.0473 | -0.0477 |
| class_0 | 1.0550 | 0.9526 | 0.0550 | -0.0474 |
| class_1 | 1.0396 | 0.9520 | 0.0396 | -0.0480 |

## Physics proxies

### overall
- gen
  - response: 1.047336 +/- 0.036894
  - relative_error: 0.047336 +/- 0.036894
  - energy_ratio_pt: 2.268672 +/- 0.410665
  - energy_ratio_m0: 12.514580 +/- 3.593085
  - energy_ratio_lr: 4.189343 +/- 0.147575
- bicubic
  - response: 0.952320 +/- 0.001028
  - relative_error: -0.047680 +/- 0.001028
  - energy_ratio_pt: 2.065392 +/- 0.383411
  - energy_ratio_m0: 11.427881 +/- 3.464842
  - energy_ratio_lr: 3.809279 +/- 0.004113

### class_0
- gen
  - response: 1.055019 +/- 0.035743
  - relative_error: 0.055019 +/- 0.035743
  - energy_ratio_pt: 2.304123 +/- 0.417386
  - energy_ratio_m0: 11.556118 +/- 3.080368
  - energy_ratio_lr: 4.220076 +/- 0.142970
- bicubic
  - response: 0.952609 +/- 0.000965
  - relative_error: -0.047391 +/- 0.000965
  - energy_ratio_pt: 2.084261 +/- 0.397498
  - energy_ratio_m0: 10.478934 +/- 2.976359
  - energy_ratio_lr: 3.810435 +/- 0.003859

### class_1
- gen
  - response: 1.039556 +/- 0.036407
  - relative_error: 0.039556 +/- 0.036407
  - energy_ratio_pt: 2.232775 +/- 0.400561
  - energy_ratio_m0: 13.485098 +/- 3.808252
  - energy_ratio_lr: 4.158222 +/- 0.145629
- bicubic
  - response: 0.952027 +/- 0.001007
  - relative_error: -0.047973 +/- 0.001007
  - energy_ratio_pt: 2.046286 +/- 0.367613
  - energy_ratio_m0: 12.388765 +/- 3.654586
  - energy_ratio_lr: 3.808108 +/- 0.004030

Notes:
- `response` is the ratio of reconstructed total image intensity to the target HR intensity.
- `energy_ratio_pt` and `energy_ratio_m0` are image-intensity proxies normalized by event metadata.
- `class_0` / `class_1` are the label values in the parquet `y` column.