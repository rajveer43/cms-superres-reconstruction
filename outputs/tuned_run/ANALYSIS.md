# Run Analysis

Run directory: `outputs/tuned_run`

## Training curves

- Final train G loss: 5.252087
- Final train D loss: 0.217390
- Final train L1: 0.098019
- Final val L1: 0.108302

## Results table

| Group | GAN response | Bicubic response | GAN relative error | Bicubic relative error |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.9930 | 0.9523 | -0.0070 | -0.0477 |
| class_0 | 0.9985 | 0.9526 | -0.0015 | -0.0474 |
| class_1 | 0.9875 | 0.9520 | -0.0125 | -0.0480 |

## Physics proxies

### overall
- gen
  - response: 0.993017 +/- 0.035769
  - relative_error: -0.006983 +/- 0.035769
  - energy_ratio_pt: 2.152607 +/- 0.398430
  - energy_ratio_m0: 11.876235 +/- 3.441734
  - energy_ratio_lr: 3.972069 +/- 0.143078
- bicubic
  - response: 0.952320 +/- 0.001028
  - relative_error: -0.047680 +/- 0.001028
  - energy_ratio_pt: 2.065392 +/- 0.383411
  - energy_ratio_m0: 11.427881 +/- 3.464842
  - energy_ratio_lr: 3.809279 +/- 0.004113

### class_0
- gen
  - response: 0.998513 +/- 0.031589
  - relative_error: -0.001487 +/- 0.031589
  - energy_ratio_pt: 2.182837 +/- 0.406274
  - energy_ratio_m0: 10.951058 +/- 2.961361
  - energy_ratio_lr: 3.994052 +/- 0.126355
- bicubic
  - response: 0.952609 +/- 0.000965
  - relative_error: -0.047391 +/- 0.000965
  - energy_ratio_pt: 2.084261 +/- 0.397498
  - energy_ratio_m0: 10.478934 +/- 2.976359
  - energy_ratio_lr: 3.810435 +/- 0.003859

### class_1
- gen
  - response: 0.987453 +/- 0.038769
  - relative_error: -0.012547 +/- 0.038769
  - energy_ratio_pt: 2.121997 +/- 0.387933
  - energy_ratio_m0: 12.813050 +/- 3.635348
  - energy_ratio_lr: 3.949811 +/- 0.155076
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