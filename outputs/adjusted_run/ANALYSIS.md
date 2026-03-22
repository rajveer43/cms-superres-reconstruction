# Run Analysis

Run directory: `outputs/adjusted_run`

## Training curves

- Final train G loss: 5.638090
- Final train D loss: 0.033384
- Final train L1: 0.087203
- Final val L1: 0.099067

## Results table

| Group | GAN response | Bicubic response | GAN relative error | Bicubic relative error |
| --- | ---: | ---: | ---: | ---: |
| Overall | 1.0217 | 0.9523 | 0.0217 | -0.0477 |
| class_0 | 1.0251 | 0.9526 | 0.0251 | -0.0474 |
| class_1 | 1.0183 | 0.9520 | 0.0183 | -0.0480 |

## Physics proxies

### overall
- gen
  - response: 1.021708 +/- 0.031942
  - relative_error: 0.021708 +/- 0.031942
  - energy_ratio_pt: 2.213080 +/- 0.400564
  - energy_ratio_m0: 12.219588 +/- 3.559804
  - energy_ratio_lr: 4.086833 +/- 0.127769
- bicubic
  - response: 0.952320 +/- 0.001028
  - relative_error: -0.047680 +/- 0.001028
  - energy_ratio_pt: 2.065392 +/- 0.383411
  - energy_ratio_m0: 11.427881 +/- 3.464842
  - energy_ratio_lr: 3.809279 +/- 0.004113

### class_0
- gen
  - response: 1.025052 +/- 0.029326
  - relative_error: 0.025052 +/- 0.029326
  - energy_ratio_pt: 2.239975 +/- 0.412624
  - energy_ratio_m0: 11.237298 +/- 3.031410
  - energy_ratio_lr: 4.100210 +/- 0.117305
- bicubic
  - response: 0.952609 +/- 0.000965
  - relative_error: -0.047391 +/- 0.000965
  - energy_ratio_pt: 2.084261 +/- 0.397498
  - energy_ratio_m0: 10.478934 +/- 2.976359
  - energy_ratio_lr: 3.810435 +/- 0.003859

### class_1
- gen
  - response: 1.018322 +/- 0.034056
  - relative_error: 0.018322 +/- 0.034056
  - energy_ratio_pt: 2.185847 +/- 0.386066
  - energy_ratio_m0: 13.214234 +/- 3.772587
  - energy_ratio_lr: 4.073288 +/- 0.136223
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