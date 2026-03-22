# Run Analysis

Run directory: `outputs/final_run`

## Training curves

- Final train G loss: 5.174381
- Final train D loss: 0.030104
- Final train L1: 0.082250
- Final val L1: 0.097253

## Results table

| Group | GAN response | Bicubic response | GAN relative error | Bicubic relative error |
| --- | ---: | ---: | ---: | ---: |
| Overall | 1.0098 | 0.9523 | 0.0098 | -0.0477 |
| class_0 | 1.0143 | 0.9526 | 0.0143 | -0.0474 |
| class_1 | 1.0052 | 0.9520 | 0.0052 | -0.0480 |

## Physics proxies

### overall
- gen
  - response: 1.009765 +/- 0.032392
  - relative_error: 0.009765 +/- 0.032392
  - energy_ratio_pt: 2.186587 +/- 0.391586
  - energy_ratio_m0: 12.072165 +/- 3.507818
  - energy_ratio_lr: 4.039061 +/- 0.129570
- bicubic
  - response: 0.952320 +/- 0.001028
  - relative_error: -0.047680 +/- 0.001028
  - energy_ratio_pt: 2.065392 +/- 0.383411
  - energy_ratio_m0: 11.427881 +/- 3.464842
  - energy_ratio_lr: 3.809279 +/- 0.004113

### class_0
- gen
  - response: 1.014252 +/- 0.029706
  - relative_error: 0.014252 +/- 0.029706
  - energy_ratio_pt: 2.215422 +/- 0.402082
  - energy_ratio_m0: 11.112924 +/- 2.970045
  - energy_ratio_lr: 4.057010 +/- 0.118824
- bicubic
  - response: 0.952609 +/- 0.000965
  - relative_error: -0.047391 +/- 0.000965
  - energy_ratio_pt: 2.084261 +/- 0.397498
  - energy_ratio_m0: 10.478934 +/- 2.976359
  - energy_ratio_lr: 3.810435 +/- 0.003859

### class_1
- gen
  - response: 1.005221 +/- 0.034310
  - relative_error: 0.005221 +/- 0.034310
  - energy_ratio_pt: 2.157390 +/- 0.378431
  - energy_ratio_m0: 13.043472 +/- 3.735916
  - energy_ratio_lr: 4.020886 +/- 0.137239
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