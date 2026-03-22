# Dataset Analysis

Total events: 2048
Low-resolution image shape: (3, 64, 64)
High-resolution image shape: (3, 125, 125)

## Class balance

| Class | Count |
| --- | ---: |
| 0 | 1042 |
| 1 | 1006 |

## Global statistics

- pt mean/std: 116.8989 / 26.8183
- m0 mean/std: 21.4095 / 6.7025
- LR mean energy: 61.0800
- HR mean energy: 244.3201
- Mean LR/HR response: 0.2500
- Mean LR nonzero fraction: 0.0292
- Mean HR nonzero fraction: 0.0175

## By class

### Class 0
- count: 1042
- pt mean: 115.4765
- m0 mean: 23.1585
- mean LR/HR response: 0.2500

### Class 1
- count: 1006
- pt mean: 118.3721
- m0 mean: 19.5979
- mean LR/HR response: 0.2500

## Takeaways

- The dataset is heavily sparse, so a log-scale transform and energy-aware loss are justified.
- LR images systematically under-represent total energy relative to HR targets.
- The classes are balanced enough for a single conditional-free reconstruction model.
