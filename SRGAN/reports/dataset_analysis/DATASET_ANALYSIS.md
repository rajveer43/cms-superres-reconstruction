# Dataset Analysis

Total events: 10240
Low-resolution image shape: (3, 64, 64)
High-resolution image shape: (3, 125, 125)

## Class balance

| Class | Count |
| --- | ---: |
| 0 | 5080 |
| 1 | 5160 |

## Global statistics

- pt mean/std: 116.7563 / 26.2634
- m0 mean/std: 21.2515 / 6.4842
- LR mean energy: 61.3277
- HR mean energy: 245.3109
- Mean LR/HR response: 0.2500
- Mean LR nonzero fraction: 0.0292
- Mean HR nonzero fraction: 0.0175

## By class

### Class 0
- count: 5080
- pt mean: 115.3106
- m0 mean: 23.0520
- mean LR/HR response: 0.2500

### Class 1
- count: 5160
- pt mean: 118.1796
- m0 mean: 19.4789
- mean LR/HR response: 0.2500

## Takeaways

- The dataset is heavily sparse, so a log-scale transform and energy-aware loss are justified.
- LR images systematically under-represent total energy relative to HR targets.
- The classes are balanced enough for a single conditional-free reconstruction model.
