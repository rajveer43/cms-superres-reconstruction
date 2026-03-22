# Results Summary

The final run is the best checkpoint for presentation.

## Best checkpoint

- Run directory: `outputs/final_run`
- Checkpoint: `outputs/final_run/checkpoints/best.pt`

## Comparison table

| Run | Val L1 | GAN response | Raw L1 | Raw PSNR | Comment |
| --- | ---: | ---: | ---: | ---: | --- |
| Final run | 0.09725 | 1.0098 | 0.00698 | 14.32 dB | Best balance overall |
| Adjusted run | 0.09907 | 1.0217 | 0.00709 | 14.29 dB | Slightly high response |
| Bicubic baseline | - | 0.9523 | 0.00846 | 14.13 dB | Lower fidelity |

## Interpretation

- The GAN improves raw-space L1 over bicubic.
- The response is very close to 1.0, which is the main physics constraint.
- The model still is not uniformly better than bicubic across every normalized metric.

## Best figures

- `outputs/final_run/plots/loss_curves.png`
- `outputs/final_run/plots/epoch_018_side_by_side.png`
- `outputs/final_run/plots/summary_panel.png`

## Presentation-ready conclusion

The final model is a credible end-to-end super-resolution baseline for this dataset:

- it trains stably
- it learns the jet image structure
- it preserves total intensity well
- it improves raw-space reconstruction over bicubic

The remaining limitation is that a single metric does not capture the full physics quality, so the presentation should state both the improvement and the residual gap.
