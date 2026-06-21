# Experiment Report

**Dataset:** QCDToGGQQ CMS Jet Images  
**Run directory:** `experiments/2026-06-12/parquet_run_01`  
**Platform:** device=mps num_workers=4 pin_memory=False use_amp=False kaggle=False colab=False

## Run command

```bash
python train_srgan.py \
    --dataset-type parquet \
    --hr-size 125 125 \
    --epochs 20 \
    --batch-size 64 \
    --output-dir experiments/2026-06-12/parquet_run_01
```

## Dataset

| Property | Value |
| --- | --- |
| Name | QCDToGGQQ CMS Jet Images |
| N train | N/A (streaming) |
| N val | 1000 |
| HR shape | (125, 125) |
| Scale factor | 2.0 |

## Training config

| Param | Value |
| --- | --- |
| epochs | 20 |
| batch_size | 64 |
| lr | 0.0002 |
| lambda_l1 | 50.0 |
| lambda_physics | 15.0 |
| gen_channels | 64 |
| gen_blocks | 8 |
| d_lr_ratio | 0.5 |
| n_critic | 1 |

## Best epoch

| Metric | Value |
| --- | --- |
| Best epoch | 20 |
| val_l1 | 0.07742 |
| val_psnr_norm | 2.57 dB |
| val_response | 0.9865 |

## Physics metrics

| Metric | GAN | Bicubic |
| --- | --- | --- |
| Response mean   | 0.9857 | 1.0875 |
| Response std    | 0.0325  | 0.0215  |
| Response median | 0.9781 | 1.0886 |
| |rel error| mean | 0.0288 | — |

### Per-class (QCD label)

| Class | N | GAN response μ | GAN response σ | Bicubic response μ |
| --- | --- | --- | --- | --- |
| class_0 | 521 | 0.9770 | 0.0280 | 1.0906 |
| class_1 | 479 | 0.9952 | 0.0344 | 1.0841 |

## Correlation metrics

| Metric | Value |
| --- | --- |
| Pearson r (pixel) | 0.6851 |
| SSIM mean | 0.9808 ± 0.0159 |
| Radial profile MSE | 0.0000 |
| Azimuthal profile MSE | 0.0000 |
| GAN sparsity | 0.828 |
| HR sparsity  | 0.983 |

## Figures

### Training
![Loss curves](figures/training/loss_curves.png)
![Response](figures/training/physics_response_curve.png)
![PSNR](figures/training/psnr_curve.png)

### Reconstruction
![Sample 0](figures/reconstruction/sample_grid_0.png)
![Mean shower](figures/reconstruction/mean_shower_comparison.png)
![Residual](figures/reconstruction/residual_map.png)

### Physics
![Response hist](figures/physics/energy_response_hist.png)
![Energy scatter](figures/physics/energy_scatter.png)
![Residual scatter](figures/physics/energy_scatter_residual.png)

### Correlations
![Radial](figures/correlations/radial_profile.png)
![Azimuthal](figures/correlations/azimuthal_profile.png)
![Pixel corr](figures/correlations/pixel_correlation.png)
![SSIM](figures/correlations/ssim_hist.png)
![Sparsity](figures/correlations/sparsity_comparison.png)
![Channel corr](figures/correlations/channel_correlation_heatmap.png)

## Known limitations

1. **Synthetic LR has no real noise model** — LR is bicubic downsampled; real detector noise not simulated.
2. **Single calorimeter layer replicated to 3 channels** (HDF5 only) — approximation of ECAL/HCAL/Tracks.
3. **Energy response not calibrated to 1.0** — physics loss drives it toward 1.0 but a post-hoc calibration step is needed for publication-level accuracy.