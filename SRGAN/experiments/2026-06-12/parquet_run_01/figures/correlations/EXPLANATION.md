# Correlation Analysis and Structural Validation

## 1. Overview

This folder contains the structural validation plots for the parquet run. These figures check whether the GAN preserves the internal geometry of the shower, inter-channel relationships, sparsity, and image-level similarity.

The goal here is not just to match total energy, but to reproduce the way energy is distributed across detector coordinates and detector components.

## 2. Row Profile

**Figure Title**: Row (Radial) Profile (`radial_profile.png`)

![Row radial profile comparing HR, GAN, and bicubic reconstructions.](radial_profile.png)

**What the plot shows**  
This curve shows the mean deposited energy as a function of row bin, which corresponds to a radial or vertical shower profile depending on the detector mapping.

**How to read the plot**  
The peak position indicates where the shower core sits. The width and tails show how far the energy spreads away from the core.

**Interpretation**  
The GAN curve tracks the HR profile closely at the peak and through the tails. Bicubic is also close in the broad shape, but the GAN better preserves the sharp peak and local structure around the maximum.

**Physics meaning**  
This means the model retains the shower's radial development, which is important for measuring shower width, containment, and particle-dependent shower shape.

## 3. Column Profile

**Figure Title**: Column (Azimuthal) Profile (`azimuthal_profile.png`)

![Column azimuthal profile comparing HR, GAN, and bicubic reconstructions.](azimuthal_profile.png)

**What the plot shows**  
This curve shows the mean deposited energy as a function of column bin, i.e. the azimuthal profile.

**How to read the plot**  
The central spike and surrounding shoulders show how the shower is structured across the azimuthal axis. Matching the position and sharpness of those features is the key criterion.

**Interpretation**  
The GAN reproduces the central spike and the adjacent falloff very well. It stays close to the HR curve across the full range, while bicubic is smoother and slightly less faithful to the narrow peak.

**Physics meaning**  
Preserving the azimuthal structure matters because it encodes how localized the energy deposition is in the detector plane.

## 4. Channel Correlation

**Figure Title**: Channel Correlation Heatmap (`channel_correlation_heatmap.png`)

![Channel correlation heatmaps for HR channels and GAN channels.](channel_correlation_heatmap.png)

**What the plot shows**  
This figure compares the correlation matrices for the HR detector channels and the GAN-reconstructed channels.

**How to read the plot**  
Similar off-diagonal structure between the two matrices means the GAN preserved how energy in one layer relates to energy in the others.

**Interpretation**  
The GAN correlation matrix is nearly identical to the HR matrix. The same positive and negative relationships between ECAL, HCAL, and Tracks are present after reconstruction.

**Physics meaning**  
This indicates that the model preserves longitudinal shower development and cross-layer dependencies rather than reconstructing each channel independently.

## 5. Pixel Correlation

**Figure Title**: Pixel Correlation (`pixel_correlation.png`)

![Pixel-level correlation between HR and GAN energy values.](pixel_correlation.png)

**What the plot shows**  
Each point compares a single detector cell in HR against the corresponding cell in the GAN output.

**How to read the plot**  
Points close to the diagonal indicate accurate pixel-level reconstruction. Scatter near zero energy is normal because those cells are dominated by sparsity and small fluctuations.

**Interpretation**  
The points cluster tightly around the diagonal, which means the GAN is learning a cell-by-cell mapping that is strongly correlated with the HR target.

**Physics meaning**  
This is the local consistency check: if the model is accurate at the pixel level, it is much more likely to preserve detailed shower morphology and downstream observables.

## 6. Sparsity Comparison

**Figure Title**: Sparsity Comparison (`sparsity_comparison.png`)

![Sparsity comparison between LR upsampled, GAN output, and HR target.](sparsity_comparison.png)

**What the plot shows**  
This bar chart shows the fraction of detector cells with nearly zero energy for LR, GAN, and HR.

**How to read the plot**  
Higher sparsity means more empty cells, which is expected in calorimeter images. A reconstructed image should move toward the HR sparsity level rather than filling the detector with weak noise.

**Interpretation**  
The GAN is much closer to the HR sparsity than the LR baseline, even if it does not fully match the true empty-cell fraction. That means it suppresses many non-physical activations introduced by interpolation.

**Physics meaning**  
Matching sparsity is important because calorimeter showers occupy only a small region of the detector, and unnecessary low-level activation can distort cluster finding and shape variables.

## 7. SSIM Distribution

**Figure Title**: Per-sample SSIM (`ssim_hist.png`)

![Per-sample SSIM distribution for GAN versus HR.](ssim_hist.png)

**What the plot shows**  
This histogram shows the distribution of SSIM values computed between each GAN reconstruction and its HR target.

**How to read the plot**  
Values closer to 1 indicate better structural similarity. The mean and spread give a compact summary of reconstruction consistency across the dataset.

**Interpretation**  
The SSIM values are concentrated in a narrow high-quality band, with a mean close to 0.98. That suggests the GAN reconstructs most events with consistently strong structural fidelity.

**Physics meaning**  
This is a useful summary metric because it reflects whether the model preserves the global shape and local contrast of showers, not just integrated energy.

## 8. Overall Assessment

Together, these plots show that the GAN:

* Preserves the mean radial and azimuthal shower profiles.
* Reproduces inter-channel correlations accurately.
* Maintains strong pixel-level agreement with HR.
* Improves sparsity relative to the LR baseline.
* Achieves high structural similarity event by event.

That is the right combination for physically meaningful calorimeter super-resolution.
