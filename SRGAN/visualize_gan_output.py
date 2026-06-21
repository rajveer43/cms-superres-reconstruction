"""Render a per-channel LR -> GAN -> HR comparison grid from a saved .npz sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CHANNELS = ("ECAL", "HCAL", "Tracks")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=str, default="outputs/tuned_run/samples/epoch_006.npz")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", type=str, default="reports/gan_output_viz.png")
    args = p.parse_args()

    data = np.load(args.sample)
    lr = data["lr"][args.index]
    fake = data["fake"][args.index]
    hr = data["hr"][args.index]

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    cols = [("LR input", lr), ("GAN output", fake), ("HR target", hr)]

    for r, ch in enumerate(CHANNELS):
        vmax = max(np.log1p(fake[r]).max(), np.log1p(hr[r]).max(), 1e-6)
        for c, (title, img) in enumerate(cols):
            ax = axes[r, c]
            channel = img[r] if r < img.shape[0] else img[0]
            im = ax.imshow(np.log1p(channel), cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"{title}\n{ch}  (E={channel.sum():.1f})", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("GAN jet-image super-resolution (log1p energy)", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")
    print(f"LR {lr.shape} -> fake {fake.shape} | HR {hr.shape}")
    print(f"energy: LR={lr.sum():.1f}  fake={fake.sum():.1f}  HR={hr.sum():.1f}  "
          f"response={fake.sum() / max(hr.sum(), 1e-6):.3f}")


if __name__ == "__main__":
    main()
