#!/usr/bin/env python
"""Evaluate a checkpoint on a held-out split and write a JSON report.

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/<run>/checkpoints/best.pt \
        --data-dir ../datasets \
        --out reports/swinir/eval_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from jetsr_swin.utils import load_dotenv_from_repo

load_dotenv_from_repo(Path(__file__))

from jetsr_swin.data import (
    ChannelStats,
    ParquetJetSRDataset,
    denormalize,
    discover_parquet_files,
    normalize,
    split_files,
)
from jetsr_swin.evaluation.metrics import MetricAccumulator, energy_response, l1_raw, psnr, psnr_norm, ssim_2d
from jetsr_swin.models.swinir import SwinIRConfig, SwinIRGenerator
from jetsr_swin.utils import select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="../datasets")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--val-ratio", type=float, default=0.33)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = SwinIRConfig(**ckpt["model_cfg"])
    model = SwinIRGenerator(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = ChannelStats.from_dict(ckpt["stats"])

    files = discover_parquet_files(Path(args.data_dir))
    train_files, val_files, test_files = split_files(files, args.val_ratio, args.test_ratio)
    files_for_split = val_files if args.split == "val" else (test_files or val_files)

    ds = ParquetJetSRDataset(
        files_for_split,
        batch_size=args.batch_size,
        shuffle_files=False,
        shuffle_batches=False,
        batch_buffer_size=1,
        include_meta=model_cfg.use_film,
        max_batches=args.max_batches,
    )

    acc = MetricAccumulator()
    batch_limit = args.max_batches or "?"
    with torch.no_grad():
        for i, batch in enumerate(ds, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            lr_n = normalize(batch["lr"], stats)
            hr_n = normalize(batch["hr"], stats)
            meta = None
            if model_cfg.use_film:
                meta = {
                    "pt": (batch["pt"] - 116.9) / 60.0,
                    "m0": (batch["m0"] - 21.4) / 12.0,
                    "y": batch["y"],
                }
            pred_n = model(lr_n, meta=meta)
            pred_r = denormalize(pred_n, stats)
            hr_r = denormalize(hr_n, stats)
            bs = lr_n.shape[0]

            if i % 20 == 0 or i == 1:
                print(f"\r  batch {i}/{batch_limit} ...", end="", flush=True)

            acc.update("l1_norm", float((pred_n - hr_n).abs().mean().item()), bs)
            acc.update("l1_raw", l1_raw(pred_r, hr_r), bs)
            acc.update("psnr", psnr(pred_r, hr_r), bs)
            acc.update("psnr_norm", psnr_norm(pred_n, hr_n), bs)
            try:
                acc.update("ssim", ssim_2d(pred_r, hr_r), bs)
            except Exception:
                pass
            em, es = energy_response(pred_r, hr_r)
            acc.update("energy_response_mean", em, bs)
            acc.update("energy_response_std", es, bs)

            if "y" in batch:
                per_class = {}
                for cls in batch["y"].unique().tolist():
                    mask = batch["y"] == cls
                    if not mask.any():
                        continue
                    pe = pred_r[mask].sum(dim=(1, 2, 3))
                    te = hr_r[mask].sum(dim=(1, 2, 3))
                    r = (pe / torch.clamp(te, min=1e-8)).mean().item()
                    per_class[int(cls)] = (r, int(mask.sum().item()))
                acc.update_per_class("energy_response_mean", per_class)

    print()  # end the \r progress line
    summary = acc.summary()
    summary["checkpoint"] = args.checkpoint
    summary["split"] = args.split

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
