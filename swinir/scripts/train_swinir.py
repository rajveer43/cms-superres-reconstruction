#!/usr/bin/env python
"""Train SwinIR (or SwinIR+FiLM) for jet image super-resolution.

Usage:
    python scripts/train_swinir.py --config configs/swinir_base.yaml
    python scripts/train_swinir.py --config configs/swinir_film.yaml --smoke
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

# Allow running as `python scripts/...` without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from jetsr_swin.models.swinir import SwinIRConfig
from jetsr_swin.training import TrainConfig, Trainer
from jetsr_swin.utils import load_dotenv_from_repo


def _filter_fields(d: dict, dataclass_cls) -> dict:
    valid = {f.name for f in fields(dataclass_cls)}
    return {k: v for k, v in d.items() if k in valid}


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text())
    model_raw = raw.pop("model", {})
    model_cfg = SwinIRConfig(**_filter_fields(model_raw, SwinIRConfig))
    train_kwargs = _filter_fields(raw, TrainConfig)
    return TrainConfig(model=model_cfg, **train_kwargs)


def apply_smoke(cfg: TrainConfig) -> TrainConfig:
    cfg.epochs = 1
    cfg.max_train_batches = 3
    cfg.max_val_batches = 2
    cfg.max_stats_batches = 2
    cfg.batch_size = 2
    cfg.stats_batch_size = 4
    cfg.wandb_enabled = False
    cfg.amp = False
    cfg.log_every_n_steps = 1
    return cfg


def main() -> None:
    env_path = load_dotenv_from_repo(Path(__file__))
    if env_path:
        print(f"[env] loaded {env_path}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true", help="Tiny run for sanity, no W&B")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.no_wandb:
        cfg.wandb_enabled = False
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device

    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
