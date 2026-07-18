from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data import (
    ChannelStats,
    ParquetJetSRDataset,
    denormalize,
    discover_parquet_files,
    normalize,
    split_files,
    stream_channel_stats,
)
from ..evaluation.metrics import MetricAccumulator, energy_response, l1_raw, psnr, ssim_2d
from ..evaluation.plots import sample_panel
from ..logging import ActivationGradientTracker, build_logger, log_attention_maps, log_rel_pos_bias
from ..losses import CombinedLoss
from ..models.swinir import SwinIRConfig, SwinIRGenerator
from ..utils import git_sha, save_yaml, select_device, set_seed, supports_amp


@dataclass
class TrainConfig:
    # Paths
    data_dir: str = "../datasets"
    output_dir: str = "outputs/swinir_run"

    # Data
    val_ratio: float = 0.33
    test_ratio: float = 0.0
    batch_size: int = 8
    stats_batch_size: int = 32
    max_stats_batches: int | None = 10
    batch_buffer_size: int = 8
    max_train_batches: int | None = None
    max_val_batches: int | None = 20

    # Optim
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    lambda_l1: float = 50.0
    lambda_phys: float = 12.0

    # Misc
    seed: int = 42
    device: str | None = None  # auto
    amp: bool = True

    # Model
    model: SwinIRConfig = field(default_factory=SwinIRConfig)
    include_meta: bool = False  # set True for FiLM

    # Logging
    wandb_enabled: bool = True
    wandb_project: str | None = None
    run_name: str = "swinir_base"
    log_every_n_steps: int = 20
    log_attention_every_n_epochs: int = 1
    log_samples_every_n_epochs: int = 1
    hook_target_substrings: tuple[str, ...] = ("blocks.0", "blocks.1", "blocks.2", "blocks.3")


def _cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)

        self.device = select_device(cfg.device)
        self.use_amp = cfg.amp and supports_amp(self.device)

        # Output dirs
        self.out_dir = Path(cfg.output_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.sample_dir = self.out_dir / "samples"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Data splits
        files = discover_parquet_files(Path(cfg.data_dir))
        train_files, val_files, test_files = split_files(files, cfg.val_ratio, cfg.test_ratio)
        self.train_files, self.val_files, self.test_files = train_files, val_files, test_files

        # Normalization stats
        self.stats = stream_channel_stats(
            train_files, batch_size=cfg.stats_batch_size, max_batches=cfg.max_stats_batches
        )
        save_yaml({"normalization": self.stats.to_dict()}, self.out_dir / "normalization.yaml")

        # Model
        self.model = SwinIRGenerator(cfg.model).to(self.device)
        self.loss_fn = CombinedLoss(lambda_l1=cfg.lambda_l1, lambda_phys=cfg.lambda_phys).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay
        )

        # Scheduler: stepped per training batch. We estimate steps per epoch from max_train_batches.
        steps_per_epoch = cfg.max_train_batches if cfg.max_train_batches else 200
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch
        self.scheduler = _cosine_with_warmup(self.optimizer, total_steps, warmup_steps)

        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp) if self.use_amp else None

        # Datasets
        self.train_ds = ParquetJetSRDataset(
            train_files,
            batch_size=cfg.batch_size,
            shuffle_files=True,
            shuffle_batches=True,
            batch_buffer_size=cfg.batch_buffer_size,
            include_meta=cfg.include_meta,
            max_batches=cfg.max_train_batches,
        )
        self.val_ds = ParquetJetSRDataset(
            val_files,
            batch_size=cfg.batch_size,
            shuffle_files=False,
            shuffle_batches=False,
            batch_buffer_size=1,
            include_meta=cfg.include_meta,
            max_batches=cfg.max_val_batches,
        )

        # Logger
        def _sanitize(v):
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, (list, tuple)):
                return [_sanitize(x) for x in v]
            if isinstance(v, dict):
                return {k: _sanitize(x) for k, x in v.items()}
            return str(v)

        wandb_config = {
            **{k: _sanitize(getattr(cfg, k)) for k in vars(cfg) if k != "model"},
            "model": _sanitize(vars(cfg.model)),
            "git_sha": git_sha(),
            "device": str(self.device),
            "torch_version": str(torch.__version__),
            "amp": self.use_amp,
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
            "n_test_files": len(test_files),
        }
        self.logger = build_logger(
            enabled=cfg.wandb_enabled,
            project=cfg.wandb_project,
            run_name=cfg.run_name,
            config=wandb_config,
            output_dir=self.out_dir,
        )
        if self.logger.enabled:
            self.logger.watch(self.model, log_freq=cfg.log_every_n_steps * 5)

        # Hooks for activation/gradient L2 norms per Swin block
        self.tracker = ActivationGradientTracker(self.model, target_substrings=cfg.hook_target_substrings)

        # Persist config snapshot
        save_yaml(wandb_config, self.out_dir / "config.yaml")

        self.global_step = 0
        self.best_val_l1 = math.inf

    # ----- batch helpers -----

    def _to_device_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _meta(self, batch: dict[str, Tensor]) -> dict[str, Tensor] | None:
        if not self.cfg.include_meta:
            return None
        # Cheap normalization for FiLM input. Means/stds from analyze_dataset.py.
        pt_mean, pt_std = 116.9, 60.0
        m0_mean, m0_std = 21.4, 12.0
        return {
            "pt": (batch["pt"] - pt_mean) / pt_std,
            "m0": (batch["m0"] - m0_mean) / m0_std,
            "y": batch["y"],
        }

    # ----- main loop -----

    def fit(self) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch)

            epoch_log = {**train_metrics, **{f"val/{k}": v for k, v in val_metrics.items()}, "epoch": epoch, "epoch_time_s": time.time() - t0}
            self.logger.log(epoch_log, step=self.global_step)

            # Print one summary line per epoch
            msg = (
                f"epoch {epoch:03d}  "
                f"train_l1={train_metrics.get('train/l1', 0):.5f}  "
                f"train_phys={train_metrics.get('train/phys', 0):.5f}  "
                f"val_l1={val_metrics.get('l1', 0):.5f}  "
                f"val_psnr={val_metrics.get('psnr', 0):.2f}  "
                f"resp={val_metrics.get('energy_response_mean', 0):.4f}  "
                f"time={epoch_log['epoch_time_s']:.1f}s"
            )
            print(msg)
            (self.out_dir / "metrics.jsonl").open("a").write(json.dumps(epoch_log) + "\n")

            # Save checkpoints
            ckpt = self._build_ckpt(epoch)
            torch.save(ckpt, self.ckpt_dir / f"epoch_{epoch:03d}.pt")
            if val_metrics["l1"] < self.best_val_l1:
                self.best_val_l1 = val_metrics["l1"]
                torch.save(ckpt, self.ckpt_dir / "best.pt")

        self.logger.finish()

    def _build_ckpt(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "stats": self.stats.to_dict(),
            "model_cfg": vars(self.cfg.model),
            "train_cfg": {k: getattr(self.cfg, k) for k in vars(self.cfg) if k != "model"},
        }

    # ----- one epoch -----

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = defaultdict(float)
        seen = 0

        for step, batch in enumerate(self.train_ds, start=1):
            batch = self._to_device_batch(batch)
            lr_norm = normalize(batch["lr"], self.stats)
            hr_norm = normalize(batch["hr"], self.stats)
            meta = self._meta(batch)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                pred = self.model(lr_norm, meta=meta, store_attn=False)
                loss, components = self.loss_fn(pred, hr_norm, self.stats)

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            bs = lr_norm.shape[0]
            seen += bs
            for k, v in components.items():
                running[k] += float(v.item()) * bs

            self.global_step += 1
            if self.global_step % self.cfg.log_every_n_steps == 0:
                grad_norm = self._grad_norm()
                hook_stats = self.tracker.flush()
                payload = {
                    "train/loss": float(loss.item()),
                    "train/l1_step": float(components["l1"].item()),
                    "train/phys_step": float(components["phys"].item()),
                    "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                    "train/grad_norm": grad_norm,
                }
                for mod_name, stats in hook_stats.items():
                    for k, v in stats.items():
                        payload[f"internals/{mod_name}/{k}"] = v
                self.logger.log(payload, step=self.global_step)

        return {f"train/{k}": v / max(1, seen) for k, v in running.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        acc = MetricAccumulator()
        sample_batch_for_panel: dict[str, Tensor] | None = None
        sample_batch_for_attn: dict[str, Tensor] | None = None

        for batch in self.val_ds:
            batch = self._to_device_batch(batch)
            lr_norm = normalize(batch["lr"], self.stats)
            hr_norm = normalize(batch["hr"], self.stats)
            meta = self._meta(batch)
            pred_norm = self.model(lr_norm, meta=meta, store_attn=False)

            # Compute per-batch metrics
            pred_raw = denormalize(pred_norm, self.stats)
            hr_raw = denormalize(hr_norm, self.stats)
            bs = lr_norm.shape[0]

            acc.update("l1", float((pred_norm - hr_norm).abs().mean().item()), bs)
            acc.update("l1_raw", l1_raw(pred_raw, hr_raw), bs)
            acc.update("psnr", psnr(pred_raw, hr_raw), bs)
            try:
                acc.update("ssim", ssim_2d(pred_raw, hr_raw), bs)
            except Exception:
                pass
            er_mean, er_std = energy_response(pred_raw, hr_raw)
            acc.update("energy_response_mean", er_mean, bs)
            acc.update("energy_response_std", er_std, bs)

            # Per-class slice
            if "y" in batch:
                per_class: dict[int, tuple[float, int]] = {}
                for cls in batch["y"].unique().tolist():
                    mask = batch["y"] == cls
                    if not mask.any():
                        continue
                    pe = pred_raw[mask].sum(dim=(1, 2, 3))
                    te = hr_raw[mask].sum(dim=(1, 2, 3))
                    r = (pe / torch.clamp(te, min=1e-8)).mean().item()
                    per_class[int(cls)] = (r, int(mask.sum().item()))
                acc.update_per_class("energy_response_mean", per_class)

            if sample_batch_for_panel is None:
                sample_batch_for_panel = batch
            if sample_batch_for_attn is None:
                sample_batch_for_attn = batch

        # Sample panel logging
        if (
            sample_batch_for_panel is not None
            and self.cfg.log_samples_every_n_epochs > 0
            and epoch % self.cfg.log_samples_every_n_epochs == 0
        ):
            lr_norm = normalize(sample_batch_for_panel["lr"], self.stats)
            hr_norm = normalize(sample_batch_for_panel["hr"], self.stats)
            meta = self._meta(sample_batch_for_panel)
            pred_norm = self.model(lr_norm, meta=meta, store_attn=True)
            pred_raw = denormalize(pred_norm, self.stats)
            hr_raw = denormalize(hr_norm, self.stats)
            fig = sample_panel(sample_batch_for_panel["lr"], pred_raw, hr_raw, sample_idx=0)
            self.logger.log_image("samples/val_panel", fig, step=self.global_step)

            if epoch % max(1, self.cfg.log_attention_every_n_epochs) == 0:
                log_attention_maps(self.model, self.logger, step=self.global_step)
                log_rel_pos_bias(self.model, self.logger, step=self.global_step)

        return acc.summary()

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().float().norm().item() ** 2)
        return total ** 0.5
