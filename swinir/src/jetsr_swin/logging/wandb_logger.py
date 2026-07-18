from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class NullLogger:
    """No-op logger so training code never has to None-check."""

    def log(self, data: dict, step: int | None = None) -> None: ...
    def log_image(self, name: str, fig, step: int | None = None) -> None: ...
    def watch(self, model, log_freq: int = 200) -> None: ...
    def finish(self) -> None: ...

    @property
    def enabled(self) -> bool:
        return False


class WandbLogger:
    """Thin wrapper around wandb with safe Matplotlib figure handling."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict,
        output_dir: Path,
        mode: str = "online",
    ) -> None:
        if wandb is None:
            raise RuntimeError("wandb is not installed. `pip install wandb` or set logger.enabled=False")
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            dir=str(output_dir),
            mode=mode,
            reinit=True,
        )

    def log(self, data: dict, step: int | None = None) -> None:
        wandb.log(data, step=step)

    def log_image(self, name: str, fig, step: int | None = None) -> None:
        wandb.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def watch(self, model, log_freq: int = 200) -> None:
        wandb.watch(model, log="all", log_freq=log_freq)

    def finish(self) -> None:
        wandb.finish()

    @property
    def enabled(self) -> bool:
        return True


def build_logger(
    enabled: bool,
    project: str | None,
    run_name: str,
    config: dict,
    output_dir: Path,
) -> "WandbLogger | NullLogger":
    if not enabled:
        print("[wandb] logging disabled (config: wandb_enabled=false)")
        return NullLogger()
    if wandb is None:
        raise RuntimeError(
            "wandb_enabled=true but the `wandb` package is not installed. "
            "Run `pip install wandb` or pass --no-wandb to skip logging."
        )
    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb_enabled=true and WANDB_MODE=online, but WANDB_API_KEY is not set. "
            "Put it in swinir/.env, run `wandb login`, set WANDB_MODE=offline, "
            "or pass --no-wandb."
        )
    project = project or os.environ.get("WANDB_PROJECT", "jetsr-swin")
    print(f"[wandb] init project={project} run={run_name} mode={mode}")
    return WandbLogger(
        project=project, run_name=run_name, config=config, output_dir=output_dir, mode=mode
    )
