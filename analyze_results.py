from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from train_srgan import (
    ChannelStats,
    Generator,
    batch_to_tensor,
    denormalize,
    discover_parquet_files,
    normalize,
    split_files,
)


def load_metrics(metrics_path: Path) -> list[dict]:
    metrics = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            metrics.append(json.loads(line))
    if not metrics:
        raise RuntimeError(f"No metrics found in {metrics_path}")
    return metrics


def plot_loss_curves(metrics: list[dict], out_path: Path) -> None:
    epochs = [m["epoch"] for m in metrics]
    train_g = [m["train_g_loss"] for m in metrics]
    train_d = [m["train_d_loss"] for m in metrics]
    train_l1 = [m["train_l1"] for m in metrics]
    val_l1 = [m["val_l1"] for m in metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_g, marker="o", label="train G loss")
    plt.plot(epochs, train_d, marker="o", label="train D loss")
    plt.plot(epochs, train_l1, marker="o", label="train L1")
    plt.plot(epochs, val_l1, marker="o", label="val L1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def collapse_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW array, got {arr.shape}")
    return arr.sum(axis=0)


def resize_for_display(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.tensor(image, dtype=torch.float32)[None, None, :, :]
    resized = F.interpolate(tensor, size=shape, mode="bicubic", align_corners=False)
    return resized[0, 0].numpy()


def make_side_by_side_figure(sample_path: Path, out_path: Path) -> None:
    batch = np.load(sample_path)
    lr = batch["lr"][0]
    fake = batch["fake"][0]
    hr = batch["hr"][0]

    lr_img = resize_for_display(collapse_image(lr), hr.shape[1:])
    bi_img = resize_for_display(collapse_image(lr), hr.shape[1:])
    gan_img = collapse_image(fake)
    hr_img = collapse_image(hr)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    titles = ["LR (upsampled)", "Bicubic", "GAN", "HR"]
    images = [lr_img, bi_img, gan_img, hr_img]
    for ax, title, img in zip(axes, titles, images):
        im = ax.imshow(img, cmap="magma", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(sample_path.stem, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_summary_panel(metrics: list[dict], sample_path: Path, physics: dict, out_path: Path) -> None:
    batch = np.load(sample_path)
    lr = batch["lr"][0]
    fake = batch["fake"][0]
    hr = batch["hr"][0]

    epochs = [m["epoch"] for m in metrics]
    train_l1 = [m["train_l1"] for m in metrics]
    val_l1 = [m["val_l1"] for m in metrics]

    lr_img = resize_for_display(collapse_image(lr), hr.shape[1:])
    bi_img = resize_for_display(collapse_image(lr), hr.shape[1:])
    gan_img = collapse_image(fake)
    hr_img = collapse_image(hr)

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.85])

    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(epochs, train_l1, marker="o", label="train L1")
    ax_loss.plot(epochs, val_l1, marker="o", label="val L1")
    ax_loss.set_title("Training / Validation L1")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    sample_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for ax, title, img in zip(sample_axes, ["LR (upsampled)", "Bicubic", "GAN", "HR"], [lr_img, bi_img, gan_img, hr_img]):
        im = ax.imshow(img, cmap="magma", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax_resp = fig.add_subplot(gs[2, :])
    groups = ["overall", "class_0", "class_1"]
    x = np.arange(len(groups))
    width = 0.35
    gen_vals = [physics[g]["gen"]["response"]["mean"] for g in groups]
    bi_vals = [physics[g]["bicubic"]["response"]["mean"] for g in groups]
    ax_resp.bar(x - width / 2, gen_vals, width, label="GAN")
    ax_resp.bar(x + width / 2, bi_vals, width, label="Bicubic")
    ax_resp.set_xticks(x)
    ax_resp.set_xticklabels(groups)
    ax_resp.set_ylabel("Response = generated / target intensity")
    ax_resp.set_title("Physics Response Summary")
    ax_resp.grid(True, axis="y", alpha=0.3)
    ax_resp.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_stats(stats_path: Path) -> ChannelStats:
    raw = json.loads(stats_path.read_text())
    return ChannelStats(mean=torch.tensor(raw["mean"]), std=torch.tensor(raw["std"]))


@dataclass
class RunningStats:
    total: float = 0.0
    total_sq: float = 0.0
    n: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.total_sq += value * value
        self.n += 1

    def mean(self) -> float:
        return self.total / max(self.n, 1)

    def std(self) -> float:
        mean = self.mean()
        return math.sqrt(max(self.total_sq / max(self.n, 1) - mean * mean, 0.0))


def analyze_physics(
    ckpt_path: Path,
    data_dir: Path,
    val_ratio: float,
    batch_size: int,
    max_batches: int,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stats = ChannelStats(mean=torch.tensor(ckpt["stats"]["mean"]), std=torch.tensor(ckpt["stats"]["std"]))

    generator = Generator()
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    files = discover_parquet_files(data_dir)
    _, val_files = split_files(files, val_ratio)

    class_stats: dict[str, dict[str, dict[str, RunningStats]]] = defaultdict(
        lambda: {"gen": defaultdict(RunningStats), "bicubic": defaultdict(RunningStats)}
    )
    overall: dict[str, dict[str, RunningStats]] = {"gen": defaultdict(RunningStats), "bicubic": defaultdict(RunningStats)}

    def update_group(model: str, group: str, pred_energy: float, hr_energy: float, lr_energy: float, pt: float, m0: float) -> None:
        for name, val in {
            "response": pred_energy / max(hr_energy, 1e-8),
            "relative_error": (pred_energy - hr_energy) / max(hr_energy, 1e-8),
            "energy_ratio_pt": pred_energy / max(pt, 1e-8),
            "energy_ratio_m0": pred_energy / max(m0, 1e-8),
            "energy_ratio_lr": pred_energy / max(lr_energy, 1e-8),
        }.items():
            class_stats[group][model][name].update(val)
            overall[model][name].update(val)

    with torch.no_grad():
        batches_seen = 0
        for path in val_files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=["X_jets_LR", "X_jets", "pt", "m0", "y"],
                use_threads=True,
            ):
                lr = batch_to_tensor(batch.column(0))
                hr = batch_to_tensor(batch.column(1))
                pt = torch.tensor(batch.column(2).to_pylist(), dtype=torch.float32)
                m0 = torch.tensor(batch.column(3).to_pylist(), dtype=torch.float32)
                y = torch.tensor(batch.column(4).to_pylist(), dtype=torch.int64)

                lr_n = normalize(lr, stats)
                hr_n = normalize(hr, stats)
                fake_n = generator(lr_n, hr_n.shape[-2:])
                fake = denormalize(fake_n, stats)
                bicubic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

                for i in range(hr.shape[0]):
                    group = f"class_{int(y[i].item())}"
                    hr_energy = float(hr[i].sum().item())
                    lr_energy = float(lr[i].sum().item())
                    update_group("gen", group, float(fake[i].sum().item()), hr_energy, lr_energy, float(pt[i].item()), float(m0[i].item()))
                    update_group("bicubic", group, float(bicubic[i].sum().item()), hr_energy, lr_energy, float(pt[i].item()), float(m0[i].item()))

                batches_seen += 1
                if batches_seen >= max_batches:
                    break
            if batches_seen >= max_batches:
                break

    output: dict[str, dict] = {"overall": {"gen": {}, "bicubic": {}}}
    for model in ["gen", "bicubic"]:
        for name, stat in overall[model].items():
            output["overall"][model][name] = {"mean": stat.mean(), "std": stat.std()}

    for group, models in class_stats.items():
        output[group] = {"gen": {}, "bicubic": {}}
        for model in ["gen", "bicubic"]:
            for name, stat in models[model].items():
                output[group][model][name] = {"mean": stat.mean(), "std": stat.std()}

    output["validation_batches"] = batches_seen
    return output


def build_report(run_dir: Path, physics: dict, metrics: list[dict]) -> str:
    last = metrics[-1]
    lines = [
        "# Run Analysis",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Training curves",
        "",
        f"- Final train G loss: {last['train_g_loss']:.6f}",
        f"- Final train D loss: {last['train_d_loss']:.6f}",
        f"- Final train L1: {last['train_l1']:.6f}",
        f"- Final val L1: {last['val_l1']:.6f}",
        "",
        "## Results table",
        "",
        "| Group | GAN response | Bicubic response | GAN relative error | Bicubic relative error |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Overall | {physics['overall']['gen']['response']['mean']:.4f} | {physics['overall']['bicubic']['response']['mean']:.4f} | {physics['overall']['gen']['relative_error']['mean']:.4f} | {physics['overall']['bicubic']['relative_error']['mean']:.4f} |",
        f"| class_0 | {physics['class_0']['gen']['response']['mean']:.4f} | {physics['class_0']['bicubic']['response']['mean']:.4f} | {physics['class_0']['gen']['relative_error']['mean']:.4f} | {physics['class_0']['bicubic']['relative_error']['mean']:.4f} |",
        f"| class_1 | {physics['class_1']['gen']['response']['mean']:.4f} | {physics['class_1']['bicubic']['response']['mean']:.4f} | {physics['class_1']['gen']['relative_error']['mean']:.4f} | {physics['class_1']['bicubic']['relative_error']['mean']:.4f} |",
        "",
        "## Physics proxies",
        "",
    ]
    for group_name in ["overall"] + sorted(k for k in physics.keys() if k.startswith("class_")):
        if group_name == "validation_batches":
            continue
        lines.append(f"### {group_name}")
        for model in ["gen", "bicubic"]:
            lines.append(f"- {model}")
            for metric_name, values in physics[group_name][model].items():
                lines.append(f"  - {metric_name}: {values['mean']:.6f} +/- {values['std']:.6f}")
        lines.append("")
    lines += [
        "Notes:",
        "- `response` is the ratio of reconstructed total image intensity to the target HR intensity.",
        "- `energy_ratio_pt` and `energy_ratio_m0` are image-intensity proxies normalized by event metadata.",
        "- `class_0` / `class_1` are the label values in the parquet `y` column.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots and physics-metric analysis for a training run")
    parser.add_argument("--run-dir", type=str, default="outputs/results_run")
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--val-ratio", type=float, default=0.33)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(run_dir / "metrics.jsonl")
    plot_loss_curves(metrics, plots_dir / "loss_curves.png")

    sample_dir = run_dir / "samples"
    sample_paths = sorted(sample_dir.glob("epoch_*.npz"))
    for sample_path in sample_paths:
        make_side_by_side_figure(sample_path, plots_dir / f"{sample_path.stem}_side_by_side.png")

    physics = analyze_physics(
        ckpt_path=run_dir / "checkpoints" / "best.pt",
        data_dir=Path(args.data_dir),
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    if sample_paths:
        make_summary_panel(metrics, sample_paths[-1], physics, plots_dir / "summary_panel.png")

    (run_dir / "physics_metrics.json").write_text(json.dumps(physics, indent=2))
    (run_dir / "ANALYSIS.md").write_text(build_report(run_dir, physics, metrics))

    print(json.dumps({"plots_dir": str(plots_dir), "physics_metrics": physics}, indent=2))


if __name__ == "__main__":
    main()
