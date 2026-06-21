"""Dataset EDA — streams parquet files and emits a markdown + JSON report."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent))

from srgan import batch_to_tensor


@dataclass
class RunningMoments:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.total_sq += value * value

    def mean(self) -> float:
        return self.total / max(self.n, 1)

    def std(self) -> float:
        mean = self.mean()
        return float(max(self.total_sq / max(self.n, 1) - mean * mean, 0.0) ** 0.5)


def discover_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files


def summarize_dataset(data_dir: Path, batch_size: int = 512, max_events: int | None = None) -> dict:
    files = discover_files(data_dir)

    class_counts: Counter = Counter()
    pt_stats = RunningMoments()
    m0_stats = RunningMoments()
    lr_ratio_stats = RunningMoments()
    hr_ratio_stats = RunningMoments()
    lr_sparsity = RunningMoments()
    hr_sparsity = RunningMoments()
    lr_energy = RunningMoments()
    hr_energy = RunningMoments()
    pt_by_class: dict[int, RunningMoments] = defaultdict(RunningMoments)
    m0_by_class: dict[int, RunningMoments] = defaultdict(RunningMoments)
    response_by_class: dict[int, RunningMoments] = defaultdict(RunningMoments)

    inferred_lr_shape = None
    inferred_hr_shape = None
    seen_events = 0

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["X_jets_LR", "X_jets", "pt", "m0", "y"],
            use_threads=True,
        ):
            lr = batch_to_tensor(batch.column(0)).float()
            hr = batch_to_tensor(batch.column(1)).float()
            pt = torch.tensor(batch.column(2).to_pylist(), dtype=torch.float32)
            m0 = torch.tensor(batch.column(3).to_pylist(), dtype=torch.float32)
            y = torch.tensor(batch.column(4).to_pylist(), dtype=torch.int64)

            if inferred_lr_shape is None:
                inferred_lr_shape = tuple(lr.shape[1:])
            if inferred_hr_shape is None:
                inferred_hr_shape = tuple(hr.shape[1:])

            class_counts.update(int(v) for v in y.tolist())
            for v in pt.tolist():
                pt_stats.update(float(v))
            for v in m0.tolist():
                m0_stats.update(float(v))

            lr_sum = lr.sum(dim=(1, 2, 3))
            hr_sum = hr.sum(dim=(1, 2, 3))
            lr_nonzero = (lr != 0).float().mean(dim=(1, 2, 3))
            hr_nonzero = (hr != 0).float().mean(dim=(1, 2, 3))
            response = lr_sum / torch.clamp(hr_sum, min=1e-8)

            for v in lr_sum.tolist():
                lr_energy.update(float(v))
            for v in hr_sum.tolist():
                hr_energy.update(float(v))
            for v in response.tolist():
                lr_ratio_stats.update(float(v))
            for v in (hr_sum / torch.clamp(lr_sum, min=1e-8)).tolist():
                hr_ratio_stats.update(float(v))
            for v in lr_nonzero.tolist():
                lr_sparsity.update(float(v))
            for v in hr_nonzero.tolist():
                hr_sparsity.update(float(v))

            for cls in torch.unique(y).tolist():
                mask = y == int(cls)
                if not mask.any():
                    continue
                for v in pt[mask].tolist():
                    pt_by_class[int(cls)].update(float(v))
                for v in m0[mask].tolist():
                    m0_by_class[int(cls)].update(float(v))
                for v in response[mask].tolist():
                    response_by_class[int(cls)].update(float(v))

            seen_events += int(y.numel())
            if max_events is not None and seen_events >= max_events:
                break
        if max_events is not None and seen_events >= max_events:
            break

    total_events = sum(class_counts.values())
    summary: dict = {
        "files": [str(p) for p in files],
        "total_events": total_events,
        "class_counts": dict(sorted(class_counts.items())),
        "image_shapes": {"lr": inferred_lr_shape, "hr": inferred_hr_shape},
        "pt": {"mean": pt_stats.mean(), "std": pt_stats.std()},
        "m0": {"mean": m0_stats.mean(), "std": m0_stats.std()},
        "energy": {
            "lr_mean": lr_energy.mean(),
            "hr_mean": hr_energy.mean(),
            "lr_to_hr_ratio_mean": lr_ratio_stats.mean(),
            "hr_to_lr_ratio_mean": hr_ratio_stats.mean(),
        },
        "sparsity": {
            "lr_nonzero_fraction_mean": lr_sparsity.mean(),
            "hr_nonzero_fraction_mean": hr_sparsity.mean(),
        },
        "by_class": {},
    }
    for cls in sorted(class_counts):
        summary["by_class"][str(cls)] = {
            "count": class_counts[cls],
            "pt_mean": pt_by_class[cls].mean(),
            "m0_mean": m0_by_class[cls].mean(),
            "response_mean": response_by_class[cls].mean(),
        }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    class_ids = [str(k) for k in sorted(int(k) for k in summary["class_counts"].keys())]
    class_counts = [summary["class_counts"][int(k)] for k in class_ids]

    plt.figure(figsize=(5, 4))
    plt.bar(class_ids, class_counts, color="#4C78A8")
    plt.xlabel("Class label")
    plt.ylabel("Event count")
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_balance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["LR -> HR response"], [summary["energy"]["lr_to_hr_ratio_mean"]], color="#F58518")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Mean intensity ratio")
    plt.title("Average Low-Resolution to High-Resolution Response")
    plt.tight_layout()
    plt.savefig(fig_dir / "energy_response.png", dpi=180)
    plt.close()

    lines = [
        "# Dataset Analysis", "",
        f"Total events: {summary['total_events']}",
        f"LR shape: {summary['image_shapes']['lr']}",
        f"HR shape: {summary['image_shapes']['hr']}", "",
        "## Class balance", "", "| Class | Count |", "| --- | ---: |",
    ]
    for cls, count in sorted(summary["class_counts"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {cls} | {count} |")
    lines += [
        "", "## Global statistics", "",
        f"- pt mean/std: {summary['pt']['mean']:.4f} / {summary['pt']['std']:.4f}",
        f"- m0 mean/std: {summary['m0']['mean']:.4f} / {summary['m0']['std']:.4f}",
        f"- LR mean energy: {summary['energy']['lr_mean']:.4f}",
        f"- HR mean energy: {summary['energy']['hr_mean']:.4f}",
        f"- Mean LR/HR response: {summary['energy']['lr_to_hr_ratio_mean']:.4f}",
        f"- Mean LR nonzero fraction: {summary['sparsity']['lr_nonzero_fraction_mean']:.4f}",
        f"- Mean HR nonzero fraction: {summary['sparsity']['hr_nonzero_fraction_mean']:.4f}",
        "", "## By class", "",
    ]
    for cls, stats in summary["by_class"].items():
        lines += [
            f"### Class {cls}",
            f"- count: {stats['count']}",
            f"- pt mean: {stats['pt_mean']:.4f}",
            f"- m0 mean: {stats['m0_mean']:.4f}",
            f"- mean LR/HR response: {stats['response_mean']:.4f}", "",
        ]

    (out_dir / "DATASET_ANALYSIS.md").write_text("\n".join(lines))
    (out_dir / "dataset_analysis.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the jet image dataset")
    parser.add_argument("--data-dir", type=str, default="../datasets")
    parser.add_argument("--out-dir", type=str, default="reports/dataset_analysis")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    summary = summarize_dataset(Path(args.data_dir), batch_size=args.batch_size, max_events=args.max_events)
    write_report(summary, Path(args.out_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
