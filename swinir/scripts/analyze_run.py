#!/usr/bin/env python
"""Build a self-contained Markdown report for a single run.

Usage:
    python scripts/analyze_run.py --run-dir outputs/swinir_base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        print("No metrics rows.")
        return

    last = rows[-1]
    best_val = min(r.get("val/l1", float("inf")) for r in rows)

    md = []
    md.append(f"# Run report: {run_dir.name}\n")
    md.append(f"- Epochs run: {len(rows)}")
    md.append(f"- Best val L1: {best_val:.5f}")
    md.append(f"- Last val L1: {last.get('val/l1', 'n/a')}")
    md.append(f"- Last val PSNR: {last.get('val/psnr', 'n/a')}")
    md.append(f"- Last energy response mean: {last.get('val/energy_response_mean', 'n/a')}")
    md.append(f"- Last response gap (|class0-class1|): {last.get('val/response_gap', 'n/a')}")
    md.append("")
    md.append("## Per-epoch metrics")
    md.append("")
    md.append("| epoch | train/l1 | train/phys | val/l1 | val/psnr | val/resp |")
    md.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        md.append(
            f"| {r.get('epoch', '?')} "
            f"| {r.get('train/l1', 0):.5f} "
            f"| {r.get('train/phys', 0):.5f} "
            f"| {r.get('val/l1', 0):.5f} "
            f"| {r.get('val/psnr', 0):.2f} "
            f"| {r.get('val/energy_response_mean', 0):.4f} |"
        )

    out = run_dir / "REPORT.md"
    out.write_text("\n".join(md))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
