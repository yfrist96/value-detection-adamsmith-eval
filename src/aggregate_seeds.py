#!/usr/bin/env python3
"""
Aggregate multi-seed fine-tuning results into mean +/- std per metric.

For each requested dataset, reads all `experiments/results/<dataset>/seed_*/metrics.csv`
files, extracts the LAST-epoch row of each (matches the headline-reporting
convention in the paper), and writes a per-dataset summary CSV plus prints a
mean/std table to stdout.

Usage:

    python -m src.aggregate_seeds --datasets joint,combined

Outputs:

    experiments/results/<dataset>_seed_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SKIP_COLS = {"epoch", "train_loss", "seed"}


def _seed_from_dirname(p: Path) -> Optional[int]:
    name = p.name
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def aggregate_one(dataset_name: str, results_root: Path) -> Optional[pd.DataFrame]:
    ds_root = results_root / dataset_name
    if not ds_root.exists():
        return None

    seed_dirs = sorted(p for p in ds_root.glob("seed_*") if p.is_dir() and _seed_from_dirname(p) is not None)
    if not seed_dirs:
        return None

    last_rows = []
    for seed_dir in seed_dirs:
        m = seed_dir / "metrics.csv"
        if not m.exists():
            continue
        df = pd.read_csv(m)
        if df.empty:
            continue
        last = df.iloc[-1].copy()
        last["seed"] = _seed_from_dirname(seed_dir)
        last_rows.append(last)

    if not last_rows:
        return None

    combined = pd.DataFrame(last_rows)
    seeds_used = sorted(int(s) for s in combined["seed"].dropna())

    metric_cols = [c for c in combined.columns if c not in SKIP_COLS]

    rows = []
    for c in metric_cols:
        vals = combined[c].dropna().astype(float)
        rows.append(
            {
                "metric": c,
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": int(len(vals)),
                "seeds": ",".join(str(s) for s in seeds_used),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--datasets",
        type=str,
        default="joint,combined",
        help="Comma-separated dataset names (default: joint,combined)",
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Root results dir containing <dataset>/seed_*/metrics.csv",
    )
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    any_written = False
    for ds in datasets:
        summary = aggregate_one(ds, results_root)
        if summary is None:
            print(f"[skip] {ds}: no seed_* subdirs found under {results_root / ds}")
            continue

        out_path = results_root / f"{ds}_seed_summary.csv"
        summary.to_csv(out_path, index=False)
        any_written = True

        print(f"\n=== {ds} (n_seeds={summary['n_seeds'].iloc[0]}, seeds={summary['seeds'].iloc[0]}) ===")
        # Format mean/std as "X.XXX +/- Y.YYY" for readability
        display = summary.copy()
        display["mean ± std"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(display["mean"], display["std"])]
        print(display[["metric", "mean ± std", "n_seeds"]].to_string(index=False))
        print(f"[OK] Saved {out_path}")

    if not any_written:
        print("\n[warn] No summaries written. Did you run training with multiple seeds first?")
        print("       Try: python -m src.train_multi_seed --datasets joint,combined --seeds 42,43,44")


if __name__ == "__main__":
    main()
