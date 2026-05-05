#!/usr/bin/env python3
"""
Driver for multi-seed fine-tuning.

Runs `fine_tune_dataset` for the cartesian product of (datasets) x (seeds).
Per-run output lands at experiments/results/<dataset>/seed_<seed>/ thanks to
the seed-aware layout introduced in src/train.py.

Typical usage (matches the paper's Combined+Joint multi-seed campaign):

    python -m src.train_multi_seed --datasets joint,combined --seeds 42,43,44

After all runs finish, aggregate with:

    python -m src.aggregate_seeds --datasets joint,combined
"""
from __future__ import annotations

import argparse
import time

from src.train import fine_tune_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--datasets",
        type=str,
        default="joint,combined",
        help="Comma-separated dataset names (default: joint,combined)",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated integer seeds (default: 42,43,44)",
    )
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    total = len(datasets) * len(seeds)
    print(f"Plan: {len(datasets)} dataset(s) x {len(seeds)} seed(s) = {total} fine-tuning run(s)")
    print(f"  datasets: {datasets}")
    print(f"  seeds:    {seeds}\n")

    t0 = time.time()
    for i, ds in enumerate(datasets, start=1):
        for j, seed in enumerate(seeds, start=1):
            run_idx = (i - 1) * len(seeds) + j
            print(f"\n=== [{run_idx}/{total}] dataset={ds} | seed={seed} ===")
            t_start = time.time()
            fine_tune_dataset(ds, seed=seed)
            print(f"=== [{run_idx}/{total}] done in {(time.time() - t_start) / 60:.1f} min ===")

    print(f"\n[ALL DONE] total wall-clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
