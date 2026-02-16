#!/usr/bin/env python3
"""
Cross-Domain Generalization Heatmap (from metrics.csv)

This script reads per-run CSV logs created by src/train.py:

  experiments/results/<train_dataset>/metrics.csv

and produces a Macro-F1 heatmap:
  rows = train dataset
  cols = evaluation dataset
  value = macro F1 from the LAST epoch in the CSV

Outputs:
- experiments/plots/cross_domain_macro_f1_heatmap.png
- experiments/results/cross_domain_macro_f1_matrix.csv
- experiments/results/cross_domain_macro_f1_matrix.json

Notes:
- Your current train.py logs macro-F1 columns like:
    in_test_f1
    ood_asian_f1, ood_indian_f1, ood_joint_f1, ood_ultra_f1
  and also micro versions (not used here).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_last_epoch_row(csv_path: Path) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    # assume epochs increase
    return df.iloc[-1]


def _extract_macro_scores_from_row(train_ds: str, row: pd.Series, datasets: List[str]) -> Dict[str, float]:
    """
    Returns mapping eval_dataset -> macro_f1 for last epoch.

    Uses:
      - in_test_f1 for in-domain (train_ds -> train_ds)
      - ood_<dataset>_f1 for out-of-domain cells
    """
    out: Dict[str, float] = {}

    # In-domain macro F1
    if "in_test_f1" in row and pd.notna(row["in_test_f1"]):
        out[train_ds] = float(row["in_test_f1"])

    # OOD macro F1
    for d in datasets:
        if d == train_ds:
            continue
        col = f"ood_{d}_f1"
        if col in row and pd.notna(row[col]):
            out[d] = float(row[col])

    return out


def build_macro_matrix(results_root: Path, datasets: List[str]) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Builds matrix M where M[i,j] is macro-F1 when trained on datasets[i] and evaluated on datasets[j].
    Uses last epoch from each train dataset's metrics.csv.

    Returns:
      (matrix, raw_dict)
    """
    idx = {ds: i for i, ds in enumerate(datasets)}
    mat = np.full((len(datasets), len(datasets)), np.nan, dtype=float)
    raw: Dict[str, Dict[str, float]] = {}

    for train_ds in datasets:
        csv_path = results_root / train_ds / "metrics.csv"
        if not csv_path.exists():
            continue

        last = _read_last_epoch_row(csv_path)
        if last is None:
            continue

        scores = _extract_macro_scores_from_row(train_ds, last, datasets)
        raw[train_ds] = scores

        i = idx[train_ds]
        for eval_ds, f1 in scores.items():
            j = idx[eval_ds]
            mat[i, j] = f1

    return mat, raw


def save_matrix_csv(path: Path, datasets: List[str], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train\\eval"] + datasets)
        for i, row_ds in enumerate(datasets):
            row = [row_ds] + ["" if np.isnan(x) else f"{x:.4f}" for x in mat[i]]
            w.writerow(row)


def plot_heatmap(out_path: Path, datasets: List[str], mat: np.ndarray, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 7))
    ax = plt.gca()

    im = ax.imshow(mat, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Evaluation dataset")
    ax.set_ylabel("Train dataset")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="experiments/results",
                   help="Root results dir containing <dataset>/metrics.csv")
    p.add_argument("--plots_dir", type=str, default="experiments/plots",
                   help="Where to save plots")
    p.add_argument("--out_dir", type=str, default="experiments/results",
                   help="Where to save matrix CSV/JSON")
    p.add_argument("--datasets", type=str, default="asian,indian,joint,ultra",
                   help="Comma-separated dataset order for heatmap")
    args = p.parse_args()

    results_root = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    out_dir = Path(args.out_dir)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    if not results_root.exists():
        raise SystemExit(f"[ERROR] results_dir not found: {results_root}")

    mat, raw = build_macro_matrix(results_root, datasets)

    # sanity: did we populate anything?
    if not np.isfinite(mat).any():
        missing = [ds for ds in datasets if not (results_root / ds / "metrics.csv").exists()]
        raise SystemExit(
            "[ERROR] No usable metrics.csv found / no F1 columns found.\n"
            f"Expected files like: experiments/results/<dataset>/metrics.csv\n"
            f"Missing metrics.csv for: {missing}\n"
            "Also confirm your metrics.csv contains columns: in_test_f1 and ood_<dataset>_f1"
        )

    out_csv = out_dir / "cross_domain_macro_f1_matrix.csv"
    out_json = out_dir / "cross_domain_macro_f1_matrix.json"
    plot_png = plots_dir / "cross_domain_macro_f1_heatmap.png"

    save_matrix_csv(out_csv, datasets, mat)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "datasets": datasets,
                "macro_f1_matrix": [[None if np.isnan(x) else float(x) for x in row] for row in mat],
                "raw_scores": raw,
                "note": "Rows=train datasets, Cols=eval datasets. Values taken from LAST epoch of each experiments/results/<train>/metrics.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_heatmap(plot_png, datasets, mat, "Cross-Domain Generalization (Macro-F1) — Last Epoch")

    print(f"[OK] Saved heatmap: {plot_png}")
    print(f"[OK] Saved matrix CSV: {out_csv}")
    print(f"[OK] Saved matrix JSON: {out_json}")


if __name__ == "__main__":
    main()
