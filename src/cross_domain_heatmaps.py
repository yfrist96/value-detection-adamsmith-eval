#!/usr/bin/env python3
"""
Cross-Domain Generalization Heatmap (from metrics.csv)

This script reads per-run CSV logs created by src/train.py:

  experiments/results/<train_dataset>/metrics.csv

and produces a Macro-F1 heatmap:
  rows = train dataset (+ an extra "base" row for epoch 0 / no fine-tuning)
  cols = evaluation dataset
  value =
    - for "base": macro F1 from epoch 0 (pre-fine-tuning baseline)
    - for fine-tuned rows: macro F1 from the LAST epoch in the CSV

Outputs:
- experiments/plots/cross_domain_macro_f1_heatmap.png
- experiments/results/cross_domain_macro_f1_matrix.csv
- experiments/results/cross_domain_macro_f1_matrix.json

Notes:
- train.py logs macro-F1 columns like:
    in_test_f1
    ood_asian_f1, ood_indian_f1, ood_joint_f1, ood_ultra_f1
  and also micro versions (not used here).

- The "base" row is constructed from epoch 0 rows. We fill each dataset’s base
  in-domain cell from its own metrics.csv epoch 0 in_test_f1 (diagonal), and use
  ood_<d>_f1 columns where available to fill off-diagonals.
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

from src.utils import save_fig


def _read_last_epoch_row(csv_path: Path) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[-1]


def _read_epoch_row(csv_path: Path, epoch: int) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or "epoch" not in df.columns:
        return None

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    hit = df[df["epoch"] == epoch]
    if hit.empty:
        return None
    return hit.iloc[0]


def _extract_macro_scores_from_row(train_ds: str, row: pd.Series, datasets: List[str]) -> Dict[str, float]:
    """
    Returns mapping eval_dataset -> macro_f1 for a fine-tuned run row.

    Uses:
      - in_test_f1 for in-domain (train_ds -> train_ds)
      - ood_<dataset>_f1 for out-of-domain cells
    """
    out: Dict[str, float] = {}

    if "in_test_f1" in row and pd.notna(row["in_test_f1"]):
        out[train_ds] = float(row["in_test_f1"])

    for d in datasets:
        if d == train_ds:
            continue
        col = f"ood_{d}_f1"
        if col in row and pd.notna(row[col]):
            out[d] = float(row[col])

    return out


def _extract_ood_macro_scores_from_row(row: pd.Series, datasets: List[str]) -> Dict[str, float]:
    """
    Reads any available ood_<dataset>_f1 columns from a row (used for base row construction).
    """
    out: Dict[str, float] = {}
    for d in datasets:
        col = f"ood_{d}_f1"
        if col in row and pd.notna(row[col]):
            out[d] = float(row[col])
    return out


def build_macro_matrix(results_root: Path, datasets: List[str]) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], List[str]]:
    """
    Builds matrix M where:
      - columns are eval datasets in `datasets`
      - rows are ["base"] + datasets
      - values are macro-F1

    base row:
      uses epoch 0 (pre-fine-tuning) metrics
    fine-tuned rows:
      use LAST epoch metrics
    """
    row_labels = ["base"] + datasets
    col_labels = datasets

    row_idx = {r: i for i, r in enumerate(row_labels)}
    col_idx = {c: j for j, c in enumerate(col_labels)}

    mat = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
    raw: Dict[str, Dict[str, float]] = {}

    # -------------------------
    # Row 0: base (epoch = 0)
    # -------------------------
    base_scores: Dict[str, float] = {}

    # First, fill diagonal base scores (each dataset's in_test_f1 at epoch 0)
    for ds in datasets:
        csv_path = results_root / ds / "metrics.csv"
        if not csv_path.exists():
            continue
        r0 = _read_epoch_row(csv_path, epoch=0)
        if r0 is None:
            continue
        if "in_test_f1" in r0 and pd.notna(r0["in_test_f1"]):
            base_scores[ds] = float(r0["in_test_f1"])

    # Next, try to fill off-diagonals using any available ood_* columns from epoch 0 rows
    # We'll scan epoch0 rows from all runs and keep the max coverage we can.
    for src_ds in datasets:
        csv_path = results_root / src_ds / "metrics.csv"
        if not csv_path.exists():
            continue
        r0 = _read_epoch_row(csv_path, epoch=0)
        if r0 is None:
            continue
        ood_scores = _extract_ood_macro_scores_from_row(r0, datasets)
        # merge without overwriting existing (diagonal) values unless missing
        for k, v in ood_scores.items():
            base_scores.setdefault(k, v)

    raw["base"] = base_scores
    i_base = row_idx["base"]
    for eval_ds, f1 in base_scores.items():
        if eval_ds in col_idx:
            mat[i_base, col_idx[eval_ds]] = f1

    # ----------------------------------
    # Fine-tuned rows: last epoch per run
    # ----------------------------------
    for train_ds in datasets:
        csv_path = results_root / train_ds / "metrics.csv"
        if not csv_path.exists():
            continue

        last = _read_last_epoch_row(csv_path)
        if last is None:
            continue

        scores = _extract_macro_scores_from_row(train_ds, last, datasets)
        raw[train_ds] = scores

        i = row_idx[train_ds]
        for eval_ds, f1 in scores.items():
            j = col_idx[eval_ds]
            mat[i, j] = f1

    return mat, raw, row_labels


def save_matrix_csv(path: Path, row_labels: List[str], col_labels: List[str], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train\\eval"] + col_labels)
        for i, row_name in enumerate(row_labels):
            row = [row_name] + ["" if np.isnan(x) else f"{x:.4f}" for x in mat[i]]
            w.writerow(row)


def plot_heatmap(out_path: Path, row_labels: List[str], col_labels: List[str], mat: np.ndarray, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.8, 7.2))
    ax = plt.gca()

    im = ax.imshow(mat, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
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
    save_fig(fig, out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Root results dir containing <dataset>/metrics.csv",
    )
    p.add_argument(
        "--plots_dir",
        type=str,
        default="experiments/plots",
        help="Where to save plots",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/results",
        help="Where to save matrix CSV/JSON",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="asian,indian,joint,ultra",
        help="Comma-separated dataset order for heatmap columns (and fine-tuned rows)",
    )
    args = p.parse_args()

    results_root = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    out_dir = Path(args.out_dir)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    if not results_root.exists():
        raise SystemExit(f"[ERROR] results_dir not found: {results_root}")

    mat, raw, row_labels = build_macro_matrix(results_root, datasets)

    if not np.isfinite(mat).any():
        missing = [ds for ds in datasets if not (results_root / ds / "metrics.csv").exists()]
        raise SystemExit(
            "[ERROR] No usable metrics.csv found / no F1 columns found.\n"
            f"Expected files like: experiments/results/<dataset>/metrics.csv\n"
            f"Missing metrics.csv for: {missing}\n"
            "Also confirm your metrics.csv contains columns: epoch, in_test_f1 and ood_<dataset>_f1"
        )

    out_csv = out_dir / "cross_domain_macro_f1_matrix.csv"
    out_json = out_dir / "cross_domain_macro_f1_matrix.json"
    plot_png = plots_dir / "cross_domain_macro_f1_heatmap.png"

    save_matrix_csv(out_csv, row_labels, datasets, mat)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "row_labels": row_labels,
                "col_labels": datasets,
                "macro_f1_matrix": [[None if np.isnan(x) else float(x) for x in row] for row in mat],
                "raw_scores": raw,
                "note": (
                    "Rows=[base + train datasets], Cols=eval datasets. "
                    "base row uses epoch 0 (no fine-tuning). "
                    "Other rows use LAST epoch of each experiments/results/<train>/metrics.csv."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_heatmap(plot_png, row_labels, datasets, mat, "Cross-Domain Generalization (Macro-F1) — Base + Last Epoch")

    print(f"[OK] Saved heatmap: {plot_png}")
    print(f"[OK] Saved matrix CSV: {out_csv}")
    print(f"[OK] Saved matrix JSON: {out_json}")


if __name__ == "__main__":
    main()
