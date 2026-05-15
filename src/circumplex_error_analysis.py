#!/usr/bin/env python3
"""
Adds Schwartz circumplex distance scoring to misclassification CSVs and
summarizes the attractor structure for each population.

Inputs (produced by misclassification_joint_test.py):
  experiments/results/misclf_<dataset>_test_misclassified.csv
  experiments/results/misclf_<dataset>_test_confusion_matrix.csv

Outputs:
  experiments/results/misclf_<dataset>_test_misclassified_scored.csv
  experiments/results/misclf_<dataset>_test_attractor_summary.csv
  experiments/results/circumplex_summary.csv  (cross-population overview)

Schwartz circumplex (refined theory, 2012) — coarse labels in circular order:
  SD(0) -> ST(1) -> HE(2) -> AC(3) -> PO(4) -> FA(5) -> SE(6) ->
  TR(7) -> CO(8) -> HU(9) -> BE(10) -> UN(11) -> SD(0) ...

Modular distance d(i,j) = min(|i-j|, 12-|i-j|). Max distance = 6 (diametric).

Distance buckets used to flag example strength:
  1   adjacent          (likely annotation/conceptual overlap; weakest evidence)
  2-3 near / moderate   (moderate evidence of population-typical confusion)
  4-5 cross-axis        (strong evidence; crosses a Schwartz higher-order axis)
  6   diametric         (strongest evidence)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Schwartz refined-theory circumplex order (coarse, 12-class).
CIRCUMPLEX_ORDER: List[str] = [
    "SD", "ST", "HE", "AC", "PO", "FA",
    "SE", "TR", "CO", "HU", "BE", "UN",
]
N_COARSE = len(CIRCUMPLEX_ORDER)
INDEX_OF = {c: i for i, c in enumerate(CIRCUMPLEX_ORDER)}


def circumplex_distance(a: str, b: str) -> int:
    i, j = INDEX_OF[a], INDEX_OF[b]
    d = abs(i - j)
    return min(d, N_COARSE - d)


def distance_bucket(d: int) -> str:
    if d == 0:
        return "same"
    if d == 1:
        return "adjacent"
    if d in (2, 3):
        return "near"
    if d in (4, 5):
        return "cross-axis"
    return "diametric"


# Schwartz higher-order axes (per Schwartz 2012):
#   Self-enhancement      : AC, PO, (FA on boundary)
#   Self-transcendence    : BE, UN, (HU on boundary)
#   Conservation          : SE, TR, CO, (FA, HU on boundaries)
#   Openness to change    : SD, ST, HE
AXIS_OF: Dict[str, str] = {
    "SD": "openness", "ST": "openness", "HE": "openness",
    "AC": "self-enhancement", "PO": "self-enhancement", "FA": "self-enhancement",
    "SE": "conservation", "TR": "conservation", "CO": "conservation",
    "HU": "self-transcendence", "BE": "self-transcendence", "UN": "self-transcendence",
}


def axis_pair(a: str, b: str) -> str:
    return f"{AXIS_OF[a]}->{AXIS_OF[b]}"


def score_misclassified(in_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    if "true_coarse" not in df.columns or "pred_coarse" not in df.columns:
        raise ValueError(f"{in_csv} missing required columns")

    df["circumplex_distance"] = [
        circumplex_distance(t, p) for t, p in zip(df["true_coarse"], df["pred_coarse"])
    ]
    df["distance_bucket"] = df["circumplex_distance"].apply(distance_bucket)
    df["axis_pair"] = [axis_pair(t, p) for t, p in zip(df["true_coarse"], df["pred_coarse"])]
    df["crosses_axis"] = df.apply(
        lambda r: AXIS_OF[r["true_coarse"]] != AXIS_OF[r["pred_coarse"]], axis=1
    )

    df_sorted = df.sort_values(
        ["circumplex_distance", "pred_conf"], ascending=[False, False]
    ).reset_index(drop=True)
    df_sorted.to_csv(out_csv, index=False)
    return df_sorted


def attractor_summary(cm_csv: Path, out_csv: Path) -> pd.DataFrame:
    cm = pd.read_csv(cm_csv, index_col=0)
    rows = []
    for predicted in cm.columns:
        col = cm[predicted]
        diag = int(col.get(predicted, 0))
        offdiag = int(col.sum() - diag)
        n_supporting_classes = int((col > 0).sum() - (1 if diag > 0 else 0))
        rows.append({
            "predicted_class": predicted,
            "diagonal_correct": diag,
            "offdiag_errors_into": offdiag,
            "n_true_classes_contributing_errors": n_supporting_classes,
        })
    summary = pd.DataFrame(rows).sort_values(
        "offdiag_errors_into", ascending=False
    ).reset_index(drop=True)
    summary.to_csv(out_csv, index=False)
    return summary


def cross_population_overview(per_dataset: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for ds, scored_csv in per_dataset.items():
        df = pd.read_csv(scored_csv)
        total = len(df)
        if total == 0:
            continue
        bucket_counts = df["distance_bucket"].value_counts().to_dict()
        crosses = int(df["crosses_axis"].sum())
        rows.append({
            "dataset": ds,
            "n_misclassified": total,
            "adjacent_pct": 100 * bucket_counts.get("adjacent", 0) / total,
            "near_pct": 100 * bucket_counts.get("near", 0) / total,
            "cross_axis_pct": 100 * bucket_counts.get("cross-axis", 0) / total,
            "diametric_pct": 100 * bucket_counts.get("diametric", 0) / total,
            "crosses_higher_order_axis_pct": 100 * crosses / total,
            "mean_distance": df["circumplex_distance"].mean(),
            "median_distance": df["circumplex_distance"].median(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["joint", "ultra"])
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    scored_paths: Dict[str, Path] = {}

    for ds in args.datasets:
        mis_csv = results_dir / f"misclf_{ds}_{args.split}_misclassified.csv"
        cm_csv = results_dir / f"misclf_{ds}_{args.split}_confusion_matrix.csv"
        if not mis_csv.exists():
            print(f"[SKIP] {ds}: missing {mis_csv}")
            continue
        if not cm_csv.exists():
            print(f"[SKIP] {ds}: missing {cm_csv}")
            continue

        scored = results_dir / f"misclf_{ds}_{args.split}_misclassified_scored.csv"
        summary = results_dir / f"misclf_{ds}_{args.split}_attractor_summary.csv"

        df = score_misclassified(mis_csv, scored)
        att = attractor_summary(cm_csv, summary)

        print(f"\n=== {ds} ===")
        print(f"[OK] scored misclassified: {scored} (n={len(df)})")
        print(f"[OK] attractor summary:    {summary}")

        print("\nTop attractor classes (most errors received):")
        print(att.head(5).to_string(index=False))

        print("\nDistance bucket breakdown:")
        print(df["distance_bucket"].value_counts().to_string())

        print("\nCrosses higher-order axis:",
              f"{int(df['crosses_axis'].sum())}/{len(df)} "
              f"({100 * df['crosses_axis'].mean():.1f}%)")

        scored_paths[ds] = scored

    if scored_paths:
        overview = cross_population_overview(scored_paths)
        overview_csv = results_dir / "circumplex_summary.csv"
        overview.to_csv(overview_csv, index=False)
        print(f"\n[OK] cross-population overview: {overview_csv}")
        print(overview.to_string(index=False))


if __name__ == "__main__":
    main()
