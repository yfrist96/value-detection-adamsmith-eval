#!/usr/bin/env python3
"""
Aggregate the per-setting achievement-vocabulary ablation runs into one table.

`src.ablate_achievement_vocab` writes a `summary.json` per run (one per setting:
the base Adam-Smith model plus each per-population fine-tuned checkpoint). This
script reads the `global` slice of each `summary.json` and collates the headline
cross-setting numbers (AC-prediction share and F1 deltas before/after masking)
into a single CSV, matching the table referenced in the paper.

Usage (run after the five ablation runs in README step 9):

    python -m src.aggregate_ablation

Outputs:

    experiments/results/ablation_summary_per_setting.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


# Default settings, in paper order: base model + the four per-population
# fine-tuned checkpoints. Each name is the `--output_label` used for that
# ablation run (see README step 9), and the directory under --results_dir.
DEFAULT_LABELS = [
    "ablation_achievement",
    "ablation_achievement_asian_finetuned",
    "ablation_achievement_indian_finetuned",
    "ablation_achievement_ijh_finetuned",
    "ablation_achievement_ultra_finetuned",
]


def _setting_name(label: str) -> str:
    """Human-readable setting name from an ablation output label."""
    if label == "ablation_achievement":
        return "base"
    pop = label.removeprefix("ablation_achievement_").removesuffix("_finetuned")
    return "IJH" if pop == "ijh" else pop.capitalize()


def _row_from_summary(label: str, summary: dict) -> dict:
    """Pull the headline cross-setting numbers from one run's summary.json."""
    g = summary["global"]
    ac_share_delta = g["ac_pred_share_masked"] - g["ac_pred_share_orig"]
    macro_f1_delta = g["macro_f1_masked"] - g["macro_f1_orig"]
    ac_f1_delta = g["ac_f1_masked"] - g["ac_f1_orig"]
    return {
        "setting": _setting_name(label),
        "label": label,
        "n": g["n"],
        "n_rows_with_match": summary.get("n_rows_with_match"),
        "ac_pred_share_orig": g["ac_pred_share_orig"],
        "ac_pred_share_masked": g["ac_pred_share_masked"],
        # Percentage-point shift in AC-prediction share (the headline number).
        "ac_pred_share_delta_pp": ac_share_delta * 100.0,
        "ac_f1_orig": g["ac_f1_orig"],
        "ac_f1_masked": g["ac_f1_masked"],
        "ac_f1_delta": ac_f1_delta,
        "macro_f1_orig": g["macro_f1_orig"],
        "macro_f1_masked": g["macro_f1_masked"],
        "macro_f1_delta": macro_f1_delta,
        "mcnemar_ac_pvalue": g["mcnemar_ac_pvalue"],
    }


def aggregate_ablation(labels: list[str], results_root: Path) -> Optional[pd.DataFrame]:
    rows = []
    for label in labels:
        summary_path = results_root / label / "summary.json"
        if not summary_path.exists():
            print(f"[skip] {label}: no summary.json at {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(_row_from_summary(label, summary))

    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated ablation output labels to collate (default: the five paper settings)",
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Root results dir containing <label>/summary.json",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <results_dir>/ablation_summary_per_setting.csv)",
    )
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    summary = aggregate_ablation(labels, results_root)
    if summary is None:
        print("\n[warn] No ablation summaries found. Run the ablation settings first:")
        print("       see README step 9 (python -m src.ablate_achievement_vocab ...)")
        return

    out_path = Path(args.output) if args.output else results_root / "ablation_summary_per_setting.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    display = summary.copy()
    display["AC-share Δpp"] = [f"{v:+.1f}" for v in display["ac_pred_share_delta_pp"]]
    display["Macro-F1 Δ"] = [f"{v:+.3f}" for v in display["macro_f1_delta"]]
    print("\n=== Achievement-vocabulary ablation: cross-setting summary ===")
    print(display[["setting", "AC-share Δpp", "Macro-F1 Δ", "mcnemar_ac_pvalue"]].to_string(index=False))
    print(f"\n[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
