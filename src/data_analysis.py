#!/usr/bin/env python3
"""
Dataset analysis for data/merged.csv

Expected columns (exactly 3):
- Dataset   (name of dataset: Indian, Asian, Joint, Ultra)
- Text      (string)
- Annotated Value (encoded label; numeric OR categorical)

Outputs (created under data/data_analysis/):
- summary.json
- dataset_health.csv
- dataset_counts.csv
- overall_label_counts.csv
- label_counts_by_dataset.csv
- label_counts_by_dataset_wide.csv
- text_length_stats_overall_chars.csv
- text_length_stats_overall_words.csv
- text_length_stats_by_dataset.csv
- label_entropy_by_dataset.csv
- duplicates.csv (if any)
- plots/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import save_fig


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, outpath: Path) -> None:
    fig.tight_layout()
    save_fig(fig, outpath, bbox_inches="tight")
    plt.close(fig)


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return float("nan")
    p = counts / s
    return float(-(p * np.log2(p + 1e-12)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/merged.csv", help="Path to merged.csv")
    parser.add_argument("--outdir", type=str, default="data/data_analysis", help="Output directory")

    # Defaults match YOUR file, but can be overridden
    parser.add_argument("--dataset-col", type=str, default="Dataset", help="Dataset column name")
    parser.add_argument("--text-col", type=str, default="Text", help="Text column name")
    parser.add_argument("--label-col", type=str, default="Annotated Value", help="Label column name")

    # Optional: if there are too many labels, keep plots readable by taking top-K
    parser.add_argument("--topk-labels", type=int, default=30, help="Max distinct labels to plot (overall)")
    args = parser.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"

    _safe_mkdir(outdir)
    _safe_mkdir(plots_dir)

    if not inpath.exists():
        raise FileNotFoundError(f"Input not found: {inpath.resolve()}")

    df = pd.read_csv(inpath)

    # ---- Validate columns (allow your actual names via args) ----
    needed = [args.dataset_col, args.text_col, args.label_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"If your columns have different names, pass --dataset-col/--text-col/--label-col."
        )

    # Keep only the 3 columns (avoid surprises)
    df = df[needed].copy()
    df.rename(
        columns={args.dataset_col: "dataset", args.text_col: "text", args.label_col: "label"},
        inplace=True,
    )

    # ---- Basic cleaning (non-destructive) ----
    # Keep raw values
    df["dataset_raw"] = df["dataset"]
    df["text_raw"] = df["text"]
    df["label_raw"] = df["label"]

    # Normalize
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["text"] = df["text"].astype(str)

    # Normalize label as a CATEGORY string (works for numeric or non-numeric)
    df["label_str"] = df["label_raw"].astype(str).str.strip()

    # Attempt numeric parse (some analyses use numeric if available)
    df["label_num"] = pd.to_numeric(df["label_raw"], errors="coerce")

    # Text length features
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().map(len)

    # Missing / invalid stats
    n_total = int(len(df))
    n_missing_dataset = int(df["dataset_raw"].isna().sum())
    n_missing_text = int(df["text_raw"].isna().sum())
    n_empty_text = int((df["text"].fillna("").str.strip() == "").sum())
    n_missing_label = int(df["label_raw"].isna().sum())

    n_label_num_missing = int(df["label_num"].isna().sum())
    label_is_numeric_enough = (n_label_num_missing < n_total)  # at least one numeric label parsed

    # Duplicates (same dataset+text+label_raw)
    dup_mask = df.duplicated(subset=["dataset", "text", "label_str"], keep=False)
    dups = df[dup_mask].sort_values(["dataset", "label_str", "text"])
    if len(dups) > 0:
        dups.to_csv(outdir / "duplicates.csv", index=False)

    # ---- Counts ----
    dataset_counts = df["dataset"].value_counts(dropna=False).sort_index()

    # Use categorical label counts (always works)
    overall_label_counts = df["label_str"].value_counts(dropna=False)

    label_by_dataset = (
        df.groupby(["dataset", "label_str"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "count"], ascending=[True, False])
    )

    pivot_counts = (
        label_by_dataset.pivot_table(
            index="dataset",
            columns="label_str",
            values="count",
            fill_value=0,
            aggfunc="sum",
            dropna=False,
        )
        .sort_index()
    )

    # ---- Text-length stats ----
    def length_stats(x: pd.Series) -> dict:
        x = x.dropna()
        if len(x) == 0:
            return {"n": 0}
        return {
            "n": int(len(x)),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.percentile(x, 50)),
            "p75": float(np.percentile(x, 75)),
            "max": float(x.max()),
        }

    overall_text_stats = {
        "n_chars": length_stats(df["n_chars"]),
        "n_words": length_stats(df["n_words"]),
    }

    by_dataset_stats = []
    for ds, g in df.groupby("dataset"):
        by_dataset_stats.append(
            {
                "dataset": ds,
                **{f"chars_{k}": v for k, v in length_stats(g["n_chars"]).items()},
                **{f"words_{k}": v for k, v in length_stats(g["n_words"]).items()},
            }
        )
    text_stats_by_dataset_df = pd.DataFrame(by_dataset_stats)
    if not text_stats_by_dataset_df.empty and "dataset" in text_stats_by_dataset_df.columns:
        text_stats_by_dataset_df = text_stats_by_dataset_df.sort_values("dataset")

    # ---- Label entropy (how diverse labels are) ----
    entropy_rows = []
    for ds, g in df.groupby("dataset"):
        counts = g["label_str"].value_counts().to_numpy(dtype=float)
        ent = _shannon_entropy_from_counts(counts)
        entropy_rows.append({"dataset": ds, "label_entropy_bits": ent, "n": int(len(g))})

    entropy_df = pd.DataFrame(entropy_rows)
    if not entropy_df.empty and "dataset" in entropy_df.columns:
        entropy_df = entropy_df.sort_values("dataset")

    # ---- Save tables ----
    dataset_counts.rename("count").to_csv(outdir / "dataset_counts.csv")

    # overall_label_counts: save as two columns (label,count)
    overall_label_counts.to_frame(name="count").to_csv(outdir / "overall_label_counts.csv")

    label_by_dataset.to_csv(outdir / "label_counts_by_dataset.csv", index=False)
    pivot_counts.to_csv(outdir / "label_counts_by_dataset_wide.csv")

    pd.DataFrame([{
        "n_total": n_total,
        "n_missing_dataset": n_missing_dataset,
        "n_missing_text": n_missing_text,
        "n_empty_text": n_empty_text,
        "n_missing_label": n_missing_label,
        "n_duplicates_rows": int(len(dups)),
        "n_unique_datasets": int(df["dataset"].nunique(dropna=True)),
        "n_unique_labels": int(df["label_str"].nunique(dropna=True)),
        "n_unique_labels_numeric": int(df["label_num"].nunique(dropna=True)),
        "labels_numeric_parseable_rows": int(n_total - n_label_num_missing),
        "labels_are_numeric_like": bool(label_is_numeric_enough),
    }]).to_csv(outdir / "dataset_health.csv", index=False)

    pd.DataFrame([overall_text_stats["n_chars"]]).to_csv(outdir / "text_length_stats_overall_chars.csv", index=False)
    pd.DataFrame([overall_text_stats["n_words"]]).to_csv(outdir / "text_length_stats_overall_words.csv", index=False)
    text_stats_by_dataset_df.to_csv(outdir / "text_length_stats_by_dataset.csv", index=False)

    # Always create entropy file (even if empty for some reason)
    if not entropy_df.empty:
        entropy_df.to_csv(outdir / "label_entropy_by_dataset.csv", index=False)
    else:
        pd.DataFrame(columns=["dataset", "label_entropy_bits", "n"]).to_csv(
            outdir / "label_entropy_by_dataset.csv", index=False
        )

    # Also write a compact JSON summary
    summary = {
        "input": str(inpath),
        "rows": n_total,
        "datasets": dataset_counts.to_dict(),
        "labels_counts_top20": overall_label_counts.head(20).to_dict(),
        "missing": {
            "missing_dataset": n_missing_dataset,
            "missing_text": n_missing_text,
            "empty_text": n_empty_text,
            "missing_label": n_missing_label,
        },
        "duplicates_rows": int(len(dups)),
        "text_length_overall": overall_text_stats,
        "label_entropy_by_dataset": entropy_rows,
        "labels_numeric_parseable_rows": int(n_total - n_label_num_missing),
        "labels_numeric_parseable_share": float((n_total - n_label_num_missing) / max(n_total, 1)),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Plots ----

    # 1) Dataset sizes
    fig = plt.figure(figsize=(8, 4))
    plt.bar(dataset_counts.index.astype(str), dataset_counts.values)
    plt.title("Rows per dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Count")
    _save_fig(fig, plots_dir / "dataset_sizes.png")

    # 2) Overall label distribution (top-K for readability)
    topk = max(5, int(args.topk_labels))
    overall_top = overall_label_counts.head(topk)

    fig = plt.figure(figsize=(max(10, 0.35 * len(overall_top)), 4))
    x = overall_top.index.astype(str)
    y = overall_top.values
    plt.bar(x, y)
    plt.title(f"Overall label distribution (top {len(overall_top)})")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    _save_fig(fig, plots_dir / "label_distribution_overall_topk.png")

    # 3) Label distribution by dataset (stacked) for top-K labels overall
    top_labels = overall_top.index.tolist()
    pivot_top = pivot_counts.copy()
    keep_cols = [c for c in top_labels if c in pivot_top.columns]
    pivot_top = pivot_top[keep_cols] if len(keep_cols) > 0 else pd.DataFrame(index=pivot_counts.index)

    if pivot_top.shape[1] > 0:
        fig = plt.figure(figsize=(11, 5))
        bottom = np.zeros(len(pivot_top))
        for col in pivot_top.columns:
            vals = pivot_top[col].values
            plt.bar(pivot_top.index.astype(str), vals, bottom=bottom, label=str(col))
            bottom += vals
        plt.title(f"Label distribution by dataset (stacked, top {len(pivot_top.columns)} labels)")
        plt.xlabel("Dataset")
        plt.ylabel("Count")
        plt.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
        _save_fig(fig, plots_dir / "label_distribution_by_dataset_stacked_topk.png")

    # 4) Text length histograms (overall)
    fig = plt.figure(figsize=(9, 4))
    plt.hist(df["n_words"].dropna().values, bins=50)
    plt.title("Text length distribution (words) - overall")
    plt.xlabel("# words")
    plt.ylabel("Frequency")
    _save_fig(fig, plots_dir / "text_length_hist_words_overall.png")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(df["n_chars"].dropna().values, bins=50)
    plt.title("Text length distribution (chars) - overall")
    plt.xlabel("# chars")
    plt.ylabel("Frequency")
    _save_fig(fig, plots_dir / "text_length_hist_chars_overall.png")

    # 5) Text length by dataset (boxplot)
    ds_order = sorted(df["dataset"].dropna().unique().tolist())
    data_words = [df.loc[df["dataset"] == ds, "n_words"].dropna().values for ds in ds_order]
    fig = plt.figure(figsize=(10, 4))
    plt.boxplot(data_words, labels=ds_order, showfliers=False)
    plt.title("Text length (words) by dataset")
    plt.xlabel("Dataset")
    plt.ylabel("# words")
    _save_fig(fig, plots_dir / "text_length_box_words_by_dataset.png")

    # 6) Label entropy by dataset
    if not entropy_df.empty:
        fig = plt.figure(figsize=(8, 4))
        plt.bar(entropy_df["dataset"].astype(str), entropy_df["label_entropy_bits"].values)
        plt.title("Label entropy by dataset (higher = more diverse labels)")
        plt.xlabel("Dataset")
        plt.ylabel("Entropy (bits)")
        _save_fig(fig, plots_dir / "label_entropy_by_dataset.png")

    print(f"[OK] Saved analysis to: {outdir.resolve()}")
    print(f"[OK] Plots in: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
