#!/usr/bin/env python3
"""
Score the advisor-supplied few-shot LLM prediction dumps and compare them against the
fine-tuned transfer matrix.

The prediction CSVs live in experiments/fewshot/ and carry one row per (source_row, seed,
target, item), where `source_row` is the population the in-context exemplars were drawn from
(or "zeroshot" for no exemplars) and `target` is the population whose test split is being
scored. They are advisor-owned interim data and are gitignored.

Scoring matches src/eval.py exactly: macro F1 averaged over the union of gold and predicted
classes, which is the convention behind every cell of the fine-tuned matrix. Micro F1 is
reported alongside because the union-macro denominator rewards a model for merely touching
minority classes, and on this data that difference is large enough to manufacture an apparent
effect: one exemplar source looks best everywhere under macro and the advantage disappears
under micro. Read both before believing any ordering of the source rows.

The fine-tuned reference matrices are rebuilt from the committed per-seed metric logs under
experiments/results/, so this script needs no checkpoints and no GPU.

Usage:
    python -m src.score_fewshot                 # all models in experiments/fewshot/
    python -m src.score_fewshot --model qwen3:14b
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from sklearn.metrics import f1_score

FEWSHOT_DIR = pathlib.Path("experiments/fewshot")
TARGETS = ["asian", "indian", "ijh", "ultra"]
SOURCE_ORDER = ["zeroshot", "asian", "indian", "ijh", "ultra", "combined"]
POPULATIONS = ["asian", "indian", "ijh", "ultra"]

# Fine-tuned reference matrix (macro F1, union convention, seed means).
# Source: experiments/results/cross_domain_macro_f1_matrix.csv.
FINETUNED = pd.DataFrame(
    {"asian":  [0.3956, 0.8080, 0.7398, 0.4167, 0.4622, 0.8026],
     "indian": [0.3363, 0.6683, 0.8470, 0.3054, 0.4429, 0.8311],
     "ijh":    [0.1905, 0.2911, 0.2932, 0.4254, 0.2544, 0.4298],
     "ultra":  [0.3221, 0.4371, 0.4050, 0.1776, 0.5921, 0.5739]},
    index=["base", "asian", "indian", "ijh", "ultra", "combined"],
)

RESULTS_ROOT = pathlib.Path("experiments/results")
# Ultra seed 43 diverged at epoch 1 and never fit its own training data; excluded everywhere
# (see src/aggregate_seeds.py and Section 3 of the paper). The criterion is training-set
# performance and never consults the test set.
EXCLUDED_RUNS = {("ultra", 43)}


def finetuned_matrix(metric_suffix: str) -> pd.DataFrame:
    """Rebuild the fine-tuned transfer matrix from the per-seed training logs.

    `metric_suffix` is "" for macro F1 or "_micro" for micro. Each cell is the mean over the
    converged seeds of the LAST-epoch value, which is the convention aggregate_seeds.py uses
    and the one behind every published cell. The macro matrix produced here is checked against
    the hardcoded FINETUNED constant, so the two independent paths have to agree before any
    number is used. The `base` row lives only in the un-tuned evaluation and is not rebuilt.
    """
    rows = {}
    for pop in ["asian", "indian", "ijh", "ultra", "combined"]:
        seed_dirs = [d for d in sorted((RESULTS_ROOT / pop).glob("seed_*"))
                     if (pop, int(d.name.split("_")[1])) not in EXCLUDED_RUNS]
        cells: dict[str, list[float]] = {}
        for d in seed_dirs:
            last = pd.read_csv(d / "metrics.csv").iloc[-1]
            cells.setdefault(pop, []).append(last[f"in_test_f1{metric_suffix}"])
            for target in TARGETS:
                col = f"ood_{target}_f1{metric_suffix}"
                if target != pop and col in last.index:
                    cells.setdefault(target, []).append(last[col])
        rows[pop] = {t: sum(v) / len(v) for t, v in cells.items()}
    return pd.DataFrame(rows).T.reindex(columns=TARGETS)


def finetuned_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Macro and micro fine-tuned matrices, with the macro one verified against FINETUNED."""
    macro = finetuned_matrix("")
    check = FINETUNED.drop(index="base")
    if not ((macro - check).abs() < 5e-5).all().all():
        raise SystemExit("[ERROR] rebuilt fine-tuned macro matrix disagrees with FINETUNED:\n"
                         f"{(macro - check).round(4).to_string()}")
    return macro, finetuned_matrix("_micro")


def score(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (source, target) with seed-averaged macro/micro F1 and the macro std."""
    rows = []
    for (source, target), cell in df.groupby(["source_row", "target"]):
        macro, micro = [], []
        for _, per_seed in cell.groupby("seed"):
            gold, pred = per_seed["gold"], per_seed["pred"]
            labels = sorted(set(gold) | set(pred))  # matches src/eval.py
            macro.append(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))
            micro.append(f1_score(gold, pred, average="micro", zero_division=0))
        rows.append({"source": source, "target": target,
                     "macro": sum(macro) / len(macro), "micro": sum(micro) / len(micro),
                     "macro_std": pd.Series(macro).std(ddof=1), "n_seeds": len(macro)})
    return pd.DataFrame(rows)


def matrix(per_cell: pd.DataFrame, metric: str) -> pd.DataFrame:
    m = per_cell.pivot(index="source", columns="target", values=metric)
    return m.reindex([s for s in SOURCE_ORDER if s in m.index])[TARGETS]


def source_spread(m: pd.DataFrame) -> pd.Series:
    """Best minus worst single-population source per target column.

    This is the quantity that separates the few-shot and fine-tuned regimes: it measures how
    much the choice of source population moves performance on a fixed target.
    """
    sub = m.reindex(POPULATIONS)
    return sub.max() - sub.min()


def report(name: str, df: pd.DataFrame) -> dict:
    per_cell = score(df)
    macro, micro = matrix(per_cell, "macro"), matrix(per_cell, "micro")

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print("\nmacro F1 (union convention, seed mean)")
    print(macro.round(3).to_string())
    print("\nmicro F1 (seed mean)")
    print(micro.round(3).to_string())

    ft_spread = source_spread(FINETUNED)
    fs_spread = source_spread(macro)
    print("\nsource spread per target column (best minus worst single-population source)")
    print(f"  fine-tuned  " + "  ".join(f"{t}={ft_spread[t]:.3f}" for t in TARGETS)
          + f"   mean={ft_spread.mean():.3f}")
    print(f"  few-shot    " + "  ".join(f"{t}={fs_spread[t]:.3f}" for t in TARGETS)
          + f"   mean={fs_spread.mean():.3f}")

    print("\nin-domain: fine-tuned vs few-shot")
    for p in POPULATIONS:
        print(f"  {p:7s} FT {FINETUNED.loc[p, p]:.3f}   FS {macro.loc[p, p]:.3f}   "
              f"gap {FINETUNED.loc[p, p] - macro.loc[p, p]:+.3f}")

    # Restricted to the four single-population sources on both sides: the question is whether
    # the target's own population makes better training/exemplar data than another single
    # population. Including the `combined` row would answer a different question.
    print("\ndoes matching exemplars to the target help? (own source minus best other "
          "single-population source)")
    for p in POPULATIONS:
        others = [q for q in POPULATIONS if q != p]
        ft = FINETUNED.loc[p, p] - FINETUNED.loc[others, p].max()
        fs = macro.loc[p, p] - macro.loc[others, p].max()
        print(f"  {p:7s} FT {ft:+.3f}   FS {fs:+.3f}")

    ac = df.groupby("target").apply(lambda g: (g["pred"] == "AC").mean(), include_groups=False)
    gold_ac = df[df["seed"] == df["seed"].min()].groupby("target").apply(
        lambda g: (g["gold"] == "AC").mean(), include_groups=False)
    print("\nAchievement prediction share vs gold (the attractor check)")
    print("  " + "  ".join(f"{t}: pred={ac[t]:.2f} gold={gold_ac[t]:.2f}" for t in ac.index))

    return {"model": name, "macro": macro, "micro": micro,
            "spread_macro": fs_spread.mean(), "spread_micro": source_spread(micro).mean(),
            "diag_macro": [macro.loc[p, p] for p in POPULATIONS]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="Score only this model_id (default: all)")
    ap.add_argument("--fewshot_dir", default=str(FEWSHOT_DIR))
    args = ap.parse_args()

    files = sorted(pathlib.Path(args.fewshot_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"[ERROR] no prediction CSVs in {args.fewshot_dir}")

    summaries = []
    for f in files:
        df = pd.read_csv(f)
        df["target"] = df["target"].str.lower()
        model = df["model_id"].iloc[0]
        if args.model and model != args.model:
            continue
        bad = df["pred"].isna().sum()
        if bad:
            print(f"[warn] {f.name}: {bad} unparsed predictions")
        summaries.append(report(model, df))

    if len(summaries) > 1:
        print(f"\n{'=' * 78}\nACROSS MODELS\n{'=' * 78}")
        cmp = pd.DataFrame({
            "in-domain macro (mean of 4)": {s["model"]: sum(s["diag_macro"]) / 4 for s in summaries},
            "source spread, macro": {s["model"]: s["spread_macro"] for s in summaries},
            "source spread, micro": {s["model"]: s["spread_micro"] for s in summaries},
        })
        ft_macro, ft_micro = finetuned_reference()
        cmp.loc["fine-tuned (reference)"] = [
            sum(ft_macro.loc[p, p] for p in POPULATIONS) / 4,
            source_spread(ft_macro).mean(),
            source_spread(ft_micro).mean()]
        print(cmp.round(3).to_string())
        print("\nfine-tuned micro F1 matrix (rebuilt from per-seed training logs)")
        print(ft_micro.round(3).to_string())


if __name__ == "__main__":
    main()
