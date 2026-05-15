#!/usr/bin/env python3
"""
Population-typical lexical exhibits for the value-detection corpus.

For each (population, value) pair, compute the most distinctive tokens in the
focal cell vs. the same value-class in the other populations. Distinctiveness
is measured by log-odds-ratio with an informative Dirichlet prior built from
the overall corpus background (Monroe, Colaresi, Quinn 2008).

This is the kind of "ultra-Orthodox typical achievement vocabulary" exhibit
Sharon asked for in the email thread, plus the Joint parity check.

Outputs (under experiments/results/):
  lexical_distinctive_by_pop_value.csv     # all (pop, val) cells, top tokens
  lexical_ultra_BE_top.csv                 # Ultra-BE highlight (Sharon request)
  lexical_joint_UN_top.csv                 # Joint-UN parity check

The CSVs include z-scores so the LaTeX table author can decide thresholds.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# Light tokenization: lowercase, drop punctuation, drop pure digits, drop very
# short tokens. We keep an English stopword list small and explicit; the goal
# is to surface CONTENT vocabulary, not function words.
STOPWORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "for", "on",
        "with", "at", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "its", "our", "their", "me", "him", "us",
        "them", "myself", "yourself", "himself", "herself", "itself", "ourselves",
        "themselves", "what", "which", "who", "whom", "where", "when", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "can", "will", "just", "don", "should", "now", "also",
        "may", "could", "would", "very", "well", "make", "made", "use", "used",
        "if", "while", "into", "out", "up", "down", "over", "after", "before",
        "between", "through", "during", "above", "below", "again", "further",
        "then", "once", "here", "there", "about", "against", "off", "on",
        "ll", "ve", "re", "s", "d", "t", "m",
    ]
)

_token_re = re.compile(r"[A-Za-z][A-Za-z\-']+")


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [
        t.lower()
        for t in _token_re.findall(text)
        if len(t) > 2 and t.lower() not in STOPWORDS
    ]


def counts_by_corpus(
    df: pd.DataFrame, group_cols: List[str]
) -> Tuple[Dict[Tuple, Counter], Counter]:
    """Returns (per-cell Counter dict keyed by group_cols tuple, overall Counter)."""
    per_cell: Dict[Tuple, Counter] = {}
    overall: Counter = Counter()
    for keys, sub in df.groupby(group_cols):
        c = Counter()
        for txt in sub["Text"]:
            c.update(tokenize(txt))
        per_cell[keys if isinstance(keys, tuple) else (keys,)] = c
        overall.update(c)
    return per_cell, overall


def log_odds_dirichlet(
    focal: Counter, comparison: Counter, prior: Counter, alpha: float = 0.01
) -> Dict[str, float]:
    """
    Log-odds-ratio with informative Dirichlet prior (Monroe et al. 2008).

    For each token w:
      n_f = focal count of w
      n_c = comparison count of w
      a_w = alpha * prior[w]  (smoothing from background)
      N_f = sum(focal) + sum(a)
      N_c = sum(comparison) + sum(a)

      log_odds = log((n_f + a_w) / (N_f - n_f - a_w))
               - log((n_c + a_w) / (N_c - n_c - a_w))
      var = 1/(n_f + a_w) + 1/(n_c + a_w)
      z = log_odds / sqrt(var)

    Returns {token: z_score} (higher z => more distinctive of focal).
    """
    vocab = set(focal) | set(comparison) | set(prior)
    a = {w: alpha * prior.get(w, 0) for w in vocab}
    # Ensure each token has at least a tiny pseudo-prior so denominators are non-zero.
    eps = 1e-6
    for w in vocab:
        if a[w] <= 0:
            a[w] = eps

    sum_a = sum(a.values())
    N_f = sum(focal.values()) + sum_a
    N_c = sum(comparison.values()) + sum_a

    z_scores: Dict[str, float] = {}
    for w in vocab:
        n_f = focal.get(w, 0) + a[w]
        n_c = comparison.get(w, 0) + a[w]
        # Skip very rare tokens to keep the table interpretable.
        if focal.get(w, 0) + comparison.get(w, 0) < 3:
            continue
        log_odds = np.log(n_f / max(N_f - n_f, eps)) - np.log(
            n_c / max(N_c - n_c, eps)
        )
        var = 1.0 / n_f + 1.0 / n_c
        z = log_odds / np.sqrt(var)
        z_scores[w] = float(z)
    return z_scores


def build_distinctive_table(
    df: pd.DataFrame,
    populations: List[str],
    values: List[str],
    top_k: int,
    alpha: float,
) -> pd.DataFrame:
    """For each (pop, val) cell, compute distinctive tokens vs same-val in other pops."""
    per_cell, overall_prior = counts_by_corpus(df, ["Dataset", "Annotated Value"])
    rows = []
    for pop in populations:
        for val in values:
            focal_key = (pop, val)
            if focal_key not in per_cell:
                continue
            focal = per_cell[focal_key]
            if sum(focal.values()) < 10:
                continue
            # Comparison: same value class, all OTHER populations combined.
            comparison: Counter = Counter()
            for other_pop in populations:
                if other_pop == pop:
                    continue
                comparison.update(per_cell.get((other_pop, val), Counter()))
            if sum(comparison.values()) < 10:
                continue
            z = log_odds_dirichlet(focal, comparison, overall_prior, alpha=alpha)
            # Rank tokens by z-score, keep only positive (distinctive of focal).
            ranked = sorted(z.items(), key=lambda kv: -kv[1])[:top_k]
            for rank, (tok, score) in enumerate(ranked, start=1):
                rows.append(
                    {
                        "population": pop,
                        "value": val,
                        "rank": rank,
                        "token": tok,
                        "z_score": score,
                        "n_focal": focal.get(tok, 0),
                        "n_comparison": comparison.get(tok, 0),
                        "focal_size": sum(focal.values()),
                        "comparison_size": sum(comparison.values()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/merged.csv")
    ap.add_argument("--outdir", default="experiments/results")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if {"Dataset", "Text", "Annotated Value"} - set(df.columns):
        raise ValueError(f"Missing required columns; got {list(df.columns)}")
    df["Dataset"] = df["Dataset"].astype(str).str.strip()
    df["Annotated Value"] = df["Annotated Value"].astype(str).str.strip()

    populations = sorted(df["Dataset"].dropna().unique().tolist())
    values = sorted(df["Annotated Value"].dropna().unique().tolist())
    print(f"[INFO] populations: {populations}")
    print(f"[INFO] values     : {values}")
    print(f"[INFO] rows       : {len(df)}")

    distinctive = build_distinctive_table(
        df, populations=populations, values=values, top_k=args.top_k, alpha=args.alpha
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / "lexical_distinctive_by_pop_value.csv"
    distinctive.to_csv(full_csv, index=False)
    print(f"[OK] full table: {full_csv}  (n rows = {len(distinctive)})")

    # Highlights for the paper.
    for pop, val, name in [
        ("Ultra", "BE", "lexical_ultra_BE_top.csv"),
        ("Joint", "UN", "lexical_joint_UN_top.csv"),
        ("Ultra", "AC", "lexical_ultra_AC_top.csv"),
        ("Joint", "AC", "lexical_joint_AC_top.csv"),
    ]:
        sub = distinctive[
            (distinctive["population"] == pop) & (distinctive["value"] == val)
        ].head(20)
        if len(sub) == 0:
            print(f"[SKIP] {pop} x {val}: no data")
            continue
        path = outdir / name
        sub.to_csv(path, index=False)
        print(f"\n=== {pop} × {val} (top {len(sub)} distinctive tokens) ===")
        print(
            sub[["rank", "token", "z_score", "n_focal", "n_comparison"]]
            .head(15)
            .to_string(index=False)
        )
        print(f"[OK] saved: {path}")


if __name__ == "__main__":
    main()
