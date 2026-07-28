#!/usr/bin/env python3
"""
Measure lexical-register distance between populations from TEXT ALONE, then ask whether it
predicts cross-population transfer.

Why this exists: a reviewer objected that "lexical register" is invoked as the explanatory axis
but never operationalised independently of the outcome it explains, so that register is inferred
from the transfer pattern and then used to account for it. This script breaks that loop. Every
measure below is computed from the TRAINING text only. It never sees a label, a prediction, or a
macro-F1 score. The transfer matrix is loaded afterwards, purely to correlate against.

This is not a pre-registration and must not be described as one. It is a post-hoc measure that is
blind to the outcome by construction, and it is reported whichever way it comes out.

Measures (all symmetric, all label-blind):
  jsd        Jensen-Shannon divergence between unigram distributions over content words.
             The primary measure: it uses the whole distribution, not just the frequent tail.
  jaccard    1 - |V_a & V_b| / |V_a | V_b| over content-word types.
  overlap    1 - |top100_a & top100_b| / 100, on the 100 most frequent content words.

Baseline predictors it is compared against, so "register explains transfer" is a claim with a
control rather than a bare correlation:
  same_nation      1 if the two populations share a nationality (IJH/Ultra), else 0
  size_ratio       log ratio of training-set sizes
  length_diff      absolute difference in mean response length in words

Usage:
    python -m src.register_distance
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import pathlib
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

POPS = ["asian", "indian", "ijh", "ultra"]
RESULTS = pathlib.Path("experiments/results")
TOKEN_RE = re.compile(r"[a-z][a-z'-]+")
# Nationality is the control the paper contrasts register against: the IJH/Ultra pair is the one
# that shares a nation, and it is also the pair where transfer collapses.
SAME_NATION = {frozenset({"ijh", "ultra"})}


def content_tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 2]


def load_train(pop: str) -> tuple[list[list[str]], collections.Counter]:
    df = pd.read_csv(f"data/{pop}/train.csv")
    docs = [content_tokens(t) for t in df["Text"].fillna("").astype(str)]
    return docs, collections.Counter(itertools.chain.from_iterable(docs))


def jensen_shannon(a: collections.Counter, b: collections.Counter) -> float:
    vocab = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    p = np.array([a.get(w, 0) / na for w in vocab])
    q = np.array([b.get(w, 0) / nb for w in vocab])
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    docs, counts, lengths = {}, {}, {}
    for pop in POPS:
        docs[pop], counts[pop] = load_train(pop)
        lengths[pop] = float(np.mean([len(d) for d in docs[pop]]))
        print(f"{pop:<7} n_docs={len(docs[pop]):>4}  content tokens={sum(counts[pop].values()):>5}  "
              f"types={len(counts[pop]):>4}  mean len={lengths[pop]:.2f}")

    top100 = {p: {w for w, _ in counts[p].most_common(100)} for p in POPS}
    rows = []
    for a, b in itertools.combinations(POPS, 2):
        va, vb = set(counts[a]), set(counts[b])
        rows.append({
            "pair": f"{a}-{b}", "a": a, "b": b,
            "jsd": jensen_shannon(counts[a], counts[b]),
            "jaccard": 1 - len(va & vb) / len(va | vb),
            "overlap": 1 - len(top100[a] & top100[b]) / 100,
            "same_nation": int(frozenset({a, b}) in SAME_NATION),
            "size_ratio": abs(math.log(len(docs[a]) / len(docs[b]))),
            "length_diff": abs(lengths[a] - lengths[b]),
        })
    reg = pd.DataFrame(rows)

    # --- only now do we look at the outcome -------------------------------------------------
    m = pd.read_csv(RESULTS / "cross_domain_macro_f1_matrix.csv", index_col=0)
    m.index = [str(i).replace("joint", "ijh") for i in m.index]
    m.columns = [str(c).replace("joint", "ijh") for c in m.columns]
    reg["transfer"] = [(m.loc[r["a"], r["b"]] + m.loc[r["b"], r["a"]]) / 2 for _, r in reg.iterrows()]

    reg = reg.sort_values("transfer", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    reg.to_csv(RESULTS / "register_distance.csv", index=False)

    print("\nPer-pair register distance vs mean bidirectional transfer\n")
    print(reg[["pair", "transfer", "jsd", "jaccard", "overlap", "same_nation", "size_ratio", "length_diff"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nSpearman correlation with transfer (n=6 pairs). Negative = more distance, less transfer.\n")
    out = {}
    for col in ["jsd", "jaccard", "overlap", "same_nation", "size_ratio", "length_diff"]:
        rho, p = spearmanr(reg[col], reg["transfer"])
        out[col] = {"rho": float(rho), "p": float(p)}
        print(f"  {col:<12} rho={rho:+.3f}  p={p:.3f}")
    (RESULTS / "register_distance_correlations.json").write_text(json.dumps(out, indent=2))
    print(f"\n[OK] wrote {RESULTS/'register_distance.csv'} and register_distance_correlations.json")
    print("n=6 pairs is small: read the ordering, not the p-values.")


if __name__ == "__main__":
    main()
