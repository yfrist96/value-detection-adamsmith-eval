#!/usr/bin/env python3
"""
Embedding-based register distance between populations, as an independent check on the
frequency-based measures in src/register_distance.py.

Why a second measure: of the three lexical measures in register_distance.py, only one (non-overlap
of the 100 most frequent content words) tracks transfer. The other two operate on the whole
vocabulary and are dominated by the rare tail on corpora this small. A measure that does not count
word frequencies at all is therefore worth having, because if it orders the six population pairs
the same way, the register account rests on two measures of different kind rather than on one of
three that were tried.

Same discipline as register_distance.py: every distance is computed from the TRAINING text only,
with no access to labels, predictions or scores. The transfer matrix is loaded afterwards, purely
to correlate against. This is post hoc and blind to the outcome by construction, and it is
reported whichever way it comes out.

Encoder choice matters here. We deliberately do NOT use Adam-Smith or its fine-tuned checkpoints:
those are the models whose behaviour the register account is explaining, so measuring distance in
their representation space would reintroduce exactly the circularity this analysis exists to
break. We use a general-purpose sentence encoder that has never seen our data or the value task.

Two distances are reported, because they answer different questions:
  centroid   cosine distance between the mean embeddings of two populations' training responses.
             Asks whether the populations sit in different regions of semantic space.
  energy     energy distance between the two embedding samples. Asks whether the whole
             distributions differ, not just their centres, so a population with a wide spread and
             one with a narrow spread around the same centre are separated.

Usage:
    python -m src.register_distance_embed
    python -m src.register_distance_embed --model sentence-transformers/all-mpnet-base-v2
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

# torch/transformers must be imported before sklearn on macOS or libomp aborts the process.
from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

POPS = ["asian", "indian", "ijh", "ultra"]
RESULTS = pathlib.Path("experiments/results")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_train_texts(pop: str) -> list[str]:
    df = pd.read_csv(f"data/{pop}/train.csv")
    return [t for t in df["Text"].fillna("").astype(str) if t.strip()]


def cosine_centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity between the two population centroids."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    return float(1 - np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb)))


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance between two samples of embeddings.

    2*E|X-Y| - E|X-X'| - E|Y-Y'|, using Euclidean distance on L2-normalised embeddings. Unlike the
    centroid distance this is sensitive to the shape of each population's distribution, so two
    populations that share a mean but differ in spread do not come out as identical.
    """
    def mean_pairwise(x: np.ndarray, y: np.ndarray) -> float:
        d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        return float(d.mean())

    return 2 * mean_pairwise(a, b) - mean_pairwise(a, a) - mean_pairwise(b, b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="sentence-transformers encoder; must not be Adam-Smith or a checkpoint of it")
    args = ap.parse_args()

    if "adam" in args.model.lower() or "human_value" in args.model.lower():
        raise SystemExit("[ERROR] refusing to measure register distance in the representation "
                         "space of the model under study; that is the circularity this analysis "
                         "exists to break.")

    print(f"encoder: {args.model}")
    encoder = SentenceTransformer(args.model)

    emb = {}
    for pop in POPS:
        texts = load_train_texts(pop)
        # L2-normalised, so cosine and Euclidean geometry agree and the energy distance is bounded.
        emb[pop] = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                  batch_size=64)
        print(f"{pop:<7} n={len(texts):>4}  dim={emb[pop].shape[1]}")

    rows = []
    for a, b in itertools.combinations(POPS, 2):
        rows.append({
            "pair": f"{a}-{b}", "a": a, "b": b,
            "emb_centroid": cosine_centroid_distance(emb[a], emb[b]),
            "emb_energy": energy_distance(emb[a], emb[b]),
        })
    reg = pd.DataFrame(rows)

    # --- only now do we look at the outcome -------------------------------------------------
    m = pd.read_csv(RESULTS / "cross_domain_macro_f1_matrix.csv", index_col=0)
    m.index = [str(i).replace("joint", "ijh") for i in m.index]
    m.columns = [str(c).replace("joint", "ijh") for c in m.columns]
    reg["transfer"] = [(m.loc[r["a"], r["b"]] + m.loc[r["b"], r["a"]]) / 2 for _, r in reg.iterrows()]

    # Carry the frequency-based measures alongside, so the two families sit in one table.
    prior = RESULTS / "register_distance.csv"
    if prior.exists():
        cols = ["pair", "jsd", "jaccard", "overlap"]
        reg = reg.merge(pd.read_csv(prior)[cols], on="pair", how="left")

    reg = reg.sort_values("transfer", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    reg.to_csv(RESULTS / "register_distance_embeddings.csv", index=False)

    print("\nPer-pair embedding distance vs mean bidirectional transfer\n")
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nSpearman correlation with transfer (n=6 pairs). Negative = more distance, less transfer.\n")
    out = {"encoder": args.model, "correlations": {}}
    for col in [c for c in ["emb_centroid", "emb_energy", "jsd", "jaccard", "overlap"] if c in reg]:
        rho, p = spearmanr(reg[col], reg["transfer"])
        out["correlations"][col] = {"rho": float(rho), "p": float(p)}
        print(f"  {col:<14} rho={rho:+.3f}  p={p:.3f}")
    (RESULTS / "register_distance_embeddings.json").write_text(json.dumps(out, indent=2))

    print(f"\n[OK] wrote {RESULTS/'register_distance_embeddings.csv'} and .json")
    print("n=6 pairs is small: read the ordering, not the p-values.")


if __name__ == "__main__":
    main()
