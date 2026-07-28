#!/usr/bin/env python3
"""
Dump per-item predictions for every cell of the cross-population transfer matrix.

Motivation: the paper's metrics come from `src/eval.py`, which calls
`f1_score(..., average="macro")` with no `labels=` argument. sklearn then averages over the
union of gold and predicted classes, so the denominator depends on what the model predicts.
A model that predicts into a class with no gold support in that test set picks up an extra
zero-F1 category; a model that stays inside the gold classes does not. Every matrix cell uses
the same code path, so the paper is self-consistent, but the cells are not strictly comparable.

Without saved per-item predictions this cannot be checked after the fact, which is why this
script exists. It writes predictions once, and `--recompute` scores them under all three
conventions so the difference can be measured rather than argued about:

  union    : labels = sorted(set(gold) | set(pred))    <- what src/eval.py does today
  goldonly : labels = sorted(set(gold))                <- candidate convention for the paper
  fixed12  : labels = the full 12-category coarse space

Usage:
    python -m src.dump_predictions                       # all sources x targets x available seeds
    python -m src.dump_predictions --seeds 42            # just seed 42
    python -m src.dump_predictions --recompute           # rescore existing dumps, no inference

Outputs:
    experiments/results/predictions/<source>__<target>__seed<N>.csv
    experiments/results/matrix_<convention>.csv
    experiments/results/matrix_convention_comparison.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import torch  # noqa: F401  (imported before sklearn: the reverse order trips libomp on macOS)
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import f1_score  # noqa: E402  (must follow torch/transformers, see above)

from src.data_loader import load_dataset
from src.label_map import FINE_TO_COARSE
from src.run_merged_base_adamsmith_eval import COARSE_LABELS, predict_batch
from src.utils import pick_device

RESULTS = pathlib.Path("experiments/results")
PRED_DIR = RESULTS / "predictions"
TARGETS = ["asian", "indian", "ijh", "ultra"]
SOURCES = ["asian", "indian", "ijh", "ultra", "combined"]
BASE_MODEL = "models/adam-smith"
CONVENTIONS = ("union", "goldonly", "fixed12")


def macro_f1(gold, pred, convention: str) -> float:
    if convention == "union":
        labels = sorted(set(gold) | set(pred))
    elif convention == "goldonly":
        labels = sorted(set(gold))
    elif convention == "fixed12":
        labels = COARSE_LABELS
    else:
        raise ValueError(convention)
    return float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))


def final_epoch_dir(run_dir: pathlib.Path) -> pathlib.Path | None:
    """Highest-numbered epoch_N directory, or None if the run has not finished."""
    epochs = [(int(p.name.split("_")[1]), p) for p in run_dir.glob("epoch_*") if p.name.split("_")[1].isdigit()]
    return max(epochs)[1] if epochs else None


def load_model(model_dir: pathlib.Path, device):
    """Load a checkpoint for inference.

    Fine-tuned epoch directories do NOT contain the custom `modeling_*.py` remote-code file, so
    `from_pretrained(epoch_dir, trust_remote_code=True)` fails on them. The architecture therefore
    always comes from the base model directory, and fine-tuned weights are loaded on top.

    The state-dict load uses strict=False, matching src/ablate_achievement_vocab.py. That is
    necessary (buffers differ) but dangerous: on a key-name mismatch it would silently leave the
    base weights in place and we would evaluate the base model while believing it fine-tuned.
    We therefore assert that a substantial fraction of parameters actually matched.
    """
    is_base = model_dir == pathlib.Path(BASE_MODEL)
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, trust_remote_code=True).to(device)
    # Same remote-code typo patch as src/train.py.
    if hasattr(model, "criterion") and not hasattr(model, "cirterion"):
        model.cirterion = model.criterion

    if not is_base:
        weights = model_dir / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"no model.safetensors in {model_dir}")
        state = safetensors_load_file(str(weights))
        result = model.load_state_dict(state, strict=False)
        matched = len(state) - len(result.unexpected_keys)
        if matched < 0.9 * len(state):
            raise RuntimeError(
                f"{model_dir}: only {matched}/{len(state)} checkpoint tensors matched the model. "
                "Refusing to evaluate what would silently be the base model."
            )
    model.eval()
    return tok, model


def predict_split(tok, model, texts, device, batch_size: int = 32) -> list[str]:
    preds: list[str] = []
    for i in range(0, len(texts), batch_size):
        scores = predict_batch(model, tok, list(texts[i:i + batch_size]), device)
        preds.extend(FINE_TO_COARSE[j] for j in np.argmax(scores, axis=1))
    return preds


def dump(seeds: list[int]) -> None:
    device = pick_device()
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    splits = {t: load_dataset(f"data/{t}/test.csv", return_coarse=True) for t in TARGETS}

    runs: list[tuple[str, int, pathlib.Path]] = [("base", 0, pathlib.Path(BASE_MODEL))]
    for source in SOURCES:
        for seed in seeds:
            run_dir = RESULTS / source / f"seed_{seed}"
            epoch_dir = final_epoch_dir(run_dir) if run_dir.exists() else None
            if epoch_dir is None:
                print(f"[skip] no finished checkpoint for {source} seed {seed}")
                continue
            runs.append((source, seed, epoch_dir))

    for source, seed, model_dir in runs:
        tok, model = load_model(model_dir, device)
        for target in TARGETS:
            texts, _, gold = splits[target]
            out = PRED_DIR / f"{source}__{target}__seed{seed}.csv"
            preds = predict_split(tok, model, texts, device)
            pd.DataFrame({"text": texts, "gold": gold, "pred": preds,
                          "source": source, "target": target, "seed": seed}).to_csv(out, index=False)
            print(f"[ok] {out.name:<34} n={len(texts):>4}  "
                  + "  ".join(f"{c}={macro_f1(gold, preds, c):.4f}" for c in CONVENTIONS))
        del model


def recompute() -> None:
    files = sorted(PRED_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"[ERROR] no prediction dumps in {PRED_DIR}. Run without --recompute first.")

    rows = []
    for f in files:
        d = pd.read_csv(f)
        rows.append({"source": d["source"].iloc[0], "target": d["target"].iloc[0], "seed": int(d["seed"].iloc[0]),
                     **{c: macro_f1(d["gold"], d["pred"], c) for c in CONVENTIONS},
                     "n_gold_classes": d["gold"].nunique(),
                     "n_union_classes": len(set(d["gold"]) | set(d["pred"]))})
    per_cell = pd.DataFrame(rows)
    per_cell.to_csv(RESULTS / "matrix_per_cell_all_conventions.csv", index=False)

    order = ["base"] + SOURCES
    for c in CONVENTIONS:
        m = per_cell.pivot_table(index="source", columns="target", values=c, aggfunc="mean").round(4)
        m = m.reindex([s for s in order if s in m.index])[TARGETS]
        m.to_csv(RESULTS / f"matrix_{c}.csv")

    per_cell["goldonly_minus_union"] = per_cell["goldonly"] - per_cell["union"]
    comp = per_cell.sort_values("goldonly_minus_union", ascending=False)
    comp.to_csv(RESULTS / "matrix_convention_comparison.csv", index=False)

    print("\nMacro F1 under the paper's current convention (union):")
    print(pd.read_csv(RESULTS / "matrix_union.csv", index_col=0).to_string())
    print("\nCells where the convention matters most (goldonly - union):")
    print(comp.head(8)[["source", "target", "seed", "union", "goldonly", "fixed12",
                        "n_gold_classes", "n_union_classes"]].to_string(index=False))
    moved = (comp["goldonly_minus_union"].abs() > 0.02).sum()
    print(f"\n{moved} of {len(comp)} cells move by more than 0.02 between union and goldonly.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds (default: 42,43,44)")
    ap.add_argument("--recompute", action="store_true", help="Rescore existing dumps without running inference")
    args = ap.parse_args()

    if not args.recompute:
        dump([int(s) for s in args.seeds.split(",") if s.strip()])
    recompute()


if __name__ == "__main__":
    main()
