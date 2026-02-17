#!/usr/bin/env python3
"""
Focused misclassification analysis for the in-domain JOINT test set.

Outputs:
- experiments/results/misclf_joint_test_predictions.csv
- experiments/results/misclf_joint_test_misclassified.csv
- experiments/results/misclf_joint_test_confusion_matrix.csv
- experiments/plots/misclf_joint_test_confusion_matrix.png

What it does:
1) Loads a fine-tuned checkpoint (default: latest epoch under experiments/results/joint/)
2) Runs inference on data/joint/test.csv
3) Predicts *coarse* label by:
     - sigmoid(logits) over 20 fine labels
     - sum probabilities within each coarse group (COARSE_TO_FINE)
     - argmax over coarse groups
4) Confusion matrix is COARSE x COARSE (SD/ST/..UN)
5) Saves misclassified examples with confidence and some style features

Important implementation detail:
- Your saved epoch directories do NOT include modeling_deberta_arg_classifier.py, so
  AutoModelForSequenceClassification cannot load directly from the checkpoint folder.
- We therefore instantiate the architecture from --base_model_dir (default models/adam-smith),
  then load the fine-tuned weights from checkpoint/model.safetensors.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from safetensors.torch import load_file as safetensors_load_file

from src.data_loader import load_dataset
from src.utils import pick_device
from src.label_map import COARSE_TO_FINE


# -----------------------
# Helpers: checkpoint pick
# -----------------------
def find_latest_epoch_dir(run_dir: Path) -> Path:
    """
    Given experiments/results/<dataset>/, find the highest epoch_<N> subdir.
    If none exist, return run_dir itself (allows passing a direct checkpoint dir).
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    epoch_dirs = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("epoch_"):
            try:
                n = int(p.name.split("_")[1])
                epoch_dirs.append((n, p))
            except Exception:
                pass

    if epoch_dirs:
        epoch_dirs.sort(key=lambda x: x[0])
        return epoch_dirs[-1][1]

    return run_dir


# -----------------------
# Style features (lightweight)
# -----------------------
_punct_re = re.compile(r"[^\w\s]", re.UNICODE)

def style_features(text: str) -> Dict[str, float]:
    t = "" if text is None else str(text)
    n = len(t)
    n_words = len(t.split()) if t.strip() else 0
    n_punct = len(_punct_re.findall(t))
    n_upper = sum(1 for ch in t if ch.isalpha() and ch.isupper())
    n_alpha = sum(1 for ch in t if ch.isalpha())
    upper_ratio = (n_upper / n_alpha) if n_alpha > 0 else 0.0
    return {
        "n_chars": float(n),
        "n_words": float(n_words),
        "punct_per_100_chars": float((n_punct / max(1, n)) * 100.0),
        "upper_ratio": float(upper_ratio),
    }


# -----------------------
# Coarse prediction logic
# -----------------------
def build_coarse_index() -> Tuple[List[str], np.ndarray]:
    """
    Returns:
      coarse_labels: list like ["SD","ST",...]
      coarse_mask:   shape (n_coarse, 20) mask where mask[i,f]=1 if fine f belongs to coarse i
    """
    coarse_labels = list(COARSE_TO_FINE.keys())
    coarse_mask = np.zeros((len(coarse_labels), 20), dtype=np.float32)
    for i, c in enumerate(coarse_labels):
        for f in COARSE_TO_FINE[c]:
            coarse_mask[i, f] = 1.0
    return coarse_labels, coarse_mask


@torch.no_grad()
def predict_coarse(
    model,
    tokenizer,
    texts: List[str],
    device,
    coarse_mask: np.ndarray,
    max_length: int = 256,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pred_coarse_ids: (N,)
      coarse_scores:   (N, n_coarse) where score = sum of probs in each coarse group
      fine_probs:      (N, 20)
    """
    model.eval()
    all_fine_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out[0]
        probs = torch.sigmoid(logits).detach().cpu().numpy()  # (B, 20)
        all_fine_probs.append(probs)

    fine_probs = np.vstack(all_fine_probs)  # (N, 20)

    # coarse_score[n, c] = sum_{f in coarse(c)} fine_probs[n, f]
    coarse_scores = fine_probs @ coarse_mask.T  # (N, n_coarse)
    pred_coarse_ids = coarse_scores.argmax(axis=1)

    return pred_coarse_ids, coarse_scores, fine_probs


def load_model_and_tokenizer(
    ckpt_dir: Path,
    base_model_dir: Path,
    device,
):
    """
    Tokenizer: load from checkpoint if possible (keeps exact vocab/merges).
    Model: instantiate architecture from base_model_dir (contains remote code),
           then load weights from ckpt_dir/model.safetensors.
    """
    # Tokenizer (prefer checkpoint; fallback to base)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))

    # Instantiate architecture from base model dir (has custom modeling code)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_dir), trust_remote_code=True
    ).to(device)

    # Patch remote-code typo if needed
    if hasattr(model, "criterion") and not hasattr(model, "cirterion"):
        model.cirterion = model.criterion

    # Load fine-tuned weights
    weights_path = ckpt_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    state = safetensors_load_file(str(weights_path))
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[INFO] loaded weights from: {weights_path}")
    if missing:
        print(f"[WARN] missing keys (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys (showing up to 10): {unexpected[:10]}")

    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="joint")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument(
        "--model_dir",
        type=str,
        default="experiments/results/joint",
        help="Either a checkpoint dir, or a run dir containing epoch_<N>/ subfolders.",
    )
    ap.add_argument(
        "--base_model_dir",
        type=str,
        default="models/adam-smith",
        help="Directory of the base model (contains the custom modeling code).",
    )
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="experiments/results")
    ap.add_argument("--plots_dir", type=str, default="experiments/plots")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--hi_conf", type=float, default=0.90, help="Threshold for high-confidence mistakes")
    args = ap.parse_args()

    device = pick_device()
    ckpt_dir = find_latest_epoch_dir(Path(args.model_dir))
    base_model_dir = Path(args.base_model_dir)

    print(f"[INFO] device: {device}")
    print(f"[INFO] using checkpoint: {ckpt_dir}")
    print(f"[INFO] base model dir: {base_model_dir}")

    model, tokenizer = load_model_and_tokenizer(
        ckpt_dir=ckpt_dir,
        base_model_dir=base_model_dir,
        device=device,
    )

    data_path = Path(args.data_root) / args.dataset / f"{args.split}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    texts, fine, true_coarse = load_dataset(str(data_path), return_coarse=True)

    coarse_labels, coarse_mask = build_coarse_index()
    coarse_to_id = {c: i for i, c in enumerate(coarse_labels)}

    if len(true_coarse) == 0:
        raise ValueError("Empty dataset.")

    # true_coarse can be strings ("SD") or ids
    if isinstance(true_coarse[0], str):
        true_ids = np.array([coarse_to_id[x] for x in true_coarse], dtype=int)
    else:
        true_ids = np.array(true_coarse, dtype=int)

    pred_ids, coarse_scores, fine_probs = predict_coarse(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        coarse_mask=coarse_mask,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # confidence = normalized top coarse score
    denom = coarse_scores.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    coarse_probs = coarse_scores / denom
    conf = coarse_probs.max(axis=1)

    rows = []
    for i, (t, y, yhat, cmax) in enumerate(zip(texts, true_ids, pred_ids, conf)):
        feats = style_features(t)
        top3 = np.argsort(-coarse_probs[i])[:3]
        top3_str = ";".join([f"{coarse_labels[j]}:{coarse_probs[i, j]:.3f}" for j in top3])

        rows.append(
            {
                "idx": i,
                "text": t,
                "true_coarse": coarse_labels[int(y)],
                "pred_coarse": coarse_labels[int(yhat)],
                "pred_conf": float(cmax),
                "correct": bool(y == yhat),
                "top3_coarse": top3_str,
                **feats,
            }
        )

    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    plots_dir = Path(args.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = out_dir / f"misclf_{args.dataset}_{args.split}_predictions.csv"
    mis_csv = out_dir / f"misclf_{args.dataset}_{args.split}_misclassified.csv"
    cm_csv = out_dir / f"misclf_{args.dataset}_{args.split}_confusion_matrix.csv"
    cm_png = plots_dir / f"misclf_{args.dataset}_{args.split}_confusion_matrix.png"

    df.to_csv(pred_csv, index=False)

    mis = df[df["correct"] == False].copy().sort_values("pred_conf", ascending=False)
    mis.to_csv(mis_csv, index=False)

    print(f"[OK] saved predictions:   {pred_csv}")
    print(f"[OK] saved misclassified: {mis_csv} (n={len(mis)}/{len(df)})")

    # Confusion matrix (coarse x coarse, fixed label order)
    labels = list(range(len(coarse_labels)))
    cm = confusion_matrix(true_ids, pred_ids, labels=labels)
    cm_df = pd.DataFrame(cm, index=coarse_labels, columns=coarse_labels)
    cm_df.to_csv(cm_csv)
    print(f"[OK] saved confusion matrix CSV: {cm_csv}")

    # Plot confusion matrix (matplotlib only)
    plt.figure(figsize=(9.5, 7.5))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix (Coarse) — {args.dataset} {args.split} | {ckpt_dir.name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(coarse_labels)), coarse_labels, rotation=0)
    plt.yticks(range(len(coarse_labels)), coarse_labels)
    plt.colorbar(fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v != 0:
                plt.text(j, i, str(v), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(cm_png, dpi=200)
    plt.close()
    print(f"[OK] saved confusion matrix PNG: {cm_png}")

    print("\n[REPORT] classification_report (coarse):")
    print(classification_report(true_ids, pred_ids, labels=labels, target_names=coarse_labels, digits=3))

    # Top confusion pairs (exclude diagonal)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    pairs = []
    for i in range(cm_no_diag.shape[0]):
        for j in range(cm_no_diag.shape[1]):
            if cm_no_diag[i, j] > 0:
                pairs.append((cm_no_diag[i, j], coarse_labels[i], coarse_labels[j]))
    pairs.sort(reverse=True)

    print("\n[QUAL] Top confusion pairs (true -> pred):")
    for k, (count, y, yhat) in enumerate(pairs[:10], start=1):
        print(f"  {k:02d}. {y} -> {yhat}: {count}")

    # High-confidence mistakes
    hi = mis[mis["pred_conf"] >= args.hi_conf]
    print(f"\n[QUAL] High-confidence mistakes (pred_conf >= {args.hi_conf:.2f}): {len(hi)}")
    if len(hi) > 0:
        for _, r in hi.head(10).iterrows():
            snippet = str(r["text"]).replace("\n", " ")[:140]
            print(
                f"  true={r['true_coarse']} pred={r['pred_coarse']} "
                f"conf={r['pred_conf']:.3f} | {snippet}{'...' if len(str(r['text'])) > 140 else ''}"
            )

    # Writing-style summaries (correct vs wrong)
    corr = df[df["correct"] == True]
    wrong = df[df["correct"] == False]

    def _summ(group: pd.DataFrame, col: str) -> str:
        return f"mean={group[col].mean():.2f}  median={group[col].median():.2f}"

    print("\n[QUAL] Writing-style summary (correct vs wrong):")
    for col in ["n_words", "n_chars", "punct_per_100_chars", "upper_ratio"]:
        print(f"  {col}: correct({_summ(corr, col)}) | wrong({_summ(wrong, col)})")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
