#!/usr/bin/env python3
"""
Achievement-bias ablation on the base Adam-Smith model.

Hypothesis: the base model's tendency to over-predict Achievement (AC) is
driven by a small "achievement-coded" vocabulary (achieve, impact, improve,
advance, …). We test this directly by masking those words in `data/merged.csv`
and re-running predictions. We compare original vs. masked along three axes:

  - AC prediction frequency (and AC P/R/F1 vs. annotations)
  - macro-F1 across the 12 coarse SVS classes
  - per-cell shifts in the row-normalized confusion matrix

Outputs (under `experiments/results/ablation_achievement/` and
`experiments/plots/ablation_achievement/`):

  predictions.csv        per-row: text, masked_text, dataset, true_coarse,
                         pred_orig, pred_masked, ac_prob_orig/ac_prob_masked,
                         conf_orig/conf_masked, flipped, ac_lost, words_masked
  summary.json           global + per-dataset metrics before/after, McNemar p
  summary.txt            human-readable rendering of summary.json
  global_cm_diff.{png,pdf}             signed diff CM (row-normalized)
  <dataset>_cm_diff.{png,pdf}          signed diff CM per source dataset

Run:
  python -m src.ablate_achievement_vocab \\
      --model_dir models/adam-smith \\
      --input_csv data/merged.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors.torch import load_file as safetensors_load_file
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.run_merged_base_adamsmith_eval import (
    COARSE_LABELS,
    aggregate_to_coarse_scores,
    normalize_coarse,
    predict_batch,
)
from src.utils import pick_device, save_fig


def _find_latest_epoch_dir(run_dir: Path) -> Path:
    """Pick the highest-numbered epoch_N directory under `run_dir`.

    Returns `run_dir` itself if no epoch_N subdir exists (e.g. the user
    pointed directly at an epoch dir).
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {run_dir}")
    candidates = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("epoch_"):
            try:
                candidates.append((int(p.name.split("_")[1]), p))
            except Exception:
                pass
    if not candidates:
        return run_dir
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_model_and_tokenizer(
    base_model_dir: str,
    checkpoint_dir: Optional[str],
    device,
):
    """Load the Adam-Smith architecture, optionally overlaying fine-tuned weights.

    If `checkpoint_dir` is None: return the base model (matching the original
    base-model ablation). Otherwise: load architecture from `base_model_dir`
    and apply the latest `epoch_*/model.safetensors` from `checkpoint_dir`.
    """
    if checkpoint_dir is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_dir, trust_remote_code=True
        )
        model.eval()
        model.to(device)
        print(f"[INFO] loaded BASE model from {base_model_dir}")
        return model, tokenizer, base_model_dir

    ckpt_root = Path(checkpoint_dir)
    ckpt_dir = _find_latest_epoch_dir(ckpt_root)
    print(f"[INFO] using checkpoint: {ckpt_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir, trust_remote_code=True
    ).to(device)
    if hasattr(model, "criterion") and not hasattr(model, "cirterion"):
        model.cirterion = model.criterion

    weights_path = ckpt_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing weights: {weights_path}")
    state = safetensors_load_file(str(weights_path))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[INFO] loaded fine-tuned weights from: {weights_path}")
    return model, tokenizer, str(ckpt_dir)


# ---------------------------------------------------------------------------
# Tight target vocabulary (4 seeds + morphological variants).
# Edit here to expand or narrow the list.
# ---------------------------------------------------------------------------
TARGET_WORDS: List[str] = [
    "achieve", "achieves", "achieved", "achieving",
    "achievement", "achievements",
    "impact", "impacts", "impacted", "impacting",
    "improve", "improves", "improved", "improving",
    "improvement", "improvements",
    "advance", "advances", "advanced", "advancing",
    "advancement", "advancements",
]


def build_pattern(words: List[str]) -> re.Pattern:
    # Longest-first ordering avoids "advance" matching inside "advancement".
    escaped = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def mask_text(text: str, pattern: re.Pattern, mask_token: str) -> Tuple[str, int]:
    n = 0

    def repl(_m):
        nonlocal n
        n += 1
        return mask_token

    return pattern.sub(repl, text), n


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def predict_coarse_probs(model, tokenizer, texts: List[str], device, batch_size: int) -> np.ndarray:
    """Return (N, 12) softmax-normalized coarse probabilities for the SVS labels."""
    chunks: List[np.ndarray] = []
    bs = max(1, int(batch_size))
    for i in tqdm(range(0, len(texts), bs), total=math.ceil(len(texts) / bs), desc="predict"):
        chunks.append(predict_batch(model, tokenizer, texts[i:i + bs], device))
    fine_scores = np.vstack(chunks) if chunks else np.zeros((0, 20), dtype=float)
    coarse_scores, _ = aggregate_to_coarse_scores(fine_scores)
    return softmax_rows(coarse_scores)


def per_class_prf(true: List[str], pred: List[str], cls: str) -> Tuple[float, float, float, int]:
    tp = sum(1 for t, p in zip(true, pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(true, pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(true, pred) if t == cls and p != cls)
    support = tp + fn
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, support


def macro_f1(true: List[str], pred: List[str], classes: List[str]) -> float:
    f1s = []
    for c in classes:
        _, _, f1, support = per_class_prf(true, pred, c)
        if support > 0:
            f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def confusion_matrix(true: List[str], pred: List[str], classes: List[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true, pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def row_normalize(cm: np.ndarray) -> np.ndarray:
    rs = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rs > 0, cm / np.maximum(rs, 1), 0.0)


def plot_cm_diff(out_basename: str, title: str, classes: List[str],
                 cm_orig: np.ndarray, cm_masked: np.ndarray) -> None:
    diff = row_normalize(cm_masked) - row_normalize(cm_orig)

    # Auto-scale the diverging colormap to the actual signed data range, kept
    # symmetric around zero so red/blue map to opposite signs. A small floor
    # (0.01) prevents division by zero when nothing flipped at all.
    data_max = float(np.max(np.abs(diff))) if diff.size else 0.0
    vmax = max(data_max, 0.01)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(diff, aspect="auto", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xlabel("Predicted (coarse)")
    ax.set_ylabel("Annotated (coarse)")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    rs_orig = cm_orig.sum(axis=1)
    ax.set_yticklabels([f"{c} (n={int(rs_orig[i])})" for i, c in enumerate(classes)])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Δ row-normalized rate (masked − original); scale ±{vmax:.2f}")

    # Annotate every cell with a non-trivial shift; pick text colour relative
    # to the auto-scaled vmax so dark cells stay readable at any magnitude.
    annot_threshold = max(vmax * 0.05, 0.005)
    contrast_threshold = vmax * 0.6
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff[i, j]
            if abs(v) >= annot_threshold:
                ax.text(
                    j, i, f"{v:+.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(v) >= contrast_threshold else "black",
                )
    fig.tight_layout()
    save_fig(fig, out_basename)
    plt.close(fig)


def mcnemar_ac_pvalue(pred_orig: List[str], pred_masked: List[str]) -> float:
    # b: orig predicted AC, masked did not. c: orig did not, masked did.
    b = sum(1 for o, m in zip(pred_orig, pred_masked) if o == "AC" and m != "AC")
    c = sum(1 for o, m in zip(pred_orig, pred_masked) if o != "AC" and m == "AC")
    if b + c == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.sf(chi, df=1))
    except Exception:
        return float(math.erfc(math.sqrt(chi / 2.0)))


def summarize_slice(true_: List[str], po: List[str], pm: List[str], classes: List[str]) -> Dict:
    ac_orig = sum(1 for p in po if p == "AC")
    ac_masked = sum(1 for p in pm if p == "AC")
    prec_o, rec_o, f1_o, sup = per_class_prf(true_, po, "AC")
    prec_m, rec_m, f1_m, _ = per_class_prf(true_, pm, "AC")
    return {
        "n": len(po),
        "ac_support_in_truth": sup,
        "ac_pred_count_orig": ac_orig,
        "ac_pred_count_masked": ac_masked,
        "ac_pred_share_orig": ac_orig / max(1, len(po)),
        "ac_pred_share_masked": ac_masked / max(1, len(pm)),
        "ac_recall_orig": rec_o, "ac_recall_masked": rec_m,
        "ac_precision_orig": prec_o, "ac_precision_masked": prec_m,
        "ac_f1_orig": f1_o, "ac_f1_masked": f1_m,
        "macro_f1_orig": macro_f1(true_, po, classes),
        "macro_f1_masked": macro_f1(true_, pm, classes),
        "n_flipped": sum(1 for o, m in zip(po, pm) if o != m),
        "n_ac_lost": sum(1 for o, m in zip(po, pm) if o == "AC" and m != "AC"),
        "n_ac_gained": sum(1 for o, m in zip(po, pm) if o != "AC" and m == "AC"),
        "mcnemar_ac_pvalue": mcnemar_ac_pvalue(po, pm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/adam-smith",
                    help="Base architecture directory (custom Adam-Smith code).")
    ap.add_argument("--checkpoint_dir", default=None,
                    help="If set, overlay fine-tuned weights from "
                         "<checkpoint_dir>/epoch_*/model.safetensors. Used to ablate the "
                         "fine-tuned joint model on its own test set, e.g. "
                         "--checkpoint_dir experiments/results/joint.")
    ap.add_argument("--input_csv", default="data/merged.csv")
    ap.add_argument("--dataset_col", default="Dataset")
    ap.add_argument("--text_col", default="Text")
    ap.add_argument("--annotated_col", default="Annotated Value")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output_label", default="ablation_achievement",
                    help="Subdirectory name under experiments/results and experiments/plots. "
                         "Use a distinct label per run to avoid clobbering "
                         "(e.g. ablation_achievement_joint_finetuned).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results_dir = os.path.join("experiments", "results", args.output_label)
    plots_dir = os.path.join("experiments", "plots", args.output_label)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    for col in (args.dataset_col, args.text_col, args.annotated_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.input_csv}")

    device = pick_device()
    print(f"[INFO] device: {device}")
    model, tokenizer, model_source = load_model_and_tokenizer(
        args.model_dir, args.checkpoint_dir, device
    )

    mask_token = tokenizer.mask_token or "[MASK]"
    pattern = build_pattern(TARGET_WORDS)
    print(f"[INFO] mask token: {mask_token!r}")
    print(f"[INFO] target words ({len(TARGET_WORDS)}): {TARGET_WORDS}")

    texts_orig = df[args.text_col].fillna("").astype(str).tolist()
    masked_pairs = [mask_text(t, pattern, mask_token) for t in texts_orig]
    texts_masked = [m for m, _ in masked_pairs]
    n_masked_per_row = [n for _, n in masked_pairs]
    n_rows_with_match = sum(1 for n in n_masked_per_row if n > 0)
    print(f"[INFO] rows with ≥1 match: {n_rows_with_match}/{len(texts_orig)} "
          f"({n_rows_with_match / max(1, len(texts_orig)):.1%})")

    # Sanity-check sample: 5 random matched rows, before vs after.
    rng = random.Random(args.seed)
    matched_idx = [i for i, n in enumerate(n_masked_per_row) if n > 0]
    sample_idx = rng.sample(matched_idx, k=min(5, len(matched_idx)))
    for i in sample_idx:
        print(f"\n[SAMPLE {i}] BEFORE: {texts_orig[i][:240]}")
        print(f"[SAMPLE {i}] AFTER:  {texts_masked[i][:240]}")

    print("\n[INFO] predicting on original texts ...")
    probs_orig = predict_coarse_probs(model, tokenizer, texts_orig, device, args.batch_size)
    print("[INFO] predicting on masked texts ...")
    probs_masked = predict_coarse_probs(model, tokenizer, texts_masked, device, args.batch_size)

    classes = COARSE_LABELS
    ac_idx = classes.index("AC")
    pred_orig = [classes[int(i)] for i in probs_orig.argmax(axis=1)]
    pred_masked = [classes[int(i)] for i in probs_masked.argmax(axis=1)]
    ann = [normalize_coarse(x) for x in df[args.annotated_col].fillna("").astype(str).tolist()]
    datasets = df[args.dataset_col].astype(str).tolist()

    out_rows = []
    for i in range(len(df)):
        out_rows.append({
            "idx": i,
            "dataset": datasets[i],
            "true_coarse": ann[i],
            "pred_orig": pred_orig[i],
            "pred_masked": pred_masked[i],
            "ac_prob_orig": float(probs_orig[i, ac_idx]),
            "ac_prob_masked": float(probs_masked[i, ac_idx]),
            "conf_orig": float(probs_orig[i].max()),
            "conf_masked": float(probs_masked[i].max()),
            "flipped": pred_orig[i] != pred_masked[i],
            "ac_lost": (pred_orig[i] == "AC") and (pred_masked[i] != "AC"),
            "words_masked": int(n_masked_per_row[i]),
            "text": texts_orig[i],
            "masked_text": texts_masked[i],
        })
    pred_csv = Path(results_dir) / "predictions.csv"
    pd.DataFrame(out_rows).to_csv(pred_csv, index=False)
    print(f"\n[SAVED] {pred_csv}")

    summary = {
        "model_source": model_source,
        "input_csv": args.input_csv,
        "output_label": args.output_label,
        "target_words": TARGET_WORDS,
        "mask_token": mask_token,
        "n_rows_total": len(df),
        "n_rows_with_match": n_rows_with_match,
        "global": summarize_slice(ann, pred_orig, pred_masked, classes),
        "by_dataset": {},
    }
    for ds in sorted(set(datasets)):
        idxs = [i for i, d in enumerate(datasets) if d == ds]
        summary["by_dataset"][ds] = summarize_slice(
            [ann[i] for i in idxs],
            [pred_orig[i] for i in idxs],
            [pred_masked[i] for i in idxs],
            classes,
        )

    summary_json = Path(results_dir) / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVED] {summary_json}")

    # Pretty-printed summary
    lines: List[str] = []
    lines.append("# Achievement-bias ablation summary\n")
    lines.append(f"Model source: {model_source}")
    lines.append(f"Input CSV:    {args.input_csv}")
    lines.append(f"Target words ({len(TARGET_WORDS)}): {TARGET_WORDS}")
    lines.append(f"Mask token:   {mask_token}")
    lines.append(f"Rows with ≥1 match: {n_rows_with_match}/{len(df)}\n")

    def fmt_block(name: str, s: Dict) -> None:
        lines.append(f"## {name}  (n={s['n']}, AC support in truth={s['ac_support_in_truth']})")
        lines.append(
            f"  AC-pred share: {s['ac_pred_share_orig']:.3f} → {s['ac_pred_share_masked']:.3f}  "
            f"(Δ {s['ac_pred_share_masked'] - s['ac_pred_share_orig']:+.3f})"
        )
        lines.append(
            f"  AC recall:     {s['ac_recall_orig']:.3f} → {s['ac_recall_masked']:.3f}  "
            f"(Δ {s['ac_recall_masked'] - s['ac_recall_orig']:+.3f})"
        )
        lines.append(
            f"  AC precision:  {s['ac_precision_orig']:.3f} → {s['ac_precision_masked']:.3f}  "
            f"(Δ {s['ac_precision_masked'] - s['ac_precision_orig']:+.3f})"
        )
        lines.append(
            f"  AC F1:         {s['ac_f1_orig']:.3f} → {s['ac_f1_masked']:.3f}  "
            f"(Δ {s['ac_f1_masked'] - s['ac_f1_orig']:+.3f})"
        )
        lines.append(
            f"  Macro F1:      {s['macro_f1_orig']:.3f} → {s['macro_f1_masked']:.3f}  "
            f"(Δ {s['macro_f1_masked'] - s['macro_f1_orig']:+.3f})"
        )
        lines.append(
            f"  Flipped: {s['n_flipped']}   AC lost: {s['n_ac_lost']}   AC gained: {s['n_ac_gained']}"
        )
        lines.append(f"  McNemar p (AC vs not-AC): {s['mcnemar_ac_pvalue']:.4f}")
        lines.append("")

    fmt_block("GLOBAL", summary["global"])
    for ds, s in summary["by_dataset"].items():
        fmt_block(ds, s)

    summary_txt = Path(results_dir) / "summary.txt"
    summary_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {summary_txt}")
    print("\n" + "\n".join(lines))

    # Confusion-matrix diffs (row-normalized, masked − original)
    cm_o = confusion_matrix(ann, pred_orig, classes)
    cm_m = confusion_matrix(ann, pred_masked, classes)
    plot_cm_diff(
        os.path.join(plots_dir, "global_cm_diff"),
        "CM diff (masked − original) — Global",
        classes, cm_o, cm_m,
    )
    for ds in sorted(set(datasets)):
        idxs = [i for i, d in enumerate(datasets) if d == ds]
        ann_d = [ann[i] for i in idxs]
        po_d = [pred_orig[i] for i in idxs]
        pm_d = [pred_masked[i] for i in idxs]
        cm_o_d = confusion_matrix(ann_d, po_d, classes)
        cm_m_d = confusion_matrix(ann_d, pm_d, classes)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in ds)
        plot_cm_diff(
            os.path.join(plots_dir, f"{safe}_cm_diff"),
            f"CM diff (masked − original) — {ds}",
            classes, cm_o_d, cm_m_d,
        )
    print(f"[SAVED] plots → {plots_dir}")


if __name__ == "__main__":
    main()
