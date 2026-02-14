import argparse
import math
import os
import json
from collections import Counter
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Uses your project-wide mapping
from src.label_map import LABELS, COARSE_TO_FINE, FINE_TO_COARSE


# Force stable order (matches SVS-style order)
COARSE_LABELS = ["SD", "ST", "HE", "AC", "PO", "FA", "SE", "TR", "CO", "HU", "BE", "UN"]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_coarse(x: str) -> str:
    """
    Normalize the ground-truth 'Annotated Value' into a coarse SVS code: SD, ST, HE, ...
    Returns "" if unknown.
    """
    if not isinstance(x, str):
        return ""
    s = x.strip().upper()
    if not s:
        return ""

    # Already a coarse code
    if s in COARSE_TO_FINE:
        return s

    # Some datasets might store names
    name_to_coarse = {
        "SELF-DIRECTION": "SD",
        "SELF DIRECTION": "SD",
        "STIMULATION": "ST",
        "HEDONISM": "HE",
        "ACHIEVEMENT": "AC",
        "POWER": "PO",
        "FACE": "FA",
        "SECURITY": "SE",
        "TRADITION": "TR",
        "CONFORMITY": "CO",
        "HUMILITY": "HU",
        "BENEVOLENCE": "BE",
        "UNIVERSALISM": "UN",
    }
    key = s.replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    return name_to_coarse.get(key, "")


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def predict_batch(model, tokenizer, texts: List[str], device) -> np.ndarray:
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
        # Adam-Smith custom model returns logits/probs under "output"
        scores = out["output"].detach().cpu().numpy()  # (B, 20)
    return scores


def aggregate_to_coarse_scores(scores_20: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    scores_20: (N,20)
    returns: (N,12) coarse scores where each coarse score is MAX over its fine indices.
    """
    coarse_scores = np.full((scores_20.shape[0], len(COARSE_LABELS)), -np.inf, dtype=float)
    for ci, c in enumerate(COARSE_LABELS):
        fine_idxs = COARSE_TO_FINE[c]
        coarse_scores[:, ci] = np.max(scores_20[:, fine_idxs], axis=1)
    coarse_scores[coarse_scores == -np.inf] = -1e9
    return coarse_scores, COARSE_LABELS


def plot_bar_counts(out_png: str, title: str, x_label: str, y_label: str, labels: List[str], values: List[int]):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_confusion(out_png: str, title: str, classes: List[str], cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted label (coarse)")
    ax.set_ylabel("Annotated label (coarse)")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_bar_float(out_png: str, title: str, x_label: str, y_label: str, labels: List[str], values: List[float]):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def compute_metrics_for_slice(
    ann_coarse: List[str],
    top1_coarse: List[str],
    topk_coarse: List[List[str]],
    thr_coarse_sets: List[Set[str]],
    classes: List[str],
    top_k: int
) -> Dict:
    n = len(ann_coarse)
    if n == 0:
        return {"total_rows": 0}

    top1_correct = [(ann_coarse[i] == top1_coarse[i]) if ann_coarse[i] else False for i in range(n)]
    hit_any = [(ann_coarse[i] in thr_coarse_sets[i]) if ann_coarse[i] else False for i in range(n)]
    hit_at_k = [(ann_coarse[i] in set(topk_coarse[i])) if ann_coarse[i] else False for i in range(n)]

    top1_acc = float(np.mean(top1_correct))
    hit_any_rate = float(np.mean(hit_any))
    hit_at_k_rate = float(np.mean(hit_at_k))

    # Confusion matrix on valid classes
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for a, t in zip(ann_coarse, top1_coarse):
        if a in idx and t in idx:
            cm[idx[a], idx[t]] += 1

    # Distributions
    ann_counts = Counter([a for a in ann_coarse if a in idx])
    top_counts = Counter([t for t in top1_coarse if t in idx])

    eps = 1e-12
    ann_vec = np.array([ann_counts.get(c, 0) for c in classes], dtype=float)
    top_vec = np.array([top_counts.get(c, 0) for c in classes], dtype=float)

    ann_prob = (ann_vec + eps) / (ann_vec.sum() + eps * len(ann_vec))
    top_prob = (top_vec + eps) / (top_vec.sum() + eps * len(top_vec))

    dist_mse = float(np.mean((ann_prob - top_prob) ** 2))
    dist_kl = kl_divergence(ann_prob, top_prob)

    # Per-class PRF: single-label truth vs thresholded multi-label coarse set
    per_class = {}
    for c in classes:
        tp = sum(1 for i in range(n) if ann_coarse[i] == c and (c in thr_coarse_sets[i]))
        fp = sum(1 for i in range(n) if ann_coarse[i] != c and (c in thr_coarse_sets[i]))
        fn = sum(1 for i in range(n) if ann_coarse[i] == c and (c not in thr_coarse_sets[i]))
        prec, rec, f1 = prf_from_counts(tp, fp, fn)
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(ann_counts.get(c, 0))}

    # Macro and weighted F1
    f1s = []
    weighted_f1_sum = 0.0
    total_support = 0

    for c in classes:
        f1 = per_class[c]["f1"]
        support = per_class[c]["support"]
        f1s.append(f1)
        weighted_f1_sum += f1 * support
        total_support += support

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(weighted_f1_sum / total_support) if total_support > 0 else 0.0

    return {
        "total_rows": n,
        "top1_accuracy_coarse": top1_acc,
        f"hit_at_{top_k}_coarse": hit_at_k_rate,
        "hit_any_thresholded_set_coarse": hit_any_rate,
        "distribution_mse_annotated_vs_top1": dist_mse,
        "distribution_kl_annotated_vs_top1": dist_kl,
        "per_class_vs_thresholded_multilabel": per_class,
        "confusion_matrix": cm.tolist(),  # optional: helpful for debugging
        "macro_f1_coarse": macro_f1,
        "weighted_f1_coarse": weighted_f1,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_csv", required=True)

    ap.add_argument("--dataset_col", default="Dataset")
    ap.add_argument("--text_col", default="Text")
    ap.add_argument("--annotated_col", default="Annotated Value")

    ap.add_argument("--threshold", type=float, default=0.25, help="Threshold for multi-label Hit@Any")
    ap.add_argument("--top_k", type=int, default=3, help="k for Hit@k in coarse ranking")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--apply_sigmoid", action="store_true",
                    help="Apply sigmoid to scores before thresholding (recommended if outputs are logits)")

    # your requested folder name
    ap.add_argument("--experiments_dir", default="experiments")

    args = ap.parse_args()

    results_dir = os.path.join(args.experiments_dir, "results")
    plots_dir = os.path.join(args.experiments_dir, "plots")
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    pred_csv_path = os.path.join(results_dir, "merged__base_adamsmith__predictions.csv")
    eval_rows_path = os.path.join(results_dir, "merged__base_adamsmith__eval_rows.csv")
    metrics_txt_path = os.path.join(results_dir, "merged__base_adamsmith__metrics.txt")
    metrics_json_path = os.path.join(results_dir, "merged__base_adamsmith__metrics.json")

    df = pd.read_csv(args.input_csv)
    for col in [args.dataset_col, args.text_col, args.annotated_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, trust_remote_code=True)
    model.eval()

    device = pick_device()
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # Predict (N,20)
    texts = df[args.text_col].fillna("").astype(str).tolist()
    all_scores_20 = []
    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(texts), bs), total=math.ceil(len(texts) / bs), desc="Predicting (base Adam-Smith)"):
        batch = texts[i:i + bs]
        scores = predict_batch(model, tokenizer, batch, device)
        all_scores_20.append(scores)

    scores_20 = np.vstack(all_scores_20) if all_scores_20 else np.zeros((0, len(LABELS)), dtype=float)

    # SANITY: score range + whether sigmoid is appropriate
    smin, smax = float(scores_20.min()), float(scores_20.max())
    print(f"[SANITY] Raw score range: min={smin:.4f}, max={smax:.4f}")

    # For threshold-based multilabel sets, we may want sigmoid
    if args.apply_sigmoid and scores_20.size > 0:
        scores_for_threshold = 1.0 / (1.0 + np.exp(-scores_20))
        print("[INFO] Applied sigmoid for thresholding.")
    else:
        scores_for_threshold = scores_20

    # Coarse scores for ranking (max over fine indices)
    coarse_scores, coarse_names = aggregate_to_coarse_scores(scores_20)

    # Normalize annotations into coarse codes
    raw_ann = df[args.annotated_col].fillna("").astype(str).tolist()
    ann_coarse_all = [normalize_coarse(x) for x in raw_ann]

    # SANITY: show unmapped labels
    unmapped = sum(1 for x in ann_coarse_all if not x)
    print(f"[SANITY] Unmapped Annotated Value rows: {unmapped}/{len(ann_coarse_all)} ({unmapped/len(ann_coarse_all):.2%})")
    if unmapped > 0:
        bad = pd.Series([raw_ann[i] for i, x in enumerate(ann_coarse_all) if not x]).value_counts().head(15)
        print("[SANITY] Top unmapped raw labels:")
        print(bad.to_string())

    # Per-row predictions
    top1_coarse_all: List[str] = []
    topk_coarse_all: List[List[str]] = []
    thr_coarse_sets_all: List[Set[str]] = []
    thr_labels20_all: List[str] = []

    for i in range(len(df)):
        # thresholding on (possibly sigmoid-ed) scores
        s_thr = scores_for_threshold[i]
        fine_idxs_thr = [j for j, v in enumerate(s_thr) if v >= args.threshold]
        labs20 = [LABELS[j] for j in fine_idxs_thr]
        thr_labels20_all.append(", ".join(labs20))

        thr_coarse = {FINE_TO_COARSE[j] for j in fine_idxs_thr if j in FINE_TO_COARSE}
        thr_coarse_sets_all.append(thr_coarse)

        # ranking based on coarse scores
        sp = coarse_scores[i]
        order = np.argsort(-sp)
        top1_coarse_all.append(coarse_names[int(order[0])] if len(order) else "")
        topk = [coarse_names[int(idx)] for idx in order[: max(1, int(args.top_k))]]
        topk_coarse_all.append(topk)

    # Save predictions CSV (preserve Dataset)
    out_df = df.copy()
    out_df.insert(0, "Source_Dataset", out_df[args.dataset_col].astype(str))

    out_df["base_adamsmith__threshold"] = args.threshold
    out_df["base_adamsmith__top_k"] = args.top_k

    out_df["base_adamsmith__annotated_coarse"] = ann_coarse_all
    out_df["base_adamsmith__predicted_top1_coarse"] = top1_coarse_all
    out_df["base_adamsmith__predicted_topk_coarse_ranked"] = ["; ".join(x) for x in topk_coarse_all]
    out_df["base_adamsmith__predicted_coarse_set_thresholded"] = ["; ".join(sorted(s)) for s in thr_coarse_sets_all]
    out_df["base_adamsmith__predicted_labels20_thresholded"] = thr_labels20_all

    out_df.to_csv(pred_csv_path, index=False)

    # Save eval rows
    top1_correct = [(ann_coarse_all[i] == top1_coarse_all[i]) if ann_coarse_all[i] else False for i in range(len(df))]
    hit_any = [(ann_coarse_all[i] in thr_coarse_sets_all[i]) if ann_coarse_all[i] else False for i in range(len(df))]
    hit_at_k = [(ann_coarse_all[i] in set(topk_coarse_all[i])) if ann_coarse_all[i] else False for i in range(len(df))]

    eval_rows = pd.DataFrame({
        "Source_Dataset": df[args.dataset_col].astype(str).tolist(),
        "Text": df[args.text_col].astype(str).tolist(),
        "Annotated_Coarse": ann_coarse_all,
        "Predicted_Top1_Coarse": top1_coarse_all,
        "Predicted_TopK_Coarse": ["; ".join(x) for x in topk_coarse_all],
        "Predicted_Thresholded_Coarse_Set": ["; ".join(sorted(s)) for s in thr_coarse_sets_all],
        "Top1_Correct": [int(x) for x in top1_correct],
        f"HitAt{args.top_k}": [int(x) for x in hit_at_k],
        "HitAny_ThresholdedSet": [int(x) for x in hit_any],
    })
    eval_rows.to_csv(eval_rows_path, index=False)

    # Global metrics
    classes = COARSE_LABELS[:]
    metrics_global = compute_metrics_for_slice(
        ann_coarse=ann_coarse_all,
        top1_coarse=top1_coarse_all,
        topk_coarse=topk_coarse_all,
        thr_coarse_sets=thr_coarse_sets_all,
        classes=classes,
        top_k=args.top_k
    )

    # Per-dataset metrics
    metrics_by_dataset: Dict[str, Dict] = {}
    for ds_name, df_ds in df.groupby(args.dataset_col):
        idxs = df_ds.index.tolist()
        ann_ds = [ann_coarse_all[i] for i in idxs]
        top1_ds = [top1_coarse_all[i] for i in idxs]
        topk_ds = [topk_coarse_all[i] for i in idxs]
        thr_ds = [thr_coarse_sets_all[i] for i in idxs]

        metrics_by_dataset[str(ds_name)] = compute_metrics_for_slice(
            ann_coarse=ann_ds,
            top1_coarse=top1_ds,
            topk_coarse=topk_ds,
            thr_coarse_sets=thr_ds,
            classes=classes,
            top_k=args.top_k
        )

    metrics = {
        "input_csv": args.input_csv,
        "columns": {
            "dataset_col": args.dataset_col,
            "text_col": args.text_col,
            "annotated_col": args.annotated_col
        },
        "threshold": args.threshold,
        "top_k": args.top_k,
        "apply_sigmoid": bool(args.apply_sigmoid),
        "label_space": "coarse",
        "coarse_labels": classes,
        "global": metrics_global,
        "by_dataset": metrics_by_dataset,
        "files": {
            "predictions_csv": pred_csv_path,
            "eval_rows_csv": eval_rows_path,
            "plots_dir": plots_dir,
            "metrics_txt": metrics_txt_path,
            "metrics_json": metrics_json_path
        }
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Write TXT summary
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("# Merged Dataset — Base Adam-Smith Evaluation (COARSE: SVS)\n\n")
        f.write(f"Input CSV: {args.input_csv}\n")
        f.write(f"Columns: Dataset='{args.dataset_col}', Text='{args.text_col}', Annotated='{args.annotated_col}'\n")
        f.write(f"Threshold (Hit@Any): {args.threshold}\n")
        f.write(f"Top-k (Hit@k): {args.top_k}\n")
        f.write(f"Apply sigmoid: {args.apply_sigmoid}\n\n")

        g = metrics_global
        f.write("## Global metrics\n")
        f.write(f"Total rows: {g.get('total_rows', 0)}\n")
        f.write(f"Top-1 Accuracy (coarse): {g.get('top1_accuracy_coarse', 0.0):.4f}\n")
        f.write(f"Hit@{args.top_k} (coarse): {g.get(f'hit_at_{args.top_k}_coarse', 0.0):.4f}\n")
        f.write(f"Hit@Any (thresholded set, coarse): {g.get('hit_any_thresholded_set_coarse', 0.0):.4f}\n")
        f.write(f"Distribution MSE (annotated vs top-1): {g.get('distribution_mse_annotated_vs_top1', 0.0):.6f}\n")
        f.write(f"Distribution KL  (annotated vs top-1): {g.get('distribution_kl_annotated_vs_top1', 0.0):.6f}\n\n")

        f.write("\n## Per-class (coarse, vs thresholded multilabel)\n")
        for c in classes:
            d = g["per_class_vs_thresholded_multilabel"][c]
            f.write(f"- {c:3s} | "
                    f"P: {d['precision']:.3f}  "
                    f"R: {d['recall']:.3f}  "
                    f"F1: {d['f1']:.3f}  "
                    f"Support: {d['support']}\n")

        f.write(f"\nMacro-F1 (coarse): {g.get('macro_f1_coarse',0.0):.4f}\n")
        f.write(f"Weighted-F1 (coarse): {g.get('weighted_f1_coarse',0.0):.4f}\n")

        f.write("## Per-dataset metrics\n")
        for ds_name, md in metrics_by_dataset.items():
            f.write(f"- {ds_name}: N={md.get('total_rows',0)}, "
                    f"Top1={md.get('top1_accuracy_coarse',0.0):.4f}, "
                    f"Hit@{args.top_k}={md.get(f'hit_at_{args.top_k}_coarse',0.0):.4f}, "
                    f"HitAny={md.get('hit_any_thresholded_set_coarse',0.0):.4f}\n")

        f.write("\nSaved outputs:\n")
        f.write(f"- Predictions CSV: {pred_csv_path}\n")
        f.write(f"- Eval rows CSV:   {eval_rows_path}\n")
        f.write(f"- Metrics TXT:     {metrics_txt_path}\n")
        f.write(f"- Metrics JSON:    {metrics_json_path}\n")

    # -----------------------------
    # Plots (global)
    # -----------------------------
    ann_counts = Counter([a for a in ann_coarse_all if a])
    top_counts = Counter([t for t in top1_coarse_all if t])
    multi_counts = Counter([c for s in thr_coarse_sets_all for c in s])

    plot_bar_counts(
        out_png=os.path.join(plots_dir, "merged__annotated_coarse_distribution__counts.png"),
        title="Merged dataset: Annotated label distribution (COARSE: SVS)",
        x_label="Annotated label (coarse: SD/ST/…/UN)",
        y_label="Count of rows",
        labels=classes,
        values=[ann_counts.get(c, 0) for c in classes],
    )

    plot_bar_counts(
        out_png=os.path.join(plots_dir, "merged__predicted_top1_coarse_distribution__counts.png"),
        title="Merged dataset: Predicted TOP-1 label distribution (COARSE: SVS)",
        x_label="Predicted TOP-1 label (coarse: SD/ST/…/UN)",
        y_label="Count of rows",
        labels=classes,
        values=[top_counts.get(c, 0) for c in classes],
    )

    plot_bar_counts(
        out_png=os.path.join(plots_dir, "merged__predicted_thresholded_coarse_frequency__rows.png"),
        title=f"Merged dataset: Predicted multi-label frequency by coarse label (threshold ≥ {args.threshold})",
        x_label="Predicted coarse label",
        y_label="Number of rows where coarse label appears in thresholded set",
        labels=classes,
        values=[multi_counts.get(c, 0) for c in classes],
    )

    # Confusion
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for a, t in zip(ann_coarse_all, top1_coarse_all):
        if a in idx and t in idx:
            cm[idx[a], idx[t]] += 1

    plot_confusion(
        out_png=os.path.join(plots_dir, "merged__confusion_matrix__annotated_coarse_vs_predicted_top1_coarse.png"),
        title="Confusion matrix: Annotated coarse vs Predicted TOP-1 coarse",
        classes=classes,
        cm=cm,
    )

    # Top-1 by dataset
    ds_names = sorted(metrics_by_dataset.keys())
    ds_top1 = [metrics_by_dataset[d].get("top1_accuracy_coarse", 0.0) for d in ds_names]
    plot_bar_float(
        out_png=os.path.join(plots_dir, "merged__top1_accuracy_by_source_dataset__bar.png"),
        title="Merged dataset: Top-1 Accuracy (COARSE: SVS) by Source Dataset",
        x_label="Source Dataset",
        y_label="Top-1 Accuracy (0–1)",
        labels=ds_names,
        values=ds_top1,
    )

    print("\n✅ Done.")
    print(f"[SAVED] Predictions: {pred_csv_path}")
    print(f"[SAVED] Eval rows:   {eval_rows_path}")
    print(f"[SAVED] Metrics:     {metrics_txt_path}")
    print(f"[SAVED] Plots:       {plots_dir}")


if __name__ == "__main__":
    main()
