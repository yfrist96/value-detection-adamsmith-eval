import os
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import save_fig


def plot_f1_curve(dataset_name, seed=None):
    """Plot fine-tuning macro-F1 curves for a single run.

    If `seed` is provided, reads from
        experiments/results/<dataset>/seed_<seed>/metrics.csv
    and writes
        experiments/plots/<dataset>_seed_<seed>_f1_plot.{png,pdf}.

    If `seed` is None, falls back to the legacy flat layout
        experiments/results/<dataset>/metrics.csv
        experiments/plots/<dataset>_f1_plot.{png,pdf}.
    """
    os.makedirs("experiments/plots", exist_ok=True)

    if seed is None:
        metrics_path = f"experiments/results/{dataset_name}/metrics.csv"
        plot_basename = f"experiments/plots/{dataset_name}_f1_plot"
        title = f"Fine-tuning results for {dataset_name} (macro F1)"
    else:
        metrics_path = f"experiments/results/{dataset_name}/seed_{seed}/metrics.csv"
        plot_basename = f"experiments/plots/{dataset_name}_seed_{seed}_f1_plot"
        title = f"Fine-tuning results for {dataset_name} (macro F1, seed={seed})"

    df = pd.read_csv(metrics_path)

    # Seeded runs log epochs 1..N. Splice in the epoch-0 (un-tuned base model)
    # row from the legacy flat metrics.csv so the curve starts at the
    # pre-fine-tuning baseline. Epoch 0 is encoding-independent, so reusing it
    # across single-positive and multi-positive runs is safe.
    if seed is not None:
        flat_path = f"experiments/results/{dataset_name}/metrics.csv"
        if os.path.exists(flat_path):
            flat_df = pd.read_csv(flat_path)
            e0 = flat_df[flat_df["epoch"] == 0]
            if not e0.empty:
                common = [c for c in e0.columns if c in df.columns]
                df = pd.concat([e0[common], df[common]], ignore_index=True)
                df = df.sort_values("epoch").reset_index(drop=True)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["in_train_f1"], label="Train F1 (macro)")
    plt.plot(df["epoch"], df["in_test_f1"], label="In-domain Test F1 (macro)")

    # Macro-only: the merged dataset is heavily class-imbalanced (AC ≈ 952 rows
    # vs. ST ≈ 60), so micro F1 collapses toward accuracy on the dominant
    # classes. Macro weights every Schwartz value equally, which is what we
    # actually care about. This also keeps these plots consistent with the
    # cross-domain heatmap, which is already macro-F1.
    ood_cols = [c for c in df.columns if c.startswith("ood_") and c.endswith("_f1")]
    for c in ood_cols:
        plt.plot(df["epoch"], df[c], label=c)

    plt.xlabel("Epoch")
    plt.ylabel("F1 (macro)")
    plt.title(title)
    plt.legend()
    save_fig(fig, plot_basename)
    plt.close(fig)
