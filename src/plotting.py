import os
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import save_fig


def plot_f1_curve(dataset_name):
    os.makedirs("experiments/plots", exist_ok=True)

    df = pd.read_csv(f"experiments/results/{dataset_name}/metrics.csv")

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
    plt.title(f"Fine-tuning results for {dataset_name} (macro F1)")
    plt.legend()
    save_fig(fig, f"experiments/plots/{dataset_name}_f1_plot")
    plt.close(fig)
