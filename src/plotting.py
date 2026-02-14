import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_f1_curve(dataset_name):
    import os
    os.makedirs("experiments/plots", exist_ok=True)
    
    df = pd.read_csv(f"experiments/results/{dataset_name}/metrics.csv")

    plt.figure(figsize=(10,5))
    plt.plot(df["epoch"], df["in_train_f1"], label="Train F1")
    plt.plot(df["epoch"], df["in_test_f1"], label="In-domain Test F1")

    ood_cols = [c for c in df.columns if c.startswith("ood")]
    for c in ood_cols:
        plt.plot(df["epoch"], df[c], label=c)

    plt.xlabel("Epoch")
    plt.ylabel("F1 (macro)")
    plt.title(f"Fine-tuning results for {dataset_name}")
    plt.legend()
    plt.savefig(f"experiments/plots/{dataset_name}_f1_plot.png", dpi=200)
    plt.close()
