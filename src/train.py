import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

from src.data_loader import load_dataset
from src.eval import evaluate_model
from src.utils import pick_device
from src.plotting import plot_f1_curve


# ----------------------------
# Reproducibility (recommended)
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleDataset(Dataset):
    def __init__(self, texts, fine_labels, tokenizer):
        self.texts = texts
        self.fine_labels = fine_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=256,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # IMPORTANT: training labels must be fine-grained integer ids (0..19)
        item["labels"] = torch.tensor(self.fine_labels[idx], dtype=torch.float32)
        return item


def fine_tune_dataset(
    dataset_name,
    model_dir="models/adam-smith",
    data_root="data",
    num_epochs=10,
    batch_size=2,
    lr=2e-5,
    seed=42,
    grad_accum_steps=4,
):
    set_seed(seed)
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True
    ).to(device)

    # ---- Patch remote-code typo: "cirterion" vs "criterion" ----
    if hasattr(model, "criterion") and not hasattr(model, "cirterion"):
        model.cirterion = model.criterion

    train_path = f"{data_root}/{dataset_name}/train.csv"
    test_path = f"{data_root}/{dataset_name}/test.csv"

    # Load BOTH:
    # - fine labels for training
    # - coarse labels for evaluation (macro/micro F1 on SD/ST/...)
    train_texts, train_fine, train_coarse = load_dataset(train_path, return_coarse=True)
    test_texts, test_fine, test_coarse = load_dataset(test_path, return_coarse=True)

    train_ds = SimpleDataset(train_texts, train_fine, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    base_datasets = ["joint", "asian", "indian", "ultra"]

    # if we trained on combined, evaluate OOD on all base datasets
    if dataset_name == "combined":
        out_of_domain = base_datasets
    else:
        out_of_domain = [d for d in base_datasets if d != dataset_name]


    # Seed-aware output path: lets us run multi-seed without overwriting.
    # Layout: experiments/results/<dataset>/seed_<seed>/{metrics.csv,run_config.json,epoch_*}
    save_dir = f"experiments/results/{dataset_name}/seed_{seed}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("experiments/plots", exist_ok=True)

    # Save run config (helps reproducibility)
    run_config = {
        "dataset_name": dataset_name,
        "model_dir": model_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "device": str(device),
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    results = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dl, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            # IMPORTANT: Adam-Smith uses BCEWithLogitsLoss -> labels MUST be float and shape (B, 20)
            batch["labels"] = batch["labels"].float()

            out = model(**batch)

            # Support both HuggingFace outputs and Adam-Smith custom dict outputs
            if isinstance(out, dict) and "loss" in out:
                loss_val = out["loss"]
            elif hasattr(out, "loss"):
                loss_val = out.loss
            else:
                loss_val = out[0]

            loss = loss_val / grad_accum_steps

            loss.backward()

            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)


            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

        # if number of batches isn't divisible by grad_accum_steps
        if (step % grad_accum_steps) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        avg_train_loss = total_loss / max(1, n_batches)

        # -------------------
        # In-domain evaluation
        # -------------------
        model.eval()
        in_train = evaluate_model(model, tokenizer, train_texts, train_coarse, device)
        in_test = evaluate_model(model, tokenizer, test_texts, test_coarse, device)

        # -------------------
        # Out-of-domain eval
        # -------------------
        ood = {}
        for d in out_of_domain:
            test_file = f"{data_root}/{d}/test.csv"
            if os.path.exists(test_file):
                tt, tf, tc = load_dataset(test_file, return_coarse=True)
                ood[d] = evaluate_model(model, tokenizer, tt, tc, device)

        if str(device).startswith("mps"):
            torch.mps.empty_cache()

        # Save checkpoint
        epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

        # Save epoch metrics
        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "in_train_f1": in_train["f1_macro"],
            "in_test_f1": in_test["f1_macro"],
            "in_train_f1_micro": in_train["f1_micro"],
            "in_test_f1_micro": in_test["f1_micro"],
        }

        for d, scores in ood.items():
            row[f"ood_{d}_f1"] = scores["f1_macro"]
            row[f"ood_{d}_f1_micro"] = scores["f1_micro"]

        results.append(row)
        pd.DataFrame(results).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

        print(
            f"[{dataset_name}] Epoch {epoch}/{num_epochs} | "
            f"loss={avg_train_loss:.4f} | "
            f"in_test_f1={row['in_test_f1']:.4f}"
        )

    # Plot once at the end
    plot_f1_curve(dataset_name, seed=seed)
    print(f"[{dataset_name}] Complete! Saved results to: {save_dir}")


if __name__ == "__main__":
    # Single-seed (seed=42) convenience entry point: runs all five fine-tunes.
    # For multi-seed runs use:  python -m src.train_multi_seed --datasets joint,combined --seeds 42,43,44
    for d in ["joint", "asian", "indian", "ultra"]:
        fine_tune_dataset(d)

    # Then run the combined fine-tune (assumes data/combined/train.csv + test.csv exist)
    fine_tune_dataset("combined")
