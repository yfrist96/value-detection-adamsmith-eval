import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_merged_csv(
    merged_path="data/merged.csv",
    out_root="data",
    test_size=0.2,
    random_state=42,
    combined_from=("asian", "indian", "joint", "ultra"),
):
    if not os.path.exists(merged_path):
        raise FileNotFoundError(merged_path)

    df = pd.read_csv(merged_path)

    # Validate required columns
    required = {"Dataset", "Text", "Annotated Value"}
    if not required.issubset(df.columns):
        raise ValueError(f"merged.csv must contain columns: {required}")

    df = df.copy()
    df["Text"] = df["Text"].fillna("").astype(str)
    df["Annotated Value"] = df["Annotated Value"].astype(str).str.strip()

    datasets = df["Dataset"].unique().tolist()
    print("Found datasets:", datasets)

    # ---------------------------------------
    # 1) Split each dataset separately
    # ---------------------------------------
    for ds in datasets:
        print(f"\nProcessing dataset: {ds}")
        sub = df[df["Dataset"] == ds].reset_index(drop=True)

        if len(sub) < 5:
            print(f"⚠️  Dataset '{ds}' has too few samples ({len(sub)}). Skipping.")
            continue

        try:
            train_df, test_df = train_test_split(
                sub,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=sub["Annotated Value"],
            )
        except ValueError as e:
            print(f"  ⚠️  Stratified split failed for '{ds}' ({e}). Falling back to random split.")
            train_df, test_df = train_test_split(
                sub,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )

        ds_dir = os.path.join(out_root, ds)
        os.makedirs(ds_dir, exist_ok=True)

        train_df.to_csv(os.path.join(ds_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(ds_dir, "test.csv"), index=False)

        print(f"  ✓ Train: {len(train_df)} rows → {ds}/train.csv")
        print(f"  ✓ Test:  {len(test_df)} rows → {ds}/test.csv")

    # ---------------------------------------
    # 2) Combined dataset = merge the per-dataset splits
    # ---------------------------------------
    print("\nProcessing combined dataset (merge per-dataset splits)")

    combined_dir = os.path.join(out_root, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    train_parts = []
    test_parts = []

    missing = []
    for ds in combined_from:
        train_path = os.path.join(out_root, ds, "train.csv")
        test_path = os.path.join(out_root, ds, "test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            missing.append(ds)
            continue

        train_parts.append(pd.read_csv(train_path))
        test_parts.append(pd.read_csv(test_path))

    if missing:
        raise FileNotFoundError(
            f"Missing per-dataset splits for: {missing}. "
            f"Expected files under data/<ds>/train.csv and data/<ds>/test.csv"
        )

    combined_train = pd.concat(train_parts, ignore_index=True)
    combined_test = pd.concat(test_parts, ignore_index=True)

    # Shuffle within each split for randomness
    combined_train = combined_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    combined_test = combined_test.sample(frac=1.0, random_state=random_state + 1).reset_index(drop=True)

    combined_train.to_csv(os.path.join(combined_dir, "train.csv"), index=False)
    combined_test.to_csv(os.path.join(combined_dir, "test.csv"), index=False)

    print(f"  ✓ Combined Train: {len(combined_train)} rows → combined/train.csv")
    print(f"  ✓ Combined Test:  {len(combined_test)} rows → combined/test.csv")


if __name__ == "__main__":
    split_merged_csv()
