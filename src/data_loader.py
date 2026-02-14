import pandas as pd
import numpy as np
from src.label_map import COARSE_TO_FINE

NUM_FINE_LABELS = 20

def load_dataset(path, text_col="Text", label_col="Annotated Value", return_coarse=False):
    df = pd.read_csv(path)

    if text_col not in df or label_col not in df:
        raise ValueError(f"Missing columns in {path}. Need '{text_col}' and '{label_col}'")

    texts = df[text_col].fillna("").astype(str).tolist()
    coarse = df[label_col].astype(str).str.strip().tolist()

    bad = sorted(set(coarse) - set(COARSE_TO_FINE.keys()))
    if bad:
        raise ValueError(f"Unknown coarse labels in {path}: {bad}")

    # ONE-HOT fine target (single positive fine label per example)
    Y = np.zeros((len(coarse), NUM_FINE_LABELS), dtype=np.float32)
    for i, c in enumerate(coarse):
        fine_idx = COARSE_TO_FINE[c][0]   # representative fine label
        Y[i, fine_idx] = 1.0

    if return_coarse:
        return texts, Y, coarse
    return texts, Y
