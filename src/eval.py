import numpy as np
from sklearn.metrics import f1_score
from src.label_map import FINE_TO_COARSE

def evaluate_model(model, tokenizer, texts, coarse_labels, device):
    from src.utils import predict_logits

    logits = predict_logits(model, tokenizer, texts, device)
    fine_preds = np.argmax(logits, axis=1)

    # map predicted fine → coarse
    coarse_preds = [FINE_TO_COARSE[i] for i in fine_preds]

    # compare in coarse space
    f1_macro = f1_score(coarse_labels, coarse_preds, average="macro")
    f1_micro = f1_score(coarse_labels, coarse_preds, average="micro")

    return {
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro)
    }
