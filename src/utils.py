import torch
import numpy as np


def pick_device():
    # Prefer CUDA, then MPS (Apple), else CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_logits(model, tokenizer, texts, device, batch_size=16):
    model.eval()
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"])

        # Adam-Smith custom model returns logits under "output"
        if isinstance(out, dict) and "output" in out:
            logits = out["output"]
        elif hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, dict) and "logits" in out:
            logits = out["logits"]
        else:
            logits = out[0]

        all_logits.append(logits.detach().cpu().numpy())

    return np.concatenate(all_logits, axis=0)
