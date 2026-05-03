import os
import torch
import numpy as np


def save_fig(fig, out_path, *, dpi=300, bbox_inches=None):
    """Save a matplotlib figure as both a 300 DPI PNG and a vector PDF.

    `out_path` may carry a .png/.pdf extension or none — both forms are
    written next to each other with matching basenames.
    """
    base, ext = os.path.splitext(str(out_path))
    if ext.lower() not in (".png", ".pdf", ""):
        base = str(out_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    save_kwargs = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(f"{base}.png", dpi=dpi, **save_kwargs)
    fig.savefig(f"{base}.pdf", **save_kwargs)


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
