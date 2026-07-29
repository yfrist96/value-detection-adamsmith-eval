#!/usr/bin/env python3
"""
Run one few-shot prompt template over the combined test set and dump per-item predictions.

This is a cleaned, runnable version of the notebook used for the results reported in the
paper. The reported runs were executed by a co-author on separate infrastructure (see the
paper's Compute appendix), against models served locally by ollama through its
OpenAI-compatible API. Differences from that notebook, all deliberate:

  * command-line arguments instead of hardcoded constants;
  * the host environment setup (proxy variables, user site-packages, in-notebook pip install)
    is removed, since it was specific to the machine the runs happened on;
  * `raw_response` records the model's actual response string. The notebook reconstructed
    that column from the parsed label, so in the delivered prediction dumps it is synthetic
    rather than a verbatim log.

Decoding is left at each model's own defaults and is *not* seeded: the seed in a prompt
filename selects which exemplars appear in that prompt, and does not control sampling. The
response format pins the output to a JSON object whose single property is one of the ten
reachable value codes, so a parse failure is not reachable by construction rather than
avoided by retrying.

Usage:
    python fewshot/run_fewshot.py --model qwen3:14b --prompt fewshot/prompts/prompt-42-ijh.txt
    for p in fewshot/prompts/*.txt; do
        python fewshot/run_fewshot.py --model qwen3:14b --prompt "$p"
    done

Writes predictions-<model>/<prompt stem>.csv, appending, so an interrupted run resumes where
it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

from openai import OpenAI

# The ten value codes with non-zero support in the corpus. Face and Humility never occur.
VALUE_CODES = ["SD", "ST", "HE", "AC", "PO", "SE", "TR", "CO", "BE", "UN"]

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string", "enum": VALUE_CODES}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}


def classify(client: OpenAI, model: str, prompt: str, text: str) -> tuple[str, str]:
    """Return (parsed label, raw response string) for one statement."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt.replace("{{text}}", text)}],
        response_format=RESPONSE_FORMAT,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)["value"], raw


def existing_rows(path: pathlib.Path) -> int:
    """Rows already written, so a re-invocation resumes instead of duplicating work."""
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="ollama model tag, e.g. qwen3:14b")
    ap.add_argument("--prompt", required=True, type=pathlib.Path,
                    help="prompt template containing the {{text}} placeholder")
    ap.add_argument("--test_csv", type=pathlib.Path, default=pathlib.Path("data/combined/test.csv"),
                    help="union of the four test splits; its Dataset column gives the target")
    ap.add_argument("--url", default="http://127.0.0.1:11434", help="ollama base URL")
    ap.add_argument("--out_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    # The prompt filename carries the two facts the output needs: prompt-<seed>[-<source>].txt,
    # where a missing source means the zero-shot template.
    parts = args.prompt.stem.split("-")
    seed = int(parts[1])
    source_row = parts[2] if len(parts) > 2 else "zeroshot"

    prompt = args.prompt.read_text(encoding="utf-8")
    client = OpenAI(api_key="ollama", base_url=f"{args.url}/v1")

    out_dir = args.out_dir or pathlib.Path(f"predictions-{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.prompt.stem}.csv"
    skip = existing_rows(out_path)
    print(f"{out_path}: {skip} rows already present")

    fieldnames = ["source_row", "seed", "target", "text", "gold", "pred", "raw_response", "model_id"]
    with out_path.open("a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        if skip == 0:
            writer.writeheader()
        with args.test_csv.open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i < skip:
                    continue
                pred, raw = classify(client, args.model, prompt, row["Text"])
                writer.writerow({
                    "source_row": source_row,
                    "seed": seed,
                    "target": row["Dataset"],
                    "text": row["Text"],
                    "gold": row["Annotated Value"],
                    "pred": pred,
                    "raw_response": raw,
                    "model_id": args.model,
                })
                out.flush()


if __name__ == "__main__":
    main()
