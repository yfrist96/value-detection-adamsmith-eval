#!/usr/bin/env python3
"""
Creates the prompts for a training dataset (few-shot) or not (zer-shot).

Committed as it was run, so the templates under prompts/ can be traced to the code that
produced them. Two things are worth knowing before re-running it:

  * `make_template` calls `random.shuffle(VALUES)` on the module-level list, so the value
    order in a given prompt depends on how many templates were generated before it, which in
    turn depends on the order `pathlib.Path.iterdir()` returns the dataset directories. The
    files under prompts/ are therefore the authoritative record of what the models saw: a
    re-run reproduces the exemplar sets exactly and may permute the value ordering.
  * The exemplar selection was checked against this repository's splits. Regenerating from
    data/<population>/train.csv reproduces every delivered prompt's exemplar set exactly, and
    no exemplar occurs in any test split.

Face and Humility are commented out of VALUES below: they have zero instances across all
2,699 annotations, so the reachable label space is ten values.

Usage:
    python create_prompts [--training-data data] [--out prompts]

Training data: a directory with subdirectories, each with a "train.csv" with (at least) the columns "Dataset", "Text", and "Annotated Value"

Outputs a prompt template for each of three seeds and each subdirectory (taken for few-shot examples) plus zero-shot as text with a single variable "{{text}}" to be replaced by the text to be classified (under --out):
"""

import argparse
import pathlib
import random
import csv

VALUES = [
  {"code": "SD", "name": "Self-Direction", "description": "independent thought and action: choosing, creating, exploring"},
  {"code": "ST", "name": "Stimulation", "description": "excitement, novelty, and challenge in life"},
  {"code": "HE", "name": "Hedonism", "description": "pleasure and sensuous gratification for oneself"},
  {"code": "AC", "name": "Achievement", "description": "personal success through demonstrating competence according to social standards"},
  {"code": "PO", "name": "Power", "description": "social status and prestige; control or dominance over people and resources"},
  {"code": "SE", "name": "Security", "description": "safety, harmony, and stability of society, of relationships, and of self"},
  {"code": "TR", "name": "Tradition", "description": "respect, commitment, and acceptance of the customs and ideas of one's culture or religion"},
  {"code": "CO", "name": "Conformity", "description": "restraint of actions and impulses likely to upset or harm others or violate social norms"},
  {"code": "BE", "name": "Benevolence", "description": "preserving and enhancing the welfare of those with whom one is in frequent personal contact"},
  {"code": "UN", "name": "Universalism", "description": "understanding, appreciation, tolerance, and protection for the welfare of all people and for nature"}
]

#  {"code": "FA", "name": "Face", "description": "maintaining one's public image and avoiding humiliation"},
#  {"code": "HU", "name": "Humility", "description": "recognizing one's insignificance in the larger scheme of things"},

PROMPT_TEMPLATE_INSTRUCTION = "You are annotating short free-text statements in which employees describe goals they want to advance at their workplace. Label the statement below with the single Schwartz basic value that is its dominant motivation. Use as output format a JSON object with single property \"value\" and the two letter code of the Schwartz basic value (as below) as its value.\n\nThe Schwartz basic values:\n\n"
PROMPT_TEMPLATE_VALUE = "- {code} ({name}): {description}."
PROMPT_TEMPLATE_VALUE_WITH_EXAMPLES = "{value}\n  Examples:\n"
PROMPT_TEMPLATE_EXAMPLE = "  - Statement: {text}\n    Result: {{\"value\": \"{code}\"}}\n"
PROMPT_TEMPLATE_END = "\n\nYour task:\n- Statement: {{text}}\n  Result: "

NUM_EXAMPLES = 4

def make_template(examples):
    prompt = PROMPT_TEMPLATE_INSTRUCTION
    random.shuffle(VALUES)
    for value in VALUES:
        value_prompt = PROMPT_TEMPLATE_VALUE.format(**value)
        if examples and value["code"] in examples:
            prompt += PROMPT_TEMPLATE_VALUE_WITH_EXAMPLES.format(value=value_prompt)
            for example in examples[value["code"]]:
                prompt += PROMPT_TEMPLATE_EXAMPLE.format(code=value["code"], text=example)
        else:
            prompt += value_prompt + "\n"
        prompt += "\n"
    prompt += PROMPT_TEMPLATE_END
    return prompt

def read_examples(path, max_examples_per_value = NUM_EXAMPLES):
    examples = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = row["Dataset"]
            value = row["Annotated Value"]
            if not value in examples:
                examples[value] = {}
            if not dataset in examples[value]:
                examples[value][dataset] = []
            examples[value][dataset].append(row["Text"])
    for value in examples.keys():
        for dataset in examples[value].keys():
            random.shuffle(examples[value][dataset])
        picked = []
        for i in range(max_examples_per_value):
            if len(picked) < max_examples_per_value:
                for dataset in examples[value].keys():
                    if i < len(examples[value][dataset]):
                        picked.append(examples[value][dataset][i])
                        if len(picked) == max_examples_per_value:
                            break
        examples[value] = picked
    return examples

def write_template(output_dir, seed, data_dir = None):
    random.seed(seed)
    examples = {}
    output = output_dir / f"prompt-{seed}.txt"
    if data_dir:
        examples = read_examples(data_dir / "train.csv")
        output = output_dir / f"prompt-{seed}-{data_dir.name}.txt"

    template = make_template(examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write(template)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training-data", type=pathlib.Path, default="data")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("prompts"))
    args = ap.parse_args()

    subdirectories = [path for path in args.training_data.iterdir()]
    subdirectories.append(None)
    for data_dir in subdirectories:
        for seed in [42, 43, 44]:
            write_template(args.out, seed, data_dir = data_dir)

if __name__ == "__main__":
    main()

