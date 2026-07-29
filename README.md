# ValueDetection

## Cross-Domain Prediction of Schwartz Human Values

This repository implements a full experimental pipeline for evaluating and fine-tuning the **Adam-Smith value prediction model** across multiple datasets:

- Asians  
- Indians  
- IJH
- Ultra Orthodox Female Teachers

The project supports:

- Dataset analysis and visualization  
- Per-dataset train/test splits  
- In-domain and out-of-domain evaluation  
- Fine-tuning experiments  
- Base model evaluation on a merged dataset  
- Per-class Precision / Recall / F1  
- Hit@k and Hit@Any metrics  
- Distribution-level evaluation (MSE, KL divergence)  

> **Plot outputs.** Every plot in this pipeline is written as both a 300 DPI PNG and a vector PDF (matching basenames, side by side). PDF output is produced via the shared `save_fig` helper in `src/utils.py` and is suitable for direct inclusion in academic papers.

---

## 📂 Project Structure

```
ValueDetection/
│
├── data/
│   ├── merged.csv
│   │
│   ├── data_analysis/
│   │   ├── plots/
│   │   ├── summary.json
│   │   ├── split_datasets.txt
│   │   └── *.csv
│   │
│   ├── ijh/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── asian/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── indian/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── ultra/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   └── combined/
│       ├── train.csv         # union of the four per-population train sets
│       └── test.csv          # union of the four per-population test sets
│
├── models/
│   └── adam-smith/
│
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── train_multi_seed.py
│   ├── aggregate_seeds.py
│   ├── eval.py
│   ├── plotting.py
│   ├── cross_domain_heatmaps.py
│   ├── misclassification_ijh_test.py       # per-population error analysis (IJH, Ultra, ...)
│   ├── circumplex_error_analysis.py        # Schwartz circumplex distance scoring
│   ├── lexical_exhibits.py                 # log-odds distinctive vocabulary per cell
│   ├── register_distance.py                # lexical distance per population pair vs transfer
│   ├── dump_predictions.py                 # per-item predictions + macro-F1 convention check
│   ├── score_fewshot.py                    # few-shot LLM matrix, scored against the fine-tuned one
│   ├── ablate_achievement_vocab.py
│   ├── split_datasets.py
│   ├── data_analysis.py
│   ├── run_merged_base_adamsmith_eval.py
│   ├── label_map.py
│   └── utils.py
│
├── experiments/
│   ├── results/      # Metrics, logs, JSON summaries
│   │   ├── <dataset>/
│   │   │   ├── metrics.csv         # epoch-0 baseline (legacy flat layout, kept
│   │   │   │                       #   for the cross-domain heatmap's base row)
│   │   │   └── seed_<seed>/        # one subdir per training seed
│   │   │       ├── metrics.csv     #   per-epoch in/OOD F1 for that seed
│   │   │       └── epoch_<N>/      #   model checkpoint per epoch
│   │   ├── <dataset>_seed_summary.csv         # mean ± std across seeds
│   │   ├── cross_domain_macro_f1_matrix.csv   # the transfer matrix (seed means)
│   │   ├── transfer_per_seed.csv              # the same matrix, one row per run
│   │   ├── register_distance.csv              # per-pair lexical distance measures
│   │   ├── register_distance_correlations.json  # Spearman rho vs transfer
│   │   ├── misclf_<dataset>_test_*.csv        # per-population error tables + confusion matrices
│   │   ├── misclf_<dataset>_test_misclassified_scored.csv  # + Schwartz circumplex distance
│   │   ├── misclf_<dataset>_test_attractor_summary.csv     # per-class attractor counts
│   │   ├── circumplex_summary.csv             # cross-population circumplex overview
│   │   ├── lexical_distinctive_by_pop_value.csv  # full distinctive-vocabulary table
│   │   ├── lexical_<pop>_<value>_top.csv          # per-cell highlights (Ultra-BE, IJH-UN, ...)
│   │   ├── ablation_achievement/                    # base model, full corpus
│   │   ├── ablation_achievement_<dataset>_finetuned/   # per-population epoch-10, full corpus
│   │   ├── ablation_achievement_ijh_finetuned/      # IJH epoch-10, IJH test split
│   │   ├── ablation_achievement_ijh_finetuned_train/   # IJH epoch-10, IJH train split
│   │   └── ablation_summary_per_setting.csv         # cross-setting AC-share / F1 deltas
│   ├── train.txt     # Training CLI output
│   └── plots/        # Evaluation plots and charts
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset Setup

The corpus is distributed via Zenodo: **https://zenodo.org/records/20324552**

Download the dataset file (`dataset.csv`, 2,699 rows) and save it as **`data/merged.csv`**:

```bash
mkdir -p data
# download dataset.csv from the Zenodo record above, then:
mv ~/Downloads/dataset.csv data/merged.csv
```

`data/merged.csv` must have exactly three columns — `Dataset`, `Text`, `Annotated Value` —
where `Dataset` is one of the four populations: `Asian`, `Indian`, `IJH`, `Ultra`. The
per-population train/test splits and the `combined` split are generated from this file by
`src/split_datasets.py` (see step 2️⃣ below); everything downstream reads from those splits.

---

## Model Setup

The base classifier is **Adam-Smith** ([Schroter et al., 2023](https://aclanthology.org/2023.semeval-1.196/)),
the top system in SemEval-2023 Task 4 (ValueEval) — a DeBERTa-large multi-label value
detector. It is released on HuggingFace as
[`tum-nlp/Deberta_Human_Value_Detector`](https://huggingface.co/tum-nlp/Deberta_Human_Value_Detector)
under the **OpenRAIL++** license. Download it into `models/adam-smith/` (the path the
scripts default to):

```bash
pip install -r requirements.txt
huggingface-cli download tum-nlp/Deberta_Human_Value_Detector --local-dir models/adam-smith
```

The model ships custom modeling code, so the scripts load it with `trust_remote_code=True`.
All fine-tuned checkpoints and result files under `experiments/` are **regenerated** by the
pipeline below (training uses fixed seeds 42/43/44); they are not distributed.

---

## Experimental Pipeline

The project follows a structured experimental workflow:

### 1️⃣ Dataset Analysis

Analyze dataset distributions and label balance:

```bash
python src/data_analysis.py
```

Outputs are saved to:

```
data/data_analysis/
```

---

### 2️⃣ Dataset Splitting

Split each dataset into train/test sets:

```bash
python src/split_datasets.py
```

This creates:

```
data/<dataset>/train.csv
data/<dataset>/test.csv
```

Splits are created for:

- IJH  
- Asian  
- Indian  
- Ultra  
- Combined (merged)  

---

### 3️⃣ Fine-Tuning Experiments

Train and evaluate models for each configuration:

```bash
python -m src.train
```

Training details:

- AdamW, lr=2e-5, micro-batch size 2 with gradient accumulation over 4 steps
  (effective batch size 8), max sequence length 256 tokens
- 10 epochs per configuration, no early stopping
- Evaluation after each epoch on (i) the run's own training set, (ii) its
  held-out in-domain test set, and (iii) the held-out test sets of all other
  populations as out-of-domain (OOD) probes
- **Multi-positive label encoding.** Each annotated coarse label is supervised
  by setting *all* of its corresponding fine-grained sub-labels to a positive
  target under the BCE objective (e.g., a coarse `BE` annotation sets both
  `Benevolence: caring` and `Benevolence: dependability` to 1). This reflects
  the hierarchical structure of the label space and avoids the asymmetry of
  picking a single representative fine sub-label. See `src/data_loader.py`.
- F1 curves are reported in **macro F1** only: the merged corpus is heavily
  class-imbalanced (AC ≈ 952 rows vs. ST ≈ 60), so micro F1 collapses toward
  accuracy on the dominant classes. Macro weights every Schwartz value equally,
  which is what we actually care about, and matches the cross-domain heatmap
  below.

Results are saved to:

```
experiments/results/<dataset>/seed_<seed>/metrics.csv
experiments/results/<dataset>/seed_<seed>/epoch_<N>/    # model checkpoint per epoch
experiments/plots/<dataset>_seed_<seed>_f1_plot.{png,pdf}
```

The default `python -m src.train` entry point uses `seed=42`. A small flat-layout
`metrics.csv` is preserved at `experiments/results/<dataset>/metrics.csv` to
provide the epoch-0 (pre-fine-tuning) baseline that
`cross_domain_heatmaps.py` reads for the heatmap's `base` row; the per-epoch
fine-tuned numbers always come from the seed-aware paths.

To produce plots that include epoch 0 (pre-fine-tuning), execute:

```bash
python -c "from src.plotting import plot_f1_curve; plot_f1_curve('<ds_name>', seed=42)"
```

#### Multi-seed runs (variance reporting)

Every cell of the transfer matrix is a mean over seeds, not a single run. All
five configurations are trained with seeds 42/43/44, and Ultra additionally with
45/46 (see the note below):

```bash
# 1. Train every configuration with three seeds.
python -m src.train_multi_seed --datasets asian,indian,ijh,ultra,combined --seeds 42,43,44

# 2. Two extra seeds for Ultra, which had one diverged run.
python -m src.train_multi_seed --datasets ultra --seeds 45,46

# 3. Aggregate last-epoch metrics into mean ± std per configuration.
python -m src.aggregate_seeds --datasets asian,indian,ijh,ultra,combined
```

This writes one `experiments/results/<dataset>_seed_summary.csv` per
configuration and prints a per-metric mean ± std table to stdout.

**Excluded run: `ultra` seed 43.** It diverged at the first epoch to constant
prediction of the majority class and never recovered, ending at a *training*
macro F1 of 0.055 — exactly the score of always predicting Benevolence on that
split. Since it never fit the data it was trained on, it is an optimization
failure rather than a weak model, and `aggregate_seeds.py` drops it by default
via `--exclude ultra:43`. The criterion is training-set performance and never
looks at the test set. Pass `--exclude ""` to keep every run and see the
difference: including it drags Ultra's in-domain mean from 0.592 ± 0.035 to
0.485 ± 0.242. Training uses plain AdamW with no warmup, no LR schedule, and no
gradient clipping, which is the likely cause; 16 of the 17 runs converged.

---

### 4️⃣ Base Model Evaluation (Merged Dataset)

Evaluate the **base Adam-Smith model** (no fine-tuning) on the merged dataset:

```bash
python -m src.run_merged_base_adamsmith_eval \
  --model_dir models/adam-smith \
  --input_csv data/merged.csv \
  --top_k 3 \
  --threshold 0.25
```

Results are saved to:

```
experiments/results/merged__base_adamsmith__eval_rows.csv
experiments/results/merged__base_adamsmith__predictions.csv
experiments/results/merged__base_adamsmith__metrics.json
experiments/plots/merged__*.{png,pdf}
```

---

### 5️⃣ Cross-Domain Generalization Heatmap (Macro-F1)

Generate a **cross-domain generalization heatmap** summarizing how well each fine-tuned model transfers across datasets.

This script reads the per-run logs saved by `src/train.py`:


It uses the **last epoch** of each run, averaged across seeds via
`<dataset>_seed_summary.csv`, and constructs a Macro-F1 matrix where:

- Rows = training dataset  
- Columns = evaluation dataset  
- Diagonal cells = in-domain `in_test_f1`  
- Off-diagonal cells = out-of-domain `ood_<dataset>_f1`  

Run:

```bash
python -m src.cross_domain_heatmaps
```

The matrix columns and the per-population rows are the four populations
(`asian,indian,ijh,ultra`); the `base` row (epoch 0) and the **Combined** row are
added automatically. The Combined row is sourced from
`experiments/results/combined_seed_summary.csv`, so it appears only after the
multi-seed + aggregate steps above have produced that file (otherwise the heatmap
renders without it). Do **not** add `combined` to `--datasets` — that is the eval
target list, and the union model is not evaluated on a `combined` test set.

Every fine-tuned row is a seed mean, taken from `<dataset>_seed_summary.csv`; if
a summary file is missing the script falls back to that dataset's seed-42 run
alone, so run the aggregate step first. The `base` row is read from epoch 0,
which is the un-tuned model and therefore seed-independent.

Results are saved to:

```
experiments/results/cross_domain_macro_f1_matrix.csv
experiments/results/cross_domain_macro_f1_matrix.json
experiments/plots/cross_domain_macro_f1_heatmap.{png,pdf}
```

---

### 6️⃣ Misclassification Analysis (per-population)

Run a focused **in-domain misclassification analysis** on a population's test set
to understand *what confuses the model* beyond aggregate F1 scores.

The script:

- Loads a fine-tuned per-population checkpoint (by default, the latest `epoch_*`
  under `experiments/results/<dataset>/seed_42/`; pass `--model_dir <path>` to
  point at a different seed or run)
- Runs inference on `data/<dataset>/test.csv`
- Converts fine-grained predictions to **coarse labels** (`SD/ST/HE/.../UN`) via `src/label_map.py`
- Builds a **coarse-level confusion matrix**
- Extracts misclassified examples and highlights **high-confidence mistakes**
- Computes lightweight **writing-style signals** (length, punctuation, capitalization)

Run for each population:

```bash
# IJH (default)
python -m src.misclassification_ijh_test

# Ultra (mirror population — produces the dual-class TR + BE attractor)
python -m src.misclassification_ijh_test \
  --dataset ultra --model_dir experiments/results/ultra/seed_42
```

Results are saved to:

```
experiments/results/misclf_<dataset>_test_predictions.csv
experiments/results/misclf_<dataset>_test_misclassified.csv
experiments/results/misclf_<dataset>_test_confusion_matrix.csv
experiments/plots/misclf_<dataset>_test_confusion_matrix.{png,pdf}
```

---

### 7️⃣ Schwartz Circumplex Distance Scoring

Tag every misclassification by its **modular distance on the Schwartz coarse
circumplex** (`SD → ST → HE → AC → PO → FA → SE → TR → CO → HU → BE → UN → SD`),
so that errors crossing a higher-order axis (self-enhancement ↔ self-transcendence;
openness ↔ conservation) can be distinguished from local adjacency confusions.

Run after step 6 has produced misclassified CSVs for the relevant populations:

```bash
python -m src.circumplex_error_analysis --datasets ijh ultra
```

Outputs:

```
experiments/results/misclf_<dataset>_test_misclassified_scored.csv  # + distance, bucket, axis_pair
experiments/results/misclf_<dataset>_test_attractor_summary.csv     # per-class attractor counts
experiments/results/circumplex_summary.csv                          # cross-population overview
```

Distance buckets: `1` adjacent (weakest evidence), `2-3` near, `4-5` cross-axis
(strong evidence), `6` diametric (strongest). The paper reports IJH at mean
distance 3.6 (92.6% cross-axis) and Ultra at 3.3 (85.2% cross-axis).

---

### 8️⃣ Population-Typical Distinctive Vocabulary

For each (population, value) cell, compute the most distinctive content tokens
relative to the **same value class in the other populations**. Distinctiveness is
measured by log-odds-ratio with an informative Dirichlet prior built from the
overall corpus background ([Monroe, Colaresi, Quinn 2008](https://doi.org/10.1093/pan/mpn018)).

This is the empirical mechanism behind the population-typical "lexical register"
finding: e.g., Ultra-BE and Ultra-TR share five distinctive content tokens
(*children, education, love, students, values*), which is what drives the BE→TR
misclassification pattern.

Run:

```bash
python -m src.lexical_exhibits --input data/merged.csv
```

Outputs:

```
experiments/results/lexical_distinctive_by_pop_value.csv  # full table, top-20 per cell
experiments/results/lexical_ultra_BE_top.csv              # Ultra communal-attractor
experiments/results/lexical_ultra_AC_top.csv
experiments/results/lexical_ijh_UN_top.csv               # IJH achievement-attractor source
experiments/results/lexical_ijh_AC_top.csv
```

---

### 9️⃣ Register Distance vs Transfer

Section 8 shows *which* words are distinctive. This step asks whether lexical
distance between two populations, measured on its own, predicts how well a
classifier transfers between them — otherwise the register account is circular.

Distances are computed from the **training text alone**, with no access to
labels, predictions, or F1 scores, and only then correlated against the
transfer matrix:

```bash
python -m src.register_distance
```

Three measures are reported, and they disagree. Over the six population pairs,
the share of the 100 most frequent content words two populations do *not* share
tracks transfer closely (Spearman ρ = −0.93), while Jensen–Shannon divergence
between unigram distributions (−0.26) and Jaccard distance over word types
(+0.03) do not — with fewer than 2,000 content tokens per population, both
whole-vocabulary measures are dominated by the rare tail. Two alternative
accounts are checked on the same pairs: shared nationality (−0.66, i.e. the
wrong direction for cultural proximity) and relative training-set size (+0.37).

Outputs:

```
experiments/results/register_distance.csv
experiments/results/register_distance_correlations.json
```

A fourth measure works in embedding space instead of over word counts, as a
check on whether register distance is just semantic distance:

```bash
python -m src.register_distance_embed
python -m src.register_distance_embed --model sentence-transformers/all-mpnet-base-v2
```

Each training response is encoded with a general-purpose sentence encoder, and
the populations are compared by centroid cosine distance and by energy distance
between the embedding samples. Both give Spearman ρ = −0.60, identical under the
two encoders above. The script refuses to run with Adam-Smith or one of its
checkpoints as the encoder, since measuring register distance in the
representation space of the model under study would reintroduce the circularity
the analysis exists to break.

The interesting part is where it misses. It ranks Asian–Indian closest, matching
their transfer, but puts IJH–Ultra only fourth in distance although that pair
transfers worst of the six. Topic is held fixed by the elicitation prompt, so
what separates IJH from Ultra is vocabulary rather than subject matter. The four
measures order themselves by how heavily they weight frequent words: −0.93,
−0.60, −0.26, +0.03.

Outputs:

```
experiments/results/register_distance_embeddings.csv
experiments/results/register_distance_embeddings.json
```

With n = 6 pairs and four measures this establishes an ordering, not a
significance claim.

---

### 🔟 Achievement-Vocabulary Ablation

Test directly whether a small set of achievement-coded tokens (`achieve`, `impact`,
`improve`, `advance`, with morphological variants; 22 tokens total) causally drives
the model's over-prediction of Achievement. The script masks those tokens with
`[MASK]` and re-runs predictions, comparing original vs. masked along three axes:
AC prediction frequency (and AC P/R/F1), macro-F1 across the 12 coarse classes,
and per-cell shifts in the row-normalized confusion matrix.

Five settings are reported in the paper (base + each per-population fine-tuned
checkpoint), all evaluated on the full merged corpus:

```bash
# 1. Base Adam-Smith.
python -m src.ablate_achievement_vocab \
  --model_dir models/adam-smith \
  --input_csv data/merged.csv \
  --output_label ablation_achievement

# 2-5. Per-population fine-tuned checkpoints (Asian / Indian / IJH / Ultra).
for ds in asian indian ijh ultra; do
  python -m src.ablate_achievement_vocab \
    --checkpoint_dir experiments/results/${ds}/seed_42 \
    --input_csv data/merged.csv \
    --output_label ablation_achievement_${ds}_finetuned
done
```

Each run writes:

```
experiments/results/<output_label>/predictions.csv
experiments/results/<output_label>/summary.json
experiments/results/<output_label>/summary.txt
experiments/plots/<output_label>/global_cm_diff.{png,pdf}
experiments/plots/<output_label>/<dataset>_cm_diff.{png,pdf}
```

Once all five runs are done, collate their `summary.json` files into the
cross-setting table:

```bash
python -m src.aggregate_ablation
```

This writes `experiments/results/ablation_summary_per_setting.csv` (one row per
setting: AC-prediction share and F1 deltas before/after masking).

The headline result is the negative one: across all five settings, masking this
vocabulary shifts AC-prediction share by only `|Δ| ≤ 2.1`pp (base −1.4, Asian
−0.9, Indian −0.6, IJH −0.1, Ultra −2.1), indicating that AC over-prediction is
not primarily driven by this lexical set.

---

### 1️⃣1️⃣ Few-Shot LLM Comparison

The same transfer matrix, with in-context prompting in place of fine-tuning: a
row per population the exemplars are drawn from, the same four test splits as
columns, plus a zero-shot row standing in for the base row.

```bash
python -m src.score_fewshot                  # every model found
python -m src.score_fewshot --model qwen3:14b
```

Scoring goes through the same union-macro convention as `src/eval.py`, so the
few-shot and fine-tuned cells are directly comparable with no rescaling, and
micro F1 is reported alongside it. The fine-tuned reference matrices are rebuilt
from the committed per-seed logs under `experiments/results/`, and the script
aborts if the rebuilt macro matrix disagrees with the published one, so no
checkpoints and no GPU are needed.

The generation side lives in `experiments/fewshot/`:

```
experiments/fewshot/create_prompts.py     # builds the prompt templates from the train splits
experiments/fewshot/prompts/*.txt         # the 20 templates the models actually saw
experiments/fewshot/run_fewshot.py        # runs one template over the combined test set
```

```bash
python experiments/fewshot/create_prompts.py --training-data data --out prompts
for p in experiments/fewshot/prompts/*.txt; do
  python experiments/fewshot/run_fewshot.py --model qwen3:14b --prompt "$p"
done
```

Each template gives the model the ten reachable Schwartz values with their
standard definitions, then up to four exemplars per value drawn from one
population's **train** split, in a value order shuffled per seed. Zero-shot
templates carry the definitions and no exemplars. Models are served locally by
ollama and the response format pins the output to a JSON object holding one of
the ten codes, so parse failure is unreachable by construction rather than
avoided by retrying. Decoding uses each model's own defaults and is not seeded;
the seed selects exemplars, not sampling.

No exemplar occurs in any test split, and regenerating from `data/*/train.csv`
reproduces every delivered template's exemplar set exactly.

The per-item prediction dumps that `score_fewshot.py` reads are co-author-owned
interim data and are not in the repository yet. They are released with the paper.

---

## Label Space

Evaluation is performed in the **12 coarse SVS categories**:

| Code | Value |
|------|-------|
| SD | Self-Direction |
| ST | Stimulation |
| HE | Hedonism |
| AC | Achievement |
| PO | Power |
| FA | Face |
| SE | Security |
| TR | Tradition |
| CO | Conformity |
| HU | Humility |
| BE | Benevolence |
| UN | Universalism |

The mapping between fine-grained (20-class) and coarse (12-class) labels is defined in:

```
src/label_map.py
```

Note that only **ten** of the twelve categories are ever used in the annotated
corpus: Face (FA) and Humility (HU) have no instances in any of the four
populations. The model can still predict them, and the base model does so
occasionally, which is why the macro-F1 convention below matters.

**Macro-F1 convention.** `src/eval.py` calls
`f1_score(gold, pred, average="macro")` with no `labels=` argument, so sklearn
averages over the union of gold and predicted classes. The denominator therefore
depends on what the model predicts: a model that predicts a class with no gold
support in that test set picks up an extra zero-F1 category. Every cell of the
transfer matrix goes through this one code path, so the matrix is internally
consistent. In practice the choice only affects the base row — fine-tuned models
do not predict outside their target's gold support, so every fine-tuned cell is
identical under either convention. To measure the difference yourself:

```bash
python -m src.dump_predictions              # writes per-item predictions for every cell
python -m src.dump_predictions --recompute  # rescores them under all three conventions
```

---

## Reproducing the Reported Numbers

Model checkpoints and per-item prediction dumps are far too large to version
(the checkpoints alone run to hundreds of gigabytes), but the **metric logs that
every number in the paper is computed from are tracked in this repository** —
about 200 KB of CSV under `experiments/results/`. So the reported results can be
checked without re-running any training:

```bash
# Rebuild the transfer matrix and heatmap from the committed per-seed logs.
python -m src.aggregate_seeds --datasets asian,indian,ijh,ultra,combined
python -m src.cross_domain_heatmaps --datasets asian,indian,ijh,ultra
```

Re-running the fine-tuning itself takes roughly 18 hours across the 17 runs on a
single Apple M4 (MPS backend): Asian ~25 min, Indian ~74 min, IJH ~26 min, Ultra
~46 min, and Combined ~163 min per seed. The analyses in sections 4️⃣–🔟 run in
minutes on a laptop.

---

## License

Code in this repository is released under the **MIT license** ([LICENSE](LICENSE)).

The annotated corpus is distributed separately, on Zenodo, under
**CC-BY-4.0**. The Adam-Smith checkpoint is third-party and carries its own
**OpenRAIL++** license.

---

## 👤 Author

**Yehuda Frist**  
M.Sc. Machine Learning & Data Science  
