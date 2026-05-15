# ValueDetection

## Cross-Domain Prediction of Schwartz Human Values

This repository implements a full experimental pipeline for evaluating and fine-tuning the **Adam-Smith value prediction model** across multiple datasets:

- Asians  
- Indians  
- Joint Org. Employees
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
│   ├── joint/
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
│   ├── misclassification_joint_test.py     # per-population error analysis (Joint, Ultra, ...)
│   ├── circumplex_error_analysis.py        # Schwartz circumplex distance scoring
│   ├── lexical_exhibits.py                 # log-odds distinctive vocabulary per cell
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
│   │   ├── misclf_<dataset>_test_*.csv        # per-population error tables + confusion matrices
│   │   ├── misclf_<dataset>_test_misclassified_scored.csv  # + Schwartz circumplex distance
│   │   ├── misclf_<dataset>_test_attractor_summary.csv     # per-class attractor counts
│   │   ├── circumplex_summary.csv             # cross-population circumplex overview
│   │   ├── lexical_distinctive_by_pop_value.csv  # full distinctive-vocabulary table
│   │   ├── lexical_<pop>_<value>_top.csv          # per-cell highlights (Ultra-BE, Joint-UN, ...)
│   │   ├── ablation_achievement/                    # base model, full corpus
│   │   ├── ablation_achievement_<dataset>_finetuned/   # per-population epoch-10, full corpus
│   │   ├── ablation_achievement_joint_finetuned/    # Joint epoch-10, Joint test split
│   │   ├── ablation_achievement_joint_finetuned_train/  # Joint epoch-10, Joint train split
│   │   └── ablation_summary_per_setting.csv         # cross-setting AC-share / F1 deltas
│   ├── train.txt     # Training CLI output
│   └── plots/        # Evaluation plots and charts
│
├── .gitignore
├── requirements.txt
└── README.md
```

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

- Joint  
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

To produce mean ± std for the headline numbers, run additional seeds for the
configurations that matter most (Combined and Joint, per the paper) and
aggregate:

```bash
# 1. Train Joint and Combined with three seeds (42 already done by default).
python -m src.train_multi_seed --datasets joint,combined --seeds 42,43,44

# 2. Aggregate last-epoch metrics into mean ± std per configuration.
python -m src.aggregate_seeds --datasets joint,combined
```

This writes:

```
experiments/results/joint_seed_summary.csv
experiments/results/combined_seed_summary.csv
```

and prints a per-metric mean ± std table to stdout, ready to drop into the
paper's Results section.

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


It uses the **last epoch** of each run and constructs a Macro-F1 matrix where:

- Rows = training dataset  
- Columns = evaluation dataset  
- Diagonal cells = in-domain `in_test_f1`  
- Off-diagonal cells = out-of-domain `ood_<dataset>_f1`  

Run:

```bash
python -m src.cross_domain_heatmaps
```

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
# Joint (default)
python -m src.misclassification_joint_test

# Ultra (mirror population — produces the dual-class TR + BE attractor)
python -m src.misclassification_joint_test \
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
python -m src.circumplex_error_analysis --datasets joint ultra
```

Outputs:

```
experiments/results/misclf_<dataset>_test_misclassified_scored.csv  # + distance, bucket, axis_pair
experiments/results/misclf_<dataset>_test_attractor_summary.csv     # per-class attractor counts
experiments/results/circumplex_summary.csv                          # cross-population overview
```

Distance buckets: `1` adjacent (weakest evidence), `2-3` near, `4-5` cross-axis
(strong evidence), `6` diametric (strongest). The paper reports Joint at mean
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
experiments/results/lexical_joint_UN_top.csv              # Joint achievement-attractor source
experiments/results/lexical_joint_AC_top.csv
```

---

### 9️⃣ Achievement-Vocabulary Ablation

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

# 2-5. Per-population fine-tuned checkpoints (Asian / Indian / Joint / Ultra).
for ds in asian indian joint ultra; do
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

The headline result is the negative one: across all five settings, masking this
vocabulary shifts AC-prediction share by only `|Δ| ≤ 2.1`pp (base −1.4, Asian
−0.9, Indian −0.6, Joint −0.1, Ultra −2.1), indicating that AC over-prediction is
not primarily driven by this lexical set. The cross-setting deltas are also
summarized in `experiments/results/ablation_summary_per_setting.csv`.

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

---

## 👤 Author

**Yehuda Frist**  
M.Sc. Machine Learning & Data Science  
