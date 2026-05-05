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
│   └── ultra/
│       ├── train.csv        
│       └── test.csv         
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
│   ├── misclassification_joint_test.py
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
│   │   └── <dataset>_seed_summary.csv   # mean ± std across seeds
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

### 6️⃣ Misclassification Analysis (Joint Test Set)

Run a focused **in-domain misclassification analysis** on the **Joint test set** to better understand *what confuses the model* beyond aggregate F1 scores.

This script:

- Loads a fine-tuned Joint checkpoint (by default, the latest `epoch_*` under
  `experiments/results/joint/seed_42/`; pass `--model_dir <path>` to point at a
  different seed or run)
- Runs inference on `data/joint/test.csv`
- Converts fine-grained predictions into **coarse labels** (`SD/ST/HE/.../UN`) using `src/label_map.py` (`COARSE_TO_FINE`)
- Builds a **coarse-level confusion matrix**
- Extracts misclassified examples and highlights **high-confidence mistakes**
- Computes lightweight **writing-style signals** (length, punctuation, capitalization) to compare correct vs. incorrect predictions

Run:

```bash
python -m src.misclassification_joint_test
# or, to inspect a different seed:
python -m src.misclassification_joint_test --model_dir experiments/results/joint/seed_43
```

Results are saved to:

```
experiments/results/misclf_joint_test_predictions.csv
experiments/results/misclf_joint_test_misclassified.csv
experiments/results/misclf_joint_test_confusion_matrix.csv
experiments/plots/misclf_joint_test_confusion_matrix.{png,pdf}
```

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
