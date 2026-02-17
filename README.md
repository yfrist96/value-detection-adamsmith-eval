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
│   ├── eval.py
│   ├── plotting.py
│   ├── split_datasets.py
│   ├── data_analysis.py
│   ├── run_merged_base_adamsmith_eval.py
│   ├── label_map.py
│   └── utils.py
│
├── experiments/
│   ├── results/      # Metrics, logs, JSON summaries
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

- 10 epochs per configuration  
- Evaluation after each epoch  
- In-domain testing  
- Out-of-domain testing across other datasets  

Results are saved to:

```
experiments/results/<dataset>
experiments/plots/<dataset>_f1_plot.png
```

To produce plots that include epoch 0 (pre-fine-tunning), execute:

```bash
python -c "from src.plotting import plot_f1_curve; plot_f1_curve('<ds_name>')"
```

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
experiments/results/merged_base_adamsmith_eval_rows.csv
experiments/results/merged_base_adamsmith_predictions.csv
experiments/results/merged_base_adamsmith_metrics.json
experiments/results/merged_base_adamsmith_metrics.txt
experiments/plots/merged_...png
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
experiments/plots/cross_domain_macro_f1_heatmap.png
```

---

### 6️⃣ Misclassification Analysis (Joint Test Set)

Run a focused **in-domain misclassification analysis** on the **Joint test set** to better understand *what confuses the model* beyond aggregate F1 scores.

This script:

- Loads a fine-tuned Joint checkpoint (by default, the **latest** `epoch_*` under `experiments/results/joint/`)
- Runs inference on `data/joint/test.csv`
- Converts fine-grained predictions into **coarse labels** (`SD/ST/HE/.../UN`) using `src/label_map.py` (`COARSE_TO_FINE`)
- Builds a **coarse-level confusion matrix**
- Extracts misclassified examples and highlights **high-confidence mistakes**
- Computes lightweight **writing-style signals** (length, punctuation, capitalization) to compare correct vs. incorrect predictions

Run:

```bash
python -m src.misclassification_joint_test
```

Results are saved to:

```
experiments/results/misclf_joint_test_predictions.csv
experiments/results/misclf_joint_test_misclassified.csv
experiments/results/misclf_joint_test_confusion_matrix.csv
experiments/plots/misclf_joint_test_confusion_matrix.png
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
