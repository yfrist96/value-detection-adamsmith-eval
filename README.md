We first run data_analysis.py to get results in data/data_analysis/ using python src/data_analysis.py.

We the split the datasets into train/test for each dataset using src/split_datasets.py (includes combined).

We the run python -m src.train.py to train each config for 10 epochs and test on in and out of domain splits.

We then run run_merged_base_adamsmith_eval.py with this command: 
python -m src.run_merged_base_adamsmith_eval \
  --model_dir models/adam-smith \
  --input_csv data/merged.csv \
  --top_k 3 \
  --threshold 0.25





ValueDetection/
│
├── data/
|   ├──mereged.csv
|   |
|   ├── data_analysis/
│   │   ├── plots/
│   │   ├── summary.json
│   │   └── *.csv
|   |
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
│   ├── adam-smith/
|   |   ├── ...
|
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── eval.py
│   ├── plotting.py
|   ├── split_datasets.py
|   ├── data_analysis.py
|   ├── label_map.py
│   └── utils.py
│
├── experiments/
│   ├── results/             # F1 scores, logs per epoch
│   └── plots/               # line charts
│
├── notebooks/
│   └── exploration.ipynb
│
└── README.md
