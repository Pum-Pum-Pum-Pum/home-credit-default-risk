# Home Credit Default Risk - PyTorch Tabular Project

Production-style tabular deep learning project using the Home Credit Default Risk dataset.

This repository was built as a serious hands-on learning project for:
- GPU-aware PyTorch training
- tabular neural networks with categorical embeddings
- clean `Dataset` / `DataLoader` design
- training / validation loop engineering
- checkpointing and inference consistency
- threshold tuning and class imbalance handling
- fair comparison against a strong tree-based baseline

---

## Final conclusion

This project reached two honest conclusions:

### 1. Learning conclusion
The PyTorch pipeline was a success as a **deep learning engineering project**.

It covered:
- tabular preprocessing metadata
- custom `Dataset` and `DataLoader`
- embedding-based tabular MLP
- GPU training and validation loops
- metrics, threshold tuning, imbalance handling
- checkpointing, inference pipeline, and experiment logging

### 2. Production conclusion
For this specific structured credit-risk dataset, the **XGBoost baseline outperformed the PyTorch model** and is the better current production recommendation.

That is an important real-world lesson:

> the best learning model and the best production model are not always the same.

---

## Final comparison snapshot

### Best PyTorch run
- Best epoch: **19**
- Monitor: **validation PR-AUC**
- ROC-AUC: **0.6492**
- PR-AUC: **0.1432**
- Precision: **0.0996**
- Recall: **0.8169**
- F1: **0.1776**

### XGBoost baseline
- ROC-AUC: **0.7545**
- PR-AUC: **0.2449**

### Production recommendation
**Use XGBoost for this dataset** based on the current evidence.

For more detail, see:
- `FINAL_PROJECT_SUMMARY.md`
- `Interview_questions_with_answers.md`

---

## Project structure

```text
.
├── FINAL_PROJECT_SUMMARY.md
├── Interview_questions_with_answers.md
├── artifacts/
│   ├── checkpoints/
│   ├── logs/
│   ├── metadata/
│   └── metrics/
├── configs/
│   └── base_config.py
├── notebooks/
├── prerequisites.md
├── requirements.txt
├── scripts/
│   ├── _bootstrap.py
│   ├── compare_models.py
│   ├── run_eda.py
│   ├── run_inference_demo.py
│   ├── train_pytorch.py
│   └── train_xgb_baseline.py
└── src/
    ├── data/
    ├── inference/
    ├── models/
    ├── training/
    └── utils/
```

---

## Key components

### `src/data`
- EDA utilities
- split logic
- preprocessing metadata
- tabular `Dataset`
- `DataLoader` utilities

### `src/models`
- tabular MLP with categorical embeddings + numeric features

### `src/training`
- GPU-aware training loop
- validation loop
- metrics and threshold sweep
- class imbalance handling
- early stopping
- XGBoost baseline utilities

### `src/inference`
- artifact loading
- consistent inference pipeline

### `src/utils`
- device helpers
- checkpointing / artifact saving

---

## Main scripts

### 1. PyTorch training
```cmd
.venv311\Scripts\python.exe scripts\train_pytorch.py
```

### 2. XGBoost baseline training
```cmd
.venv311\Scripts\python.exe scripts\train_xgb_baseline.py
```

### 3. Side-by-side threshold comparison
```cmd
.venv311\Scripts\python.exe scripts\compare_models.py
```

### 4. Inference demo
```cmd
.venv311\Scripts\python.exe scripts\run_inference_demo.py
```

### 5. Exploratory end-to-end preview script
```cmd
.venv311\Scripts\python.exe scripts\run_eda.py
```

---

## Important artifacts

### PyTorch
- `artifacts/checkpoints/train_pytorch_best.pt`
- `artifacts/metrics/train_pytorch_history.json`
- `artifacts/metrics/train_pytorch_metrics.json`
- `artifacts/logs/train_pytorch_summary.txt`

### XGBoost baseline
- `artifacts/metrics/train_xgb_baseline_metrics.json`
- `artifacts/logs/train_xgb_baseline_summary.txt`

### Comparison artifacts
- `artifacts/metrics/model_threshold_comparison_step19.json`
- `artifacts/logs/model_threshold_comparison_step19.txt`

### Shared metadata / inference artifacts
- `artifacts/metadata/preprocessing_step11_demo.json`
- `artifacts/metadata/inference_config_step11_demo.json`

---

## Environment setup

Use the project virtual environment:

```cmd
.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

For full setup notes, including CUDA-enabled PyTorch installation, see:
- `prerequisites.md`

---

## What this project teaches

This repository is especially useful if you want to learn:
- PyTorch for tabular data
- GPU correctness and device placement
- embeddings for categorical features
- production-style ML project structure
- artifact management and inference consistency
- why tree models often still beat neural nets on structured tabular data

---

## Final interview-ready takeaway

> I built an end-to-end PyTorch tabular modeling pipeline with embeddings, GPU-aware training, validation, metrics, threshold tuning, checkpointing, inference artifact management, and experiment logging. I then compared it fairly against an XGBoost baseline on the same validation split. Although the PyTorch model improved after disciplined tuning, XGBoost remained clearly stronger on ROC-AUC and PR-AUC, so my production recommendation for this dataset would be the tree-based model. However, the PyTorch project was still highly valuable because it developed deep-learning engineering skills for structured data and production-style workflow design.
