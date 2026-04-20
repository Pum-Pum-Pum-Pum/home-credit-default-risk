# Final Project Summary - Home Credit Default Risk

## Project goal

This project was used for two parallel goals:

1. **Learning goal**: build a production-style end-to-end PyTorch tabular pipeline with GPU-aware training, validation, metrics, checkpointing, threshold tuning, and inference consistency.
2. **Model selection goal**: compare the PyTorch tabular model fairly against a strong tree-based baseline for a structured credit-risk problem.

---

## What was built on the PyTorch side

The PyTorch pipeline now includes:

- GPU-ready environment and device management
- tabular EDA and schema analysis
- leakage-aware train/validation split
- preprocessing metadata for categorical and numerical features
- `Dataset` and `DataLoader` design
- tabular MLP with categorical embeddings + numeric features
- GPU-aware training loop
- validation loop
- binary classification metrics
- threshold sweep logic
- class imbalance handling with `pos_weight`
- checkpoint saving
- inference pipeline with saved artifacts
- multi-epoch training
- early stopping and best-checkpoint tracking
- experiment history and summary artifacts

This means the project is a **real PyTorch tabular engineering project**, not just a notebook demo.

---

## Final model comparison summary

### Best PyTorch result from the focused improvement round

- Best epoch: **19**
- Monitor: **validation PR-AUC**
- Best monitored value: **0.144151**
- Final saved metrics:
  - ROC-AUC: **0.6492**
  - PR-AUC: **0.1432**
  - Precision: **0.0996**
  - Recall: **0.8169**
  - F1: **0.1776**
  - Positive prediction rate: **0.6620**

### XGBoost baseline

- ROC-AUC: **0.7545**
- PR-AUC: **0.2449**
- Positive prediction rate at default threshold: **0.0010**

### Side-by-side conclusion

Even after:

- a cleaner production-style project structure
- longer PyTorch training
- PR-AUC-based checkpoint monitoring
- threshold tuning
- imbalance-aware training

the **XGBoost baseline remained clearly stronger** than the PyTorch tabular MLP on this problem.

---

## Production recommendation

### Recommended production candidate for this dataset
**XGBoost**

### Why XGBoost won

The XGBoost baseline outperformed the PyTorch model on the most important validation ranking metrics:

- substantially higher ROC-AUC
- substantially higher PR-AUC
- stronger overall decision quality for this structured tabular problem

This result is consistent with a broader industry pattern:

> On medium-sized structured tabular data, tree-based boosting methods often outperform neural networks because they are more sample-efficient, handle threshold interactions naturally, cope well with missingness, and usually require less tuning and less engineering complexity.

### Production-minded conclusion

If this project were being deployed strictly to maximize current business performance on this dataset, the mature choice would be:

> **Deploy XGBoost, not the current PyTorch model.**

---

## Learning recommendation

### Recommended learning conclusion
**Keep the PyTorch project as a deep learning engineering win.**

Why?

Because even though PyTorch did not beat XGBoost here, it successfully taught:

- device management on GPU
- tensor dtype/shape reasoning
- embedding-based tabular modeling
- clean training and validation loops
- threshold tuning
- checkpointing and inference consistency
- production-style project organization

That means the PyTorch side of the project was still highly valuable for interview preparation and practical engineering growth.

### Honest takeaway

> The PyTorch model was not the best predictive model for this dataset, but building it was still the right learning exercise because it developed deep-learning engineering skills that tree baselines do not teach.

---

## Why the PyTorch model still mattered

This project demonstrated an important professional lesson:

- **best learning model** and **best production model** are not always the same thing

In this case:

- **best production model** -> XGBoost
- **best learning pipeline for deep learning engineering** -> PyTorch tabular workflow

That is a strong and mature conclusion to present in interviews.

---

## Artifact summary

Important generated artifacts include:

- PyTorch best checkpoint:
  - `artifacts/checkpoints/train_pytorch_best.pt`
- PyTorch metrics:
  - `artifacts/metrics/train_pytorch_metrics.json`
- PyTorch history:
  - `artifacts/metrics/train_pytorch_history.json`
- PyTorch summary:
  - `artifacts/logs/train_pytorch_summary.txt`
- XGBoost baseline metrics:
  - `artifacts/metrics/train_xgb_baseline_metrics.json`
- XGBoost summary:
  - `artifacts/logs/train_xgb_baseline_summary.txt`
- Threshold comparison artifact:
  - `artifacts/metrics/model_threshold_comparison_step19.json`
- Threshold comparison summary:
  - `artifacts/logs/model_threshold_comparison_step19.txt`

---

## Final interview-quality conclusion

> I built an end-to-end PyTorch tabular modeling pipeline with embeddings, GPU-aware training, validation, metrics, threshold tuning, checkpointing, and inference artifact management. I then compared it fairly against an XGBoost baseline on the same validation split. Although the PyTorch model improved after disciplined tuning, XGBoost remained clearly stronger on ROC-AUC and PR-AUC, so my production recommendation for this dataset would be the tree-based model. However, the PyTorch project was still highly valuable because it built the deep-learning engineering skills needed for structured-data interview readiness and production-style model workflow design.
