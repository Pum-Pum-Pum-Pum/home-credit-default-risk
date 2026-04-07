# Home Credit Default Risk - PyTorch Tabular Project

Production-style learning project for tabular deep learning with PyTorch.

## Initial scope
- Main table only: `Data/application_train.csv`
- Binary classification target: `TARGET`
- First goal: GPU-ready EDA and a clean PyTorch project structure

## Project structure

```text
.
├── Data/
├── artifacts/
│   ├── checkpoints/
│   ├── logs/
│   └── metrics/
├── configs/
├── notebooks/
├── prerequisites.md
├── requirements.txt
├── scripts/
└── src/
    ├── data/
    ├── inference/
    ├── models/
    ├── training/
    └── utils/
```

## Why this structure
- `src/data`: loading, schema checks, preprocessing, dataset code
- `src/models`: PyTorch model definitions
- `src/training`: training loop, validation loop, metrics, early stopping
- `src/inference`: prediction pipeline and serving-time consistency
- `src/utils`: shared helpers such as config, logging, seeds, device helpers
- `configs`: experiment configuration files
- `artifacts`: outputs such as checkpoints and metrics
- `scripts`: runnable entry points
- `prerequisites.md`: environment and CUDA installation reference
- `requirements.txt`: project dependency manifest

## Dependency management
- Use `.venv311` as the project environment
- Install CUDA-enabled PyTorch using `prerequisites.md`
- Keep `requirements.txt` updated as the project grows
- Do not rely on global Python packages for training or inference

## Current Step
Step 1 - PyTorch tabular project setup + GPU-ready EDA.
