"""Dedicated XGBoost baseline training script."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import ProjectConfig
from src.data.eda import load_main_table
from src.data.preprocessing import build_tabular_metadata
from src.data.splits import make_train_valid_split
from src.training.baselines import run_xgb_baseline
from src.utils.checkpointing import save_json_artifact, save_text_artifact


def main() -> None:
    config = ProjectConfig()
    df = load_main_table(config.data_path)
    split_data = make_train_valid_split(
        df=df,
        target_col=config.target_col,
        valid_size=config.valid_size,
        random_state=config.random_state,
    )
    metadata = build_tabular_metadata(
        train_df=split_data.train_df,
        target_col=config.target_col,
        id_cols=config.id_cols,
    )

    result = run_xgb_baseline(
        train_df=split_data.train_df,
        valid_df=split_data.valid_df,
        target_col=config.target_col,
        categorical_cols=metadata.categorical_cols,
        numerical_cols=metadata.numerical_cols,
        random_state=config.random_state,
    )

    metrics_payload = {
        "xgb_roc_auc": result.roc_auc,
        "xgb_pr_auc": result.pr_auc,
        "xgb_positive_rate_pred": result.positive_rate_pred,
    }
    metrics_path = save_json_artifact(metrics_payload, "artifacts/metrics/train_xgb_baseline_metrics.json")
    summary_path = save_text_artifact(
        (
            f"XGBoost ROC-AUC: {result.roc_auc:.6f}\n"
            f"XGBoost PR-AUC: {result.pr_auc:.6f}\n"
            f"Predicted positive rate: {result.positive_rate_pred:.6f}\n"
        ),
        "artifacts/logs/train_xgb_baseline_summary.txt",
    )

    print({
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        **metrics_payload,
    })


if __name__ == "__main__":
    main()
