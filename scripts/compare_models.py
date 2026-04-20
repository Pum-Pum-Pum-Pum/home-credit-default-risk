"""Threshold optimization and side-by-side business comparison for PyTorch vs XGBoost."""

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
from src.inference.pipeline import load_inference_artifacts, score_dataframe
from src.models.tabular_mlp import TabularMLPConfig
from src.training.baselines import run_xgb_baseline
from src.training.metrics import threshold_sweep
from src.training.trainer import get_device
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

    model_config = TabularMLPConfig(
        embedding_dropout=config.embedding_dropout,
        mlp_hidden_dims=config.mlp_hidden_dims,
        mlp_dropout=config.mlp_dropout,
        use_batch_norm=config.use_batch_norm,
    )
    device = get_device()

    pytorch_artifacts = load_inference_artifacts(
        checkpoint_path="artifacts/checkpoints/train_pytorch_best.pt",
        metadata_path="artifacts/metadata/preprocessing_step11_demo.json",
        inference_config_path="artifacts/metadata/inference_config_step11_demo.json",
        model_config=model_config,
        device=device,
    )
    valid_inference_input = split_data.valid_df.drop(columns=[config.target_col]).copy()
    pytorch_scores = score_dataframe(valid_inference_input, pytorch_artifacts)
    pytorch_probs = pytorch_scores["score_probability"].to_numpy()
    valid_targets = split_data.valid_df[config.target_col].to_numpy()

    xgb_result = run_xgb_baseline(
        train_df=split_data.train_df,
        valid_df=split_data.valid_df,
        target_col=config.target_col,
        categorical_cols=metadata.categorical_cols,
        numerical_cols=metadata.numerical_cols,
        random_state=config.random_state,
    )

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    pytorch_threshold_results = threshold_sweep(valid_targets, pytorch_probs, thresholds)
    xgb_threshold_results = threshold_sweep(xgb_result.valid_targets, xgb_result.valid_probs, thresholds)

    artifact_payload = {
        "thresholds": thresholds,
        "pytorch": pytorch_threshold_results,
        "xgboost": xgb_threshold_results,
    }
    comparison_path = save_json_artifact(
        artifact_payload,
        "artifacts/metrics/model_threshold_comparison_step19.json",
    )
    summary_path = save_text_artifact(
        (
            "Step 19 model comparison completed.\n"
            f"Saved JSON comparison to: {comparison_path}\n"
            "Review PR-AUC, recall, precision, predicted positive rate, and confusion behavior by threshold.\n"
        ),
        "artifacts/logs/model_threshold_comparison_step19.txt",
    )

    print({
        "comparison_path": comparison_path,
        "summary_path": summary_path,
        "pytorch_threshold_results": pytorch_threshold_results,
        "xgboost_threshold_results": xgb_threshold_results,
    })


if __name__ == "__main__":
    main()