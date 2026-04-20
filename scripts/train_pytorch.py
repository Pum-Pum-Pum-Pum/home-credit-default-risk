"""Dedicated PyTorch training script for the Home Credit tabular project."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import ProjectConfig
from src.data.dataloaders import DataLoaderConfig, create_dataloaders
from src.data.dataset import HomeCreditDataset
from src.data.eda import load_main_table
from src.data.preprocessing import build_tabular_metadata
from src.data.splits import make_train_valid_split
from src.models.tabular_mlp import TabularMLP, TabularMLPConfig
from src.training.metrics import compute_binary_classification_metrics, summarize_metrics
from src.training.trainer import (
    EarlyStoppingConfig,
    TrainingConfig,
    build_optimizer,
    build_weighted_loss_fn,
    compute_pos_weight,
    get_device,
    run_training_loop_with_early_stopping,
    run_validation_epoch,
)
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

    train_dataset = HomeCreditDataset(split_data.train_df, metadata)
    valid_dataset = HomeCreditDataset(split_data.valid_df, metadata)
    loader_config = DataLoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    train_loader, valid_loader = create_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        config=loader_config,
    )

    cat_cardinalities = [len(metadata.category_maps[col]) + 1 for col in metadata.categorical_cols]
    model_config = TabularMLPConfig(
        embedding_dropout=config.embedding_dropout,
        mlp_hidden_dims=config.mlp_hidden_dims,
        mlp_dropout=config.mlp_dropout,
        use_batch_norm=config.use_batch_norm,
    )
    model = TabularMLP(
        cat_cardinalities=cat_cardinalities,
        num_numeric_features=len(metadata.numerical_cols),
        config=model_config,
    )

    training_config = TrainingConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    device = get_device()
    model = model.to(device)
    optimizer = build_optimizer(model, training_config)

    train_targets_np = split_data.train_df[config.target_col].to_numpy(dtype="float32").reshape(-1, 1)
    pos_weight = compute_pos_weight(train_targets_np)
    weighted_loss_fn = build_weighted_loss_fn(pos_weight.to(device))

    early_stopping_config = EarlyStoppingConfig(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        checkpoint_path="artifacts/checkpoints/train_pytorch_best.pt",
        monitor="valid_pr_auc",
    )
    result = run_training_loop_with_early_stopping(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=weighted_loss_fn,
        device=device,
        num_epochs=config.num_epochs_demo,
        early_stopping=early_stopping_config,
    )

    valid_output = run_validation_epoch(
        model=model,
        valid_loader=valid_loader,
        loss_fn=weighted_loss_fn,
        device=device,
    )
    metrics = compute_binary_classification_metrics(
        targets=valid_output.targets,
        probs=valid_output.probs,
        threshold=0.4,
    )

    history_payload = [
        {
            "epoch": entry.epoch,
            "train_loss": entry.train_loss,
            "train_mean_pred_prob": entry.train_mean_pred_prob,
            "valid_loss": entry.valid_loss,
            "valid_roc_auc": entry.valid_roc_auc,
            "valid_pr_auc": entry.valid_pr_auc,
            "epoch_seconds": entry.epoch_seconds,
        }
        for entry in result.history
    ]
    history_path = save_json_artifact(history_payload, "artifacts/metrics/train_pytorch_history.json")
    metrics_path = save_json_artifact(summarize_metrics(metrics), "artifacts/metrics/train_pytorch_metrics.json")
    summary_path = save_text_artifact(
        (
            f"Best epoch: {result.best_epoch}\n"
            f"Monitor: {result.monitor}\n"
            f"Best metric value: {result.best_metric_value:.6f}\n"
            f"Checkpoint: {result.checkpoint_path}\n"
            f"Device: {device}\n"
            f"Total training seconds: {result.total_training_seconds:.4f}\n"
            f"Average epoch seconds: {result.average_epoch_seconds:.4f}\n"
        ),
        "artifacts/logs/train_pytorch_summary.txt",
    )

    print({
        "best_epoch": result.best_epoch,
        "best_metric_value": result.best_metric_value,
        "monitor": result.monitor,
        "history_path": history_path,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "checkpoint_path": result.checkpoint_path,
        "total_training_seconds": result.total_training_seconds,
        "average_epoch_seconds": result.average_epoch_seconds,
    })


if __name__ == "__main__":
    main()
