"""Large exploratory script retained for learning history and step-by-step previews.

For cleaner production-style execution prefer:
- scripts/train_pytorch.py
- scripts/train_xgb_baseline.py
- scripts/run_inference_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import ProjectConfig
from src.data.dataloaders import DataLoaderConfig, create_dataloaders, inspect_batch
from src.data.dataset import HomeCreditDataset, inspect_dataset_sample
from src.data.eda import run_step1_eda
from src.data.preprocessing import build_tabular_metadata, summarize_tabular_metadata
from src.data.splits import make_train_valid_split
from src.inference.pipeline import load_inference_artifacts, score_dataframe
from src.models.tabular_mlp import TabularMLP, TabularMLPConfig, inspect_model_forward_pass
from src.training.baselines import run_xgb_baseline
from src.training.metrics import (
    compute_binary_classification_metrics,
    summarize_metrics,
    threshold_sweep,
)
from src.training.trainer import (
    EarlyStoppingConfig,
    EpochHistoryEntry,
    TrainingConfig,
    build_weighted_loss_fn,
    build_loss_fn,
    build_optimizer,
    compute_pos_weight,
    get_device,
    inspect_training_step_devices,
    run_training_loop,
    run_training_loop_with_early_stopping,
    run_train_epoch_preview,
    run_validation_epoch,
    train_step,
)
from src.utils.checkpointing import save_json_artifact, save_model_checkpoint, save_text_artifact
from src.utils.device import get_torch_device_summary


def main() -> None:
    config = ProjectConfig()
    run_step1_eda(
        data_path=config.data_path,
        target_col=config.target_col,
        id_cols=config.id_cols,
    )

    from src.data.eda import load_main_table

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

    print("\nStep 2 preview - split summary:")
    print(f"train shape: {split_data.train_df.shape}")
    print(f"valid shape: {split_data.valid_df.shape}")

    print("\nStep 2 preview - tabular metadata:")
    print(summarize_tabular_metadata(metadata))

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

    print("\nStep 3 preview - dataset lengths:")
    print(f"train dataset length: {len(train_dataset)}")
    print(f"valid dataset length: {len(valid_dataset)}")

    print("\nStep 3 preview - first sample tensor summary:")
    print(inspect_dataset_sample(train_dataset, idx=0))

    first_train_batch = next(iter(train_loader))
    first_valid_batch = next(iter(valid_loader))

    print("\nStep 4 preview - DataLoader batch summary:")
    print("train batch:", inspect_batch(first_train_batch))
    print("valid batch:", inspect_batch(first_valid_batch))

    cat_cardinalities = [
        len(metadata.category_maps[col]) + 1
        for col in metadata.categorical_cols
    ]
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

    print("\nStep 5 preview - model forward-pass summary:")
    print(inspect_model_forward_pass(model, first_train_batch))

    training_config = TrainingConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    device = get_device()
    model = model.to(device)
    optimizer = build_optimizer(model, training_config)
    loss_fn = build_loss_fn()

    train_targets_np = split_data.train_df[config.target_col].to_numpy(dtype="float32").reshape(-1, 1)
    pos_weight = compute_pos_weight(train_targets_np)
    weighted_loss_fn = build_weighted_loss_fn(pos_weight.to(device))

    print("\nStep 6 preview - device placement summary:")
    print(inspect_training_step_devices(model, first_train_batch, device))

    train_metrics = train_step(
        model=model,
        batch=first_train_batch,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    print("\nStep 6 preview - one training step metrics:")
    print(train_metrics)

    train_epoch_preview = run_train_epoch_preview(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        max_batches=10,
    )
    valid_epoch_output = run_validation_epoch(
        model=model,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nStep 7 preview - train epoch preview metrics:")
    print(train_epoch_preview)

    print("\nStep 7 preview - validation epoch summary:")
    print({
        "valid_loss": valid_epoch_output.loss,
        "logits_shape": valid_epoch_output.logits.shape,
        "probs_shape": valid_epoch_output.probs.shape,
        "targets_shape": valid_epoch_output.targets.shape,
    })

    metrics = compute_binary_classification_metrics(
        targets=valid_epoch_output.targets,
        probs=valid_epoch_output.probs,
        threshold=0.5,
    )
    print("\nStep 8 preview - validation metrics:")
    print(summarize_metrics(metrics))

    threshold_results = threshold_sweep(
        targets=valid_epoch_output.targets,
        probs=valid_epoch_output.probs,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    print("\nStep 9 preview - threshold sweep summary:")
    for result in threshold_results:
        print(result)

    weighted_train_metrics = train_step(
        model=model,
        batch=first_train_batch,
        optimizer=optimizer,
        loss_fn=weighted_loss_fn,
        device=device,
    )
    print("\nStep 10 preview - imbalance handling summary:")
    print({
        "pos_weight": float(pos_weight.item()),
        "weighted_train_step_loss": weighted_train_metrics["loss"],
        "weighted_train_step_mean_pred_prob": weighted_train_metrics["mean_pred_prob"],
    })

    checkpoint_path = save_model_checkpoint(
        model=model,
        path="artifacts/checkpoints/tabular_mlp_step11_demo.pt",
    )
    metadata_path = save_json_artifact(
        obj={
            "feature_cols": metadata.feature_cols,
            "categorical_cols": metadata.categorical_cols,
            "numerical_cols": metadata.numerical_cols,
            "numeric_fill_values": metadata.numeric_fill_values,
            "category_maps": metadata.category_maps,
        },
        path="artifacts/metadata/preprocessing_step11_demo.json",
    )
    inference_config_path = save_json_artifact(
        obj={
            "threshold": 0.4,
            "target_col": config.target_col,
            "id_cols": config.id_cols,
            "batch_size": config.batch_size,
        },
        path="artifacts/metadata/inference_config_step11_demo.json",
    )

    print("\nStep 11 preview - saved artifacts:")
    print({
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "inference_config_path": inference_config_path,
    })

    fresh_model = TabularMLP(
        cat_cardinalities=cat_cardinalities,
        num_numeric_features=len(metadata.numerical_cols),
        config=model_config,
    ).to(device)
    fresh_optimizer = build_optimizer(fresh_model, training_config)
    history = run_training_loop(
        model=fresh_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=fresh_optimizer,
        loss_fn=weighted_loss_fn,
        device=device,
        num_epochs=config.num_epochs_demo,
    )

    print("\nStep 12 preview - multi-epoch history:")
    for entry in history:
        print({
            "epoch": entry.epoch,
            "train_loss": entry.train_loss,
            "train_mean_pred_prob": entry.train_mean_pred_prob,
            "valid_loss": entry.valid_loss,
        })

    early_stopping_config = EarlyStoppingConfig(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        checkpoint_path="artifacts/checkpoints/best_model_step14.pt",
    )
    best_model = TabularMLP(
        cat_cardinalities=cat_cardinalities,
        num_numeric_features=len(metadata.numerical_cols),
        config=model_config,
    ).to(device)
    best_optimizer = build_optimizer(best_model, training_config)
    early_stopping_result = run_training_loop_with_early_stopping(
        model=best_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=best_optimizer,
        loss_fn=weighted_loss_fn,
        device=device,
        num_epochs=config.num_epochs_demo,
        early_stopping=early_stopping_config,
    )

    print("\nStep 14 preview - early stopping summary:")
    print({
        "best_valid_loss": early_stopping_result.best_valid_loss,
        "best_epoch": early_stopping_result.best_epoch,
        "stopped_early": early_stopping_result.stopped_early,
        "best_checkpoint_path": early_stopping_result.checkpoint_path,
    })
    for entry in early_stopping_result.history:
        print({
            "epoch": entry.epoch,
            "train_loss": entry.train_loss,
            "train_mean_pred_prob": entry.train_mean_pred_prob,
            "valid_loss": entry.valid_loss,
        })

    history_payload = [
        {
            "epoch": entry.epoch,
            "train_loss": entry.train_loss,
            "train_mean_pred_prob": entry.train_mean_pred_prob,
            "valid_loss": entry.valid_loss,
        }
        for entry in early_stopping_result.history
    ]
    history_path = save_json_artifact(
        obj=history_payload,
        path="artifacts/metrics/training_history_step15.json",
    )
    summary_text = (
        f"Best epoch: {early_stopping_result.best_epoch}\n"
        f"Best valid loss: {early_stopping_result.best_valid_loss:.6f}\n"
        f"Stopped early: {early_stopping_result.stopped_early}\n"
        f"Best checkpoint: {early_stopping_result.checkpoint_path}\n"
        f"Note: Tree-based baselines should still be compared for this tabular credit-risk task.\n"
    )
    summary_path = save_text_artifact(
        text=summary_text,
        path="artifacts/logs/experiment_summary_step15.txt",
    )

    print("\nStep 15 preview - saved experiment artifacts:")
    print({
        "history_path": history_path,
        "summary_path": summary_path,
        "baseline_note": "Compare PyTorch metrics later against LightGBM/XGBoost baseline before any production claim.",
    })

    baseline_result = run_xgb_baseline(
        train_df=split_data.train_df,
        valid_df=split_data.valid_df,
        target_col=config.target_col,
        categorical_cols=metadata.categorical_cols,
        numerical_cols=metadata.numerical_cols,
        random_state=config.random_state,
    )

    print("\nStep 16 preview - XGBoost baseline summary:")
    baseline_summary = {
        "xgb_roc_auc": baseline_result.roc_auc,
        "xgb_pr_auc": baseline_result.pr_auc,
        "xgb_positive_rate_pred": baseline_result.positive_rate_pred,
        "pytorch_roc_auc_reference": metrics.roc_auc,
        "pytorch_pr_auc_reference": metrics.pr_auc,
    }
    print(baseline_summary)

    baseline_artifact_path = save_json_artifact(
        obj=baseline_summary,
        path="artifacts/metrics/baseline_comparison_step16.json",
    )
    print("\nStep 16 preview - baseline artifact path:")
    print({"baseline_artifact_path": baseline_artifact_path})

    inference_artifacts = load_inference_artifacts(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        inference_config_path=inference_config_path,
        model_config=model_config,
        device=device,
    )
    inference_input = split_data.valid_df.head(5).drop(columns=[config.target_col]).copy()
    inference_output = score_dataframe(
        df=inference_input,
        artifacts=inference_artifacts,
    )

    print("\nStep 13 preview - inference output sample:")
    print(inference_output.to_dict(orient="records"))

    print("\nTorch device summary:")
    print(get_torch_device_summary())


if __name__ == "__main__":
    main()
