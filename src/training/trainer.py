"""Training utilities for GPU-aware PyTorch tabular modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from src.utils.checkpointing import save_model_checkpoint
from src.training.metrics import compute_binary_classification_metrics


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_pos_weight: bool = True


@dataclass
class ValidationOutput:
    loss: float
    logits: np.ndarray
    probs: np.ndarray
    targets: np.ndarray


@dataclass
class EpochHistoryEntry:
    epoch: int
    train_loss: float
    train_mean_pred_prob: float
    valid_loss: float
    valid_roc_auc: float | None = None
    valid_pr_auc: float | None = None
    epoch_seconds: float | None = None


@dataclass
class EarlyStoppingConfig:
    patience: int = 2
    min_delta: float = 0.0
    checkpoint_path: str = "artifacts/checkpoints/best_model_step14.pt"
    monitor: str = "valid_loss"


@dataclass
class TrainingLoopResult:
    history: list[EpochHistoryEntry]
    best_metric_value: float
    best_epoch: int
    stopped_early: bool
    checkpoint_path: str
    monitor: str
    total_training_seconds: float
    average_epoch_seconds: float


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Create optimizer for model training."""
    return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def build_loss_fn() -> nn.Module:
    """Binary classification loss on raw logits."""
    return nn.BCEWithLogitsLoss()


def compute_pos_weight(targets: np.ndarray) -> torch.Tensor:
    """Compute positive-class weighting for imbalanced binary classification.

    Formula:
        pos_weight = n_negative / n_positive
    """
    targets_1d = targets.reshape(-1)
    n_positive = float(targets_1d.sum())
    n_total = float(len(targets_1d))
    n_negative = n_total - n_positive

    if n_positive == 0:
        raise ValueError("Cannot compute pos_weight when there are no positive samples.")

    return torch.tensor(n_negative / n_positive, dtype=torch.float32)


def build_weighted_loss_fn(pos_weight: torch.Tensor) -> nn.Module:
    """Binary cross-entropy with positive-class weighting."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move an entire batch dictionary to the selected device."""
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one training step on a single batch."""
    model.train()
    batch = move_batch_to_device(batch, device)

    optimizer.zero_grad()

    logits = model(batch["x_cat"], batch["x_num"])
    loss = loss_fn(logits, batch["y"])
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        mean_prob = probs.mean().item()

    return {
        "loss": loss.item(),
        "mean_pred_prob": mean_prob,
    }


def validation_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float | torch.Tensor]:
    """Run one validation step on a single batch."""
    model.eval()
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        logits = model(batch["x_cat"], batch["x_num"])
        loss = loss_fn(logits, batch["y"])
        probs = torch.sigmoid(logits)

    return {
        "loss": float(loss.item()),
        "logits": logits.detach(),
        "probs": probs.detach(),
        "targets": batch["y"].detach(),
    }


def run_validation_epoch(
    model: nn.Module,
    valid_loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> ValidationOutput:
    """Run a full validation epoch and collect predictions/targets."""
    losses: list[float] = []
    logits_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for batch in valid_loader:
        step_output = validation_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            device=device,
        )
        losses.append(step_output["loss"])
        logits_list.append(step_output["logits"].cpu().numpy())
        probs_list.append(step_output["probs"].cpu().numpy())
        targets_list.append(step_output["targets"].cpu().numpy())

    return ValidationOutput(
        loss=float(np.mean(losses)),
        logits=np.concatenate(logits_list, axis=0),
        probs=np.concatenate(probs_list, axis=0),
        targets=np.concatenate(targets_list, axis=0),
    )


def run_train_epoch_preview(
    model: nn.Module,
    train_loader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int = 10,
) -> dict[str, float]:
    """Run a short train-epoch preview over a limited number of batches."""
    losses: list[float] = []
    probs: list[float] = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        step_metrics = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        losses.append(step_metrics["loss"])
        probs.append(step_metrics["mean_pred_prob"])

    return {
        "train_loss_mean": float(np.mean(losses)),
        "train_mean_pred_prob": float(np.mean(probs)),
        "num_train_batches_used": float(len(losses)),
    }


def run_train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one full training epoch."""
    losses: list[float] = []
    probs: list[float] = []

    for batch in train_loader:
        step_metrics = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        losses.append(step_metrics["loss"])
        probs.append(step_metrics["mean_pred_prob"])

    return {
        "train_loss_mean": float(np.mean(losses)),
        "train_mean_pred_prob": float(np.mean(probs)),
    }


def run_training_loop(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
) -> list[EpochHistoryEntry]:
    """Run a simple multi-epoch training loop with validation tracking."""
    history: list[EpochHistoryEntry] = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = perf_counter()
        train_metrics = run_train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        valid_output = run_validation_epoch(
            model=model,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history.append(EpochHistoryEntry(
            epoch=epoch,
            train_loss=train_metrics["train_loss_mean"],
            train_mean_pred_prob=train_metrics["train_mean_pred_prob"],
            valid_loss=valid_output.loss,
        ))

    return history


def run_training_loop_with_early_stopping(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
    early_stopping: EarlyStoppingConfig,
) -> TrainingLoopResult:
    """Run multi-epoch training with best-checkpoint tracking and early stopping."""
    history: list[EpochHistoryEntry] = []
    if early_stopping.monitor == "valid_loss":
        best_metric_value = float("inf")
        lower_is_better = True
    elif early_stopping.monitor in {"valid_roc_auc", "valid_pr_auc"}:
        best_metric_value = float("-inf")
        lower_is_better = False
    else:
        raise ValueError(f"Unsupported monitor: {early_stopping.monitor}")

    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, num_epochs + 1):
        epoch_start = perf_counter()
        train_metrics = run_train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        valid_output = run_validation_epoch(
            model=model,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        valid_metrics = compute_binary_classification_metrics(
            targets=valid_output.targets,
            probs=valid_output.probs,
            threshold=0.4,
        )

        entry = EpochHistoryEntry(
            epoch=epoch,
            train_loss=train_metrics["train_loss_mean"],
            train_mean_pred_prob=train_metrics["train_mean_pred_prob"],
            valid_loss=valid_output.loss,
            valid_roc_auc=valid_metrics.roc_auc,
            valid_pr_auc=valid_metrics.pr_auc,
            epoch_seconds=perf_counter() - epoch_start,
        )
        history.append(entry)

        current_metric = getattr(entry, early_stopping.monitor)
        if current_metric is None:
            raise ValueError(f"Monitored metric {early_stopping.monitor} is None")

        if lower_is_better:
            improved = current_metric < (best_metric_value - early_stopping.min_delta)
        else:
            improved = current_metric > (best_metric_value + early_stopping.min_delta)

        if improved:
            best_metric_value = float(current_metric)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_model_checkpoint(model, early_stopping.checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping.patience:
            stopped_early = True
            break

    epoch_times = [entry.epoch_seconds for entry in history if entry.epoch_seconds is not None]
    return TrainingLoopResult(
        history=history,
        best_metric_value=best_metric_value,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        checkpoint_path=early_stopping.checkpoint_path,
        monitor=early_stopping.monitor,
        total_training_seconds=float(sum(epoch_times)),
        average_epoch_seconds=float(np.mean(epoch_times)) if epoch_times else 0.0,
    )


def inspect_training_step_devices(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, str]:
    """Summarize where model parameters and batch tensors live."""
    model = model.to(device)
    batch = move_batch_to_device(batch, device)

    first_param = next(model.parameters())
    return {
        "model_device": str(first_param.device),
        "x_cat_device": str(batch["x_cat"].device),
        "x_num_device": str(batch["x_num"].device),
        "y_device": str(batch["y"].device),
    }
