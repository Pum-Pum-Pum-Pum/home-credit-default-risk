"""Training utilities for GPU-aware PyTorch tabular modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer


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
