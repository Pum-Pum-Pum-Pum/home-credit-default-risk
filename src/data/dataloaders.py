"""DataLoader utilities for Home Credit tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.data.dataset import HomeCreditDataset


@dataclass
class DataLoaderConfig:
    batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_train: bool = False


def create_dataloaders(
    train_dataset: HomeCreditDataset,
    valid_dataset: HomeCreditDataset,
    config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last_train,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, valid_loader


def inspect_batch(batch: dict[str, torch.Tensor]) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return shape/dtype summary for a DataLoader batch."""
    return {
        key: (tuple(value.shape), value.dtype)
        for key, value in batch.items()
    }
