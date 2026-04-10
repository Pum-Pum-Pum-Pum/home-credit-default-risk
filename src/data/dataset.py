"""PyTorch Dataset for tabular application data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import TabularMetadata


@dataclass
class EncodedSample:
    x_cat: torch.Tensor
    x_num: torch.Tensor
    y: torch.Tensor


class HomeCreditDataset(Dataset):
    """Dataset that converts tabular rows into PyTorch-ready tensors.

    Output contract:
    - x_cat: categorical indices, dtype torch.long, shape [n_categorical_features]
    - x_num: numerical values, dtype torch.float32, shape [n_numerical_features]
    - y: binary target, dtype torch.float32, shape [1]
    """

    def __init__(self, df: pd.DataFrame, metadata: TabularMetadata) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.metadata = metadata

        self.x_cat = self._encode_categorical_block(self.df, metadata)
        self.x_num = self._encode_numerical_block(self.df, metadata)
        self.y = self._encode_target(self.df, metadata.target_col)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_cat": self.x_cat[idx],
            "x_num": self.x_num[idx],
            "y": self.y[idx],
        }

    @staticmethod
    def _encode_categorical_block(df: pd.DataFrame, metadata: TabularMetadata) -> torch.Tensor:
        encoded_cols: list[np.ndarray] = []

        for col in metadata.categorical_cols:
            mapping = metadata.category_maps[col]
            encoded = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
                .map(lambda value: mapping.get(value, 0))
                .to_numpy(dtype=np.int64)
            )
            encoded_cols.append(encoded)

        if not encoded_cols:
            return torch.empty((len(df), 0), dtype=torch.long)

        x_cat = np.column_stack(encoded_cols)
        return torch.tensor(x_cat, dtype=torch.long)

    @staticmethod
    def _encode_numerical_block(df: pd.DataFrame, metadata: TabularMetadata) -> torch.Tensor:
        numeric_df = df[metadata.numerical_cols].copy()

        for col in metadata.numerical_cols:
            numeric_df[col] = numeric_df[col].fillna(metadata.numeric_fill_values[col])

        if numeric_df.shape[1] == 0:
            return torch.empty((len(df), 0), dtype=torch.float32)

        return torch.tensor(numeric_df.to_numpy(dtype=np.float32), dtype=torch.float32)

    @staticmethod
    def _encode_target(df: pd.DataFrame, target_col: str) -> torch.Tensor:
        target = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        return torch.tensor(target, dtype=torch.float32)


def inspect_dataset_sample(dataset: HomeCreditDataset, idx: int = 0) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return sample tensor shape and dtype summary for debugging."""
    sample = dataset[idx]
    return {
        key: (tuple(value.shape), value.dtype)
        for key, value in sample.items()
    }
