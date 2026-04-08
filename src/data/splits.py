"""Train/validation split utilities for tabular modeling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame


def make_train_valid_split(
    df: pd.DataFrame,
    target_col: str,
    valid_size: float,
    random_state: int,
) -> SplitData:
    """Create a stratified train/validation split for binary classification."""
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    return SplitData(train_df=train_df, valid_df=valid_df)
