"""Production-style preprocessing preparation for PyTorch tabular data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TabularMetadata:
    target_col: str
    id_cols: list[str]
    feature_cols: list[str]
    categorical_cols: list[str]
    numerical_cols: list[str]
    category_maps: dict[str, dict[str, int]]
    numeric_fill_values: dict[str, float]


def infer_feature_groups(
    df: pd.DataFrame,
    target_col: str,
    id_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Infer feature, categorical, and numerical columns from the training frame."""
    feature_cols = [c for c in df.columns if c not in [target_col] + id_cols]
    categorical_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]
    return feature_cols, categorical_cols, numerical_cols


def build_category_maps(train_df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, dict[str, int]]:
    """Build category-to-index mappings for embedding inputs.

    Index 0 is reserved for unknown or missing categories.
    """
    category_maps: dict[str, dict[str, int]] = {}

    for col in categorical_cols:
        values = train_df[col].fillna("__MISSING__").astype(str).unique().tolist()
        values = sorted(values)
        category_maps[col] = {value: idx + 1 for idx, value in enumerate(values)}

    return category_maps


def build_numeric_fill_values(train_df: pd.DataFrame, numerical_cols: list[str]) -> dict[str, float]:
    """Build numeric imputation values from training data only.

    We use median for robustness against skew and outliers.
    """
    return {col: float(train_df[col].median()) for col in numerical_cols}


def build_tabular_metadata(
    train_df: pd.DataFrame,
    target_col: str,
    id_cols: list[str],
) -> TabularMetadata:
    """Create preprocessing metadata needed for later PyTorch Dataset conversion."""
    feature_cols, categorical_cols, numerical_cols = infer_feature_groups(
        train_df,
        target_col=target_col,
        id_cols=id_cols,
    )

    category_maps = build_category_maps(train_df, categorical_cols)
    numeric_fill_values = build_numeric_fill_values(train_df, numerical_cols)

    return TabularMetadata(
        target_col=target_col,
        id_cols=id_cols,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        category_maps=category_maps,
        numeric_fill_values=numeric_fill_values,
    )


def summarize_tabular_metadata(metadata: TabularMetadata) -> dict[str, object]:
    """Return a compact summary for sanity-checking preprocessing state."""
    embedding_cardinalities = {
        col: len(mapping) + 1 for col, mapping in metadata.category_maps.items()
    }
    return {
        "n_features": len(metadata.feature_cols),
        "n_categorical": len(metadata.categorical_cols),
        "n_numerical": len(metadata.numerical_cols),
        "embedding_cardinalities": embedding_cardinalities,
    }
