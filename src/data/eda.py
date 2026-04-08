"""Step 1 EDA utilities for the Home Credit application table."""

from __future__ import annotations

import pandas as pd


def load_main_table(data_path: str) -> pd.DataFrame:
    """Load the main application table into a pandas DataFrame."""
    return pd.read_csv(data_path)


def detect_feature_groups(
    df: pd.DataFrame,
    target_col: str,
    id_cols: list[str],
) -> dict[str, list[str]]:
    """Split columns into feature groups for tabular modeling."""
    feature_cols = [c for c in df.columns if c not in [target_col] + id_cols]
    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    binary_like_cat_cols = [c for c in cat_cols if df[c].nunique(dropna=False) <= 2]

    return {
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
        "binary_like_categorical_cols": binary_like_cat_cols,
    }


def summarize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Return counts and rates for the binary target."""
    counts = df[target_col].value_counts(dropna=False).sort_index()
    rates = (counts / len(df)).rename("rate")
    return pd.concat([counts.rename("count"), rates], axis=1)


def missing_value_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return missing-value ratios for feature columns."""
    summary = pd.DataFrame({
        "missing_count": df[feature_cols].isna().sum(),
        "missing_ratio": df[feature_cols].isna().mean(),
    })
    return summary.sort_values("missing_ratio", ascending=False)


def categorical_cardinality(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Return category cardinality including missing as a category candidate."""
    summary = pd.DataFrame({
        "n_unique_including_nan": [df[c].nunique(dropna=False) for c in cat_cols],
    }, index=cat_cols)
    return summary.sort_values("n_unique_including_nan", ascending=False)


def run_step1_eda(data_path: str, target_col: str, id_cols: list[str]) -> None:
    """Run and print core Step 1 EDA outputs."""
    df = load_main_table(data_path)
    groups = detect_feature_groups(df, target_col=target_col, id_cols=id_cols)

    print(f"shape: {df.shape}")
    print(f"n_features: {len(groups['feature_cols'])}")
    print(f"n_categorical: {len(groups['categorical_cols'])}")
    print(f"n_numerical: {len(groups['numerical_cols'])}")

    print("\nTarget summary:")
    print(summarize_target(df, target_col))

    print("\nTop 15 missing columns:")
    print(missing_value_summary(df, groups["feature_cols"]).head(15))

    print("\nCategorical cardinality:")
    print(categorical_cardinality(df, groups["categorical_cols"]))
