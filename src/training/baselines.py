"""Tree-based baseline utilities for tabular credit-risk comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


@dataclass
class BaselineResult:
    roc_auc: float
    pr_auc: float
    positive_rate_pred: float
    valid_probs: np.ndarray | None = None
    valid_targets: np.ndarray | None = None


def build_xgb_baseline(
    categorical_cols: list[str],
    numerical_cols: list[str],
    random_state: int,
) -> Pipeline:
    """Build a production-style sklearn + XGBoost baseline pipeline."""
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_pipe, categorical_cols),
        ("num", numerical_pipe, numerical_cols),
    ])

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        enable_categorical=False,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def run_xgb_baseline(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str],
    numerical_cols: list[str],
    random_state: int,
) -> BaselineResult:
    """Fit XGBoost baseline and evaluate on validation set."""
    pipeline = build_xgb_baseline(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        random_state=random_state,
    )

    feature_cols = categorical_cols + numerical_cols
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    pipeline.fit(X_train, y_train)
    valid_probs = pipeline.predict_proba(X_valid)[:, 1]
    valid_preds = (valid_probs >= 0.5).astype(np.int64)

    return BaselineResult(
        roc_auc=float(roc_auc_score(y_valid, valid_probs)),
        pr_auc=float(average_precision_score(y_valid, valid_probs)),
        positive_rate_pred=float(valid_preds.mean()),
        valid_probs=valid_probs,
        valid_targets=y_valid.to_numpy(),
    )