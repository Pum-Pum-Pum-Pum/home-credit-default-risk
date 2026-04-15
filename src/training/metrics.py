"""Evaluation metrics for binary classification in tabular PyTorch workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryClassificationMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    confusion_matrix: np.ndarray
    positive_rate_pred: float


def apply_threshold(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities into binary predictions."""
    return (probs >= threshold).astype(np.int64)


def compute_binary_classification_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """Compute core binary-classification metrics from targets and predicted probabilities."""
    targets_1d = targets.reshape(-1)
    probs_1d = probs.reshape(-1)
    preds_1d = apply_threshold(probs_1d, threshold=threshold)

    return BinaryClassificationMetrics(
        roc_auc=float(roc_auc_score(targets_1d, probs_1d)),
        pr_auc=float(average_precision_score(targets_1d, probs_1d)),
        precision=float(precision_score(targets_1d, preds_1d, zero_division=0)),
        recall=float(recall_score(targets_1d, preds_1d, zero_division=0)),
        f1=float(f1_score(targets_1d, preds_1d, zero_division=0)),
        threshold=float(threshold),
        confusion_matrix=confusion_matrix(targets_1d, preds_1d),
        positive_rate_pred=float(preds_1d.mean()),
    )


def summarize_metrics(metrics: BinaryClassificationMetrics) -> dict[str, object]:
    """Convert metrics dataclass into a printable summary dictionary."""
    return {
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "threshold": metrics.threshold,
        "positive_rate_pred": metrics.positive_rate_pred,
        "confusion_matrix": metrics.confusion_matrix.tolist(),
    }


def threshold_sweep(
    targets: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, object]]:
    """Evaluate metrics across multiple decision thresholds."""
    results: list[dict[str, object]] = []

    for threshold in thresholds:
        metrics = compute_binary_classification_metrics(
            targets=targets,
            probs=probs,
            threshold=threshold,
        )
        results.append(summarize_metrics(metrics))

    return results
