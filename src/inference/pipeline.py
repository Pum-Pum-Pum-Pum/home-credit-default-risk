"""Inference pipeline for loading artifacts and scoring tabular rows consistently."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.dataset import HomeCreditDataset
from src.data.preprocessing import TabularMetadata
from src.models.tabular_mlp import TabularMLP, TabularMLPConfig


@dataclass
class InferenceArtifacts:
    model: TabularMLP
    metadata: TabularMetadata
    inference_config: dict[str, Any]
    device: torch.device


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON artifact from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tabular_metadata(path: str | Path, target_col: str, id_cols: list[str]) -> TabularMetadata:
    """Rebuild TabularMetadata from saved preprocessing JSON."""
    payload = load_json(path)
    return TabularMetadata(
        target_col=target_col,
        id_cols=id_cols,
        feature_cols=payload["feature_cols"],
        categorical_cols=payload["categorical_cols"],
        numerical_cols=payload["numerical_cols"],
        category_maps=payload["category_maps"],
        numeric_fill_values=payload["numeric_fill_values"],
    )


def load_inference_artifacts(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    inference_config_path: str | Path,
    model_config: TabularMLPConfig,
    device: torch.device,
) -> InferenceArtifacts:
    """Load model checkpoint, preprocessing metadata, and inference config."""
    inference_config = load_json(inference_config_path)
    metadata = load_tabular_metadata(
        path=metadata_path,
        target_col=inference_config["target_col"],
        id_cols=inference_config["id_cols"],
    )

    cat_cardinalities = [
        len(metadata.category_maps[col]) + 1
        for col in metadata.categorical_cols
    ]

    model = TabularMLP(
        cat_cardinalities=cat_cardinalities,
        num_numeric_features=len(metadata.numerical_cols),
        config=model_config,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return InferenceArtifacts(
        model=model,
        metadata=metadata,
        inference_config=inference_config,
        device=device,
    )


def prepare_inference_dataframe(df: pd.DataFrame, metadata: TabularMetadata) -> pd.DataFrame:
    """Ensure inference DataFrame contains required columns in training order."""
    inference_df = df.copy()
    for col in metadata.feature_cols:
        if col not in inference_df.columns:
            inference_df[col] = pd.NA

    ordered_cols = metadata.id_cols + metadata.feature_cols
    return inference_df[ordered_cols]


def score_dataframe(
    df: pd.DataFrame,
    artifacts: InferenceArtifacts,
) -> pd.DataFrame:
    """Score a DataFrame using saved artifacts and return probabilities + decisions."""
    inference_df = prepare_inference_dataframe(df, artifacts.metadata)
    inference_df = inference_df.copy()
    inference_df[artifacts.metadata.target_col] = 0.0

    dataset = HomeCreditDataset(inference_df, artifacts.metadata)

    with torch.no_grad():
        x_cat = dataset.x_cat.to(artifacts.device)
        x_num = dataset.x_num.to(artifacts.device)
        logits = artifacts.model(x_cat, x_num)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    threshold = float(artifacts.inference_config["threshold"])
    decisions = (probs >= threshold).astype(int)

    result = pd.DataFrame({
        "score_probability": probs,
        "predicted_label": decisions,
    })

    for id_col in artifacts.metadata.id_cols:
        if id_col in df.columns:
            result[id_col] = df[id_col].values

    return result
