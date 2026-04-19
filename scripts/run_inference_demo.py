"""Dedicated inference demo script using saved artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import ProjectConfig
from src.data.eda import load_main_table
from src.data.splits import make_train_valid_split
from src.inference.pipeline import load_inference_artifacts, score_dataframe
from src.models.tabular_mlp import TabularMLPConfig
from src.training.trainer import get_device


def main() -> None:
    config = ProjectConfig()
    df = load_main_table(config.data_path)
    split_data = make_train_valid_split(
        df=df,
        target_col=config.target_col,
        valid_size=config.valid_size,
        random_state=config.random_state,
    )

    model_config = TabularMLPConfig(
        embedding_dropout=config.embedding_dropout,
        mlp_hidden_dims=config.mlp_hidden_dims,
        mlp_dropout=config.mlp_dropout,
        use_batch_norm=config.use_batch_norm,
    )
    device = get_device()
    artifacts = load_inference_artifacts(
        checkpoint_path="artifacts/checkpoints/tabular_mlp_step11_demo.pt",
        metadata_path="artifacts/metadata/preprocessing_step11_demo.json",
        inference_config_path="artifacts/metadata/inference_config_step11_demo.json",
        model_config=model_config,
        device=device,
    )

    inference_input = split_data.valid_df.head(5).drop(columns=[config.target_col]).copy()
    output = score_dataframe(inference_input, artifacts)
    print(output.to_dict(orient="records"))


if __name__ == "__main__":
    main()
