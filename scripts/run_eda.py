"""Command-line entry point for Step 1 EDA."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import ProjectConfig
from src.data.dataloaders import DataLoaderConfig, create_dataloaders, inspect_batch
from src.data.dataset import HomeCreditDataset, inspect_dataset_sample
from src.data.eda import run_step1_eda
from src.data.preprocessing import build_tabular_metadata, summarize_tabular_metadata
from src.data.splits import make_train_valid_split
from src.models.tabular_mlp import TabularMLP, TabularMLPConfig, inspect_model_forward_pass
from src.utils.device import get_torch_device_summary


def main() -> None:
    config = ProjectConfig()
    run_step1_eda(
        data_path=config.data_path,
        target_col=config.target_col,
        id_cols=config.id_cols,
    )

    from src.data.eda import load_main_table

    df = load_main_table(config.data_path)
    split_data = make_train_valid_split(
        df=df,
        target_col=config.target_col,
        valid_size=config.valid_size,
        random_state=config.random_state,
    )
    metadata = build_tabular_metadata(
        train_df=split_data.train_df,
        target_col=config.target_col,
        id_cols=config.id_cols,
    )

    print("\nStep 2 preview - split summary:")
    print(f"train shape: {split_data.train_df.shape}")
    print(f"valid shape: {split_data.valid_df.shape}")

    print("\nStep 2 preview - tabular metadata:")
    print(summarize_tabular_metadata(metadata))

    train_dataset = HomeCreditDataset(split_data.train_df, metadata)
    valid_dataset = HomeCreditDataset(split_data.valid_df, metadata)

    loader_config = DataLoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    train_loader, valid_loader = create_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        config=loader_config,
    )

    print("\nStep 3 preview - dataset lengths:")
    print(f"train dataset length: {len(train_dataset)}")
    print(f"valid dataset length: {len(valid_dataset)}")

    print("\nStep 3 preview - first sample tensor summary:")
    print(inspect_dataset_sample(train_dataset, idx=0))

    first_train_batch = next(iter(train_loader))
    first_valid_batch = next(iter(valid_loader))

    print("\nStep 4 preview - DataLoader batch summary:")
    print("train batch:", inspect_batch(first_train_batch))
    print("valid batch:", inspect_batch(first_valid_batch))

    cat_cardinalities = [
        len(metadata.category_maps[col]) + 1
        for col in metadata.categorical_cols
    ]
    model_config = TabularMLPConfig(
        embedding_dropout=config.embedding_dropout,
        mlp_hidden_dims=config.mlp_hidden_dims,
        mlp_dropout=config.mlp_dropout,
        use_batch_norm=config.use_batch_norm,
    )
    model = TabularMLP(
        cat_cardinalities=cat_cardinalities,
        num_numeric_features=len(metadata.numerical_cols),
        config=model_config,
    )

    print("\nStep 5 preview - model forward-pass summary:")
    print(inspect_model_forward_pass(model, first_train_batch))

    print("\nTorch device summary:")
    print(get_torch_device_summary())


if __name__ == "__main__":
    main()
