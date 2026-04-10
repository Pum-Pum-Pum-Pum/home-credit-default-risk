"""Command-line entry point for Step 1 EDA."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import ProjectConfig
from src.data.dataset import HomeCreditDataset, inspect_dataset_sample
from src.data.eda import run_step1_eda
from src.data.preprocessing import build_tabular_metadata, summarize_tabular_metadata
from src.data.splits import make_train_valid_split
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

    print("\nStep 3 preview - dataset lengths:")
    print(f"train dataset length: {len(train_dataset)}")
    print(f"valid dataset length: {len(valid_dataset)}")

    print("\nStep 3 preview - first sample tensor summary:")
    print(inspect_dataset_sample(train_dataset, idx=0))

    print("\nTorch device summary:")
    print(get_torch_device_summary())


if __name__ == "__main__":
    main()
