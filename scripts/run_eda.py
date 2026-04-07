"""Command-line entry point for Step 1 EDA."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import ProjectConfig
from src.data.eda import run_step1_eda
from src.utils.device import get_torch_device_summary


def main() -> None:
    config = ProjectConfig()
    run_step1_eda(
        data_path=config.data_path,
        target_col=config.target_col,
        id_cols=config.id_cols,
    )

    print("\nTorch device summary:")
    print(get_torch_device_summary())


if __name__ == "__main__":
    main()
