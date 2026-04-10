"""Base configuration for the Home Credit PyTorch tabular project."""

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    data_path: str = "data_from_kaggle/application_train.csv"
    test_path: str = "data_from_kaggle/application_test.csv"
    target_col: str = "TARGET"
    id_cols: list[str] = field(default_factory=lambda: ["SK_ID_CURR"])

    random_state: int = 42
    valid_size: float = 0.2

    artifacts_dir: str = "artifacts"
    checkpoints_dir: str = "artifacts/checkpoints"
    metrics_dir: str = "artifacts/metrics"
    logs_dir: str = "artifacts/logs"
