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
    batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = True
    embedding_dropout: float = 0.15
    mlp_hidden_dims: tuple[int, ...] = (256, 128)
    mlp_dropout: float = 0.3
    use_batch_norm: bool = True
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs_demo: int = 30
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.0

    artifacts_dir: str = "artifacts"
    checkpoints_dir: str = "artifacts/checkpoints"
    metrics_dir: str = "artifacts/metrics"
    logs_dir: str = "artifacts/logs"
