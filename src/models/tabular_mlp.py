"""Tabular neural network with categorical embeddings and numeric features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def default_embedding_dim(cardinality: int) -> int:
    """Heuristic embedding dimension for a categorical feature."""
    return min(50, (cardinality + 1) // 2)


@dataclass
class TabularMLPConfig:
    embedding_dropout: float = 0.1
    mlp_hidden_dims: tuple[int, ...] = (256, 128)
    mlp_dropout: float = 0.2
    use_batch_norm: bool = True


class TabularMLP(nn.Module):
    """MLP for mixed tabular inputs.

    Inputs:
    - x_cat: shape [batch_size, n_cat_features], dtype torch.long
    - x_num: shape [batch_size, n_num_features], dtype torch.float32

    Output:
    - logits: shape [batch_size, 1], dtype torch.float32
    """

    def __init__(
        self,
        cat_cardinalities: list[int],
        num_numeric_features: int,
        config: TabularMLPConfig,
    ) -> None:
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.num_numeric_features = num_numeric_features
        self.config = config

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=default_embedding_dim(cardinality))
            for cardinality in cat_cardinalities
        ])

        total_embedding_dim = sum(
            embedding.embedding_dim for embedding in self.embedding_layers
        )
        self.total_input_dim = total_embedding_dim + num_numeric_features

        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        layers: list[nn.Module] = []
        in_dim = self.total_input_dim

        for hidden_dim in config.mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.mlp_dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embedded_features: list[torch.Tensor] = []

        for i, embedding_layer in enumerate(self.embedding_layers):
            embedded_feature = embedding_layer(x_cat[:, i])
            embedded_features.append(embedded_feature)

        if embedded_features:
            x_cat_emb = torch.cat(embedded_features, dim=1)
            x_cat_emb = self.embedding_dropout(x_cat_emb)
            x = torch.cat([x_cat_emb, x_num], dim=1)
        else:
            x = x_num

        logits = self.mlp(x)
        return logits


def inspect_model_forward_pass(
    model: TabularMLP,
    batch: dict[str, torch.Tensor],
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Run a no-grad forward pass and report tensor shapes/dtypes."""
    model.eval()
    with torch.no_grad():
        logits = model(batch["x_cat"], batch["x_num"])

    return {
        "x_cat": (tuple(batch["x_cat"].shape), batch["x_cat"].dtype),
        "x_num": (tuple(batch["x_num"].shape), batch["x_num"].dtype),
        "logits": (tuple(logits.shape), logits.dtype),
    }
