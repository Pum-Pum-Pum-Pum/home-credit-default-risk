"""Checkpointing and artifact-saving utilities for production-style training."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory for a file path exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_model_checkpoint(model: torch.nn.Module, path: str | Path) -> str:
    """Save PyTorch model state_dict checkpoint."""
    path = ensure_parent_dir(path)
    torch.save(model.state_dict(), path)
    return str(path)


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses and nested containers into JSON-serializable objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    return obj


def save_json_artifact(obj: Any, path: str | Path) -> str:
    """Save JSON artifact such as metadata, thresholds, or config."""
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    return str(path)


def save_text_artifact(text: str, path: str | Path) -> str:
    """Save plain-text artifact such as experiment notes or summaries."""
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)
