"""PyTorch device helpers.

This module is intentionally defensive because PyTorch may not be installed yet.
"""

from __future__ import annotations


def get_torch_device_summary() -> dict[str, str | int | bool]:
    """Return torch/cuda environment details if PyTorch is installed."""
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "torch_installed": False,
            "message": "PyTorch is not installed in the current environment.",
        }

    cuda_available = torch.cuda.is_available()
    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "device": "cuda" if cuda_available else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
    }
