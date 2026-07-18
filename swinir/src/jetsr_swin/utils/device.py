from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Pick CUDA -> MPS -> CPU. Override with preferred='cuda'/'mps'/'cpu'."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    """AMP autocast is reliable on CUDA. MPS support is experimental; we disable it
    by default to avoid silent numerical issues."""
    return device.type == "cuda"
