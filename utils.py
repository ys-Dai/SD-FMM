"""
Shared utility functions for SD-FMM.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def ensure_dirs(*dirs):
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_checkpoint(model: torch.nn.Module, path: str, epoch: int,
                    metrics: dict):
    """Save model state dict and metadata."""
    ensure_dirs(os.path.dirname(path))
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "metrics":     metrics,
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model: torch.nn.Module, path: str,
                    device: torch.device) -> dict:
    """Load model state dict from checkpoint; return metadata."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Checkpoint loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


def get_ranges(indices: np.ndarray) -> list:
    """Convert a sorted array of discrete indices into (start, end) intervals."""
    if len(indices) == 0:
        return []
    ranges, start, end = [], int(indices[0]), int(indices[0])
    for idx in indices[1:]:
        if idx == end + 1:
            end = int(idx)
        else:
            ranges.append((start, end + 1))
            start = int(idx)
            end   = int(idx)
    ranges.append((start, end + 1))
    return ranges


def format_metric(value: float, decimals: int = 4) -> str:
    """Format a metric value as a fixed-decimal string."""
    return f"{value:.{decimals}f}"
