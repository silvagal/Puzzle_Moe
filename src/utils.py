"""Utility functions for Puzzle-MoE."""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with stream output."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def expert_entropy(gating_probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of expert utilization per batch."""

    eps = 1e-8
    probs = gating_probs.clamp(min=eps)
    return -torch.sum(probs * torch.log(probs), dim=1).mean()


def save_checkpoint(
    state: Dict[str, Any], 
    is_best: bool = False, 
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint.pt",
) -> None:
    """Persist model and optimizer state to disk."""
    from pathlib import Path
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / filename
    torch.save(state, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


__all__ = ["set_seed", "create_logger", "expert_entropy", "save_checkpoint", "AverageMeter"]
