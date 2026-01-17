"""Dataset utilities for Puzzle-MoE.

This module provides a lightweight ECGDataset with semantic patch extraction,
patch permutation for self-supervision, and symbolic feature engineering.
Real DSP can replace the placeholder heuristics while keeping the API stable.
"""
from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

Permutation = Tuple[int, int, int]


class ECGDataset(Dataset):
    """Synthetic ECG dataset with semantic patching and symbolic features.

    Args:
        length: Number of samples to simulate when real data is unavailable.
        patch_size: Length of each semantic patch segment.
        mode: Either ``"ssl"`` for permutation prediction or ``"moe"`` for
            supervised classification.
        num_classes: Number of diagnostic labels for MoE fine-tuning.
        seed: Optional seed for deterministic sampling.
    """

    def __init__(
        self,
        length: int = 1024,
        patch_size: int = 64,
        mode: str = "ssl",
        num_classes: int = 5,
        seed: int | None = None,
        subset_ratio: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.length = max(1, int(length * subset_ratio))
        self.patch_size = patch_size
        self.mode = mode
        self.num_classes = num_classes
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._permutations: List[Permutation] = list(itertools.permutations(range(3)))

    def __len__(self) -> int:  # pragma: no cover - simple length getter
        return self.length

    def _simulate_signal(self) -> np.ndarray:
        """Simulate a single-lead ECG-like signal using sine components.

        Returns:
            An array shaped ``(3 * patch_size,)`` representing an ECG beat.
        """

        t = np.linspace(0, 1, self.patch_size * 3)
        p_wave = 0.1 * np.sin(2 * math.pi * 5 * t)
        qrs = 0.5 * np.sign(np.sin(2 * math.pi * 15 * t))
        t_wave = 0.2 * np.sin(2 * math.pi * 7 * t + 0.5)
        noise = 0.02 * np.random.randn(t.shape[0])
        return p_wave + qrs + t_wave + noise

    def _extract_patches(self, signal: np.ndarray) -> np.ndarray:
        """Slice the signal into P, QRS, and T wave patches.

        Args:
            signal: Input ECG beat array of length ``3 * patch_size``.

        Returns:
            Array of shape ``(3, patch_size)`` representing semantic patches.
        """

        assert signal.shape[0] == self.patch_size * 3, "Signal length mismatch"
        patches = signal.reshape(3, self.patch_size)
        return patches.astype(np.float32)

    def _extract_symbolic_features(self, signal: np.ndarray) -> np.ndarray:
        """Compute lightweight symbolic features.

        The placeholder features approximate heart rate variability (HRV) using
        signal standard deviation and QRS width via a simple peak threshold.

        Args:
            signal: Flattened ECG beat.

        Returns:
            Array containing symbolic features (HRV proxy, QRS width proxy).
        """

        hrv_proxy = float(np.std(signal))
        threshold = 0.3 * np.max(np.abs(signal))
        suprathreshold = np.where(np.abs(signal) > threshold)[0]
        qrs_width = float(suprathreshold.size / signal.shape[0])
        return np.array([hrv_proxy, qrs_width], dtype=np.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a sample for SSL or MoE training.

        For SSL, returns shuffled patches and a permutation label. For MoE,
        returns ordered patches, symbolic features, and a random label.
        """

        signal = self._simulate_signal()
        patches = self._extract_patches(signal)
        symbolic_features = self._extract_symbolic_features(signal)

        if self.mode == "ssl":
            perm_index = random.randrange(len(self._permutations))
            perm = self._permutations[perm_index]
            shuffled = patches[list(perm)]
            return {
                "patches": torch.from_numpy(shuffled),
                "permutation": torch.tensor(perm_index, dtype=torch.long),
                "symbolic_features": torch.from_numpy(symbolic_features),
            }

        label = random.randrange(self.num_classes)
        return {
            "patches": torch.from_numpy(patches),
            "label": torch.tensor(label, dtype=torch.long),
            "symbolic_features": torch.from_numpy(symbolic_features),
        }

    def permutation_space(self) -> int:
        """Return the number of unique patch permutations."""

        return len(self._permutations)


def load_ecg_dataset(path: str | Path, mode: str = "ssl", **kwargs: object) -> ECGDataset:
    """Factory to instantiate the ECGDataset.

    Args:
        path: Path to dataset root. Supports:
            - PTB-XL: path/to/ptbxl/fs500/superclasses
            - MIT-BIH: path/to/mitbih (future)
            - Mock data: any other path (uses synthetic data)
        mode: Either ``"ssl"`` or ``"moe"``.
        **kwargs: Additional arguments forwarded to dataset.

    Returns:
        An instance of a Dataset (PTBXLDataset or ECGDataset).
    """
    from pathlib import Path
    
    path = Path(path)
    
    # Check if PTB-XL preprocessed data is available
    if (path / 'train' / 'records.npz').exists():
        # Use real PTB-XL dataset with SOTA protocol
        try:
            # Try to import from linked project
            sys.path.insert(0, str(Path(__file__).parent.parent / 'ecg_ssl_project' / 'src'))
            from ptbxl_dataset import PTBXLDataset
            split = kwargs.pop('split', 'train')
            # Filter out arguments that PTBXLDataset doesn't accept
            ptbxl_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['name', 'path', 'batch_size', 'num_workers']}
            return PTBXLDataset(data_path=path, split=split, mode=mode, **ptbxl_kwargs)
        except ImportError as e:
            import warnings
            warnings.warn(f"PTBXLDataset not available ({e}), falling back to mock data")
    
    # Fall back to synthetic data
    return ECGDataset(mode=mode, **kwargs)


__all__ = ["ECGDataset", "load_ecg_dataset"]
