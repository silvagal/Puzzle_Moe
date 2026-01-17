"""PTB-XL Dataset Loader following SOTA Protocol.

This module implements the standard PTB-XL evaluation protocol used in literature:
- 500 Hz sampling rate
- 10-second records (5000 samples)
- 12-lead ECG
- Z-score normalization per channel
- 5 superclasses (NORM, MI, STTC, CD, HYP)
- Multi-label binary classification
- Macro AUROC as primary metric

Dataset split: train (17,439) / val (2,180) / test (2,180)

References:
- Strodthoff et al. (2020): PTB-XL benchmark paper
- https://physionet.org/content/ptb-xl/1.0.3/
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from src.signal_processing import apply_ecg_filters

__all__ = ["PTBXLDataset", "load_ptbxl_split"]


class PTBXLDataset(Dataset):
    """PTB-XL dataset with standard evaluation protocol.
    
    Args:
        data_path: Path to preprocessed data (contains train/val/test/records.npz).
        split: One of 'train', 'val', or 'test'.
        patch_size: Size of semantic patches (default: 64 samples).
        mode: Either 'ssl' for self-supervised jigsaw puzzle or 'moe' for supervised.
               In 'ssl' mode, returns shuffled patches and permutation index (no masking).
        num_classes: Number of diagnostic classes (default: 5 for superclasses).
        subset_ratio: Use a fraction of the dataset (default: 1.0 = all data).
        seed: Random seed for reproducibility.
        transform: Optional data augmentation function.
        leads: List of lead indices to use (default: all 12 leads).
    """
    
    # Standard PTB-XL configuration
    SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    SAMPLING_RATE = 500  # Hz
    RECORD_LENGTH = 10  # seconds
    N_SAMPLES = 5000  # 10s * 500 Hz
    N_LEADS = 12
    
    def __init__(
        self,
        data_path: str | Path,
        split: str = 'train',
        patch_size: int = 64,
        mode: str = 'ssl',
        num_classes: int = 5,
        subset_ratio: float = 1.0,
        seed: int | None = None,
        transform: Optional[object] = None,
        leads: Optional[List[int]] = None,
        include_patients: Optional[List[int]] = None,
        exclude_patients: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.patch_size = patch_size
        self.mode = mode
        self.num_classes = num_classes
        self.subset_ratio = subset_ratio
        self.seed = seed
        self.transform = transform
        self.leads = leads if leads is not None else list(range(self.N_LEADS))
        self.include_patients = include_patients
        self.exclude_patients = exclude_patients
        
        # Load preprocessed data
        split_file = self.data_path / split / 'records.npz'
        if not split_file.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found: {split_file}\n"
                f"Expected structure: {self.data_path}/{{train,val,test}}/records.npz"
            )
        
        data = np.load(split_file, allow_pickle=True)
        self.signals = data['signals']  # (N, 12, 5000) - normalized
        self.labels = data['labels']    # (N, 5) - binary multi-label
        self.patient_ids = data.get('patient_ids')
        
        # Validate data shape
        assert self.signals.shape[1] == self.N_LEADS, \
            f"Expected 12 leads, got {self.signals.shape[1]}"
        assert self.signals.shape[2] == self.N_SAMPLES, \
            f"Expected 5000 samples, got {self.signals.shape[2]}"
        assert self.labels.shape[1] == len(self.SUPERCLASSES), \
            f"Expected {len(self.SUPERCLASSES)} classes, got {self.labels.shape[1]}"
        
        # Apply patient filters if provided
        if self.patient_ids is not None:
            mask = np.ones(len(self.signals), dtype=bool)
            if include_patients is not None:
                mask &= np.isin(self.patient_ids, include_patients)
            if exclude_patients is not None:
                mask &= ~np.isin(self.patient_ids, exclude_patients)
            self.signals = self.signals[mask]
            self.labels = self.labels[mask]
            self.patient_ids = self.patient_ids[mask]
        
        # Apply subset ratio if needed
        if subset_ratio < 1.0:
            rng = np.random.RandomState(seed)
            n_samples = int(len(self.signals) * subset_ratio)
            indices = rng.choice(len(self.signals), size=n_samples, replace=False)
            self.signals = self.signals[indices]
            self.labels = self.labels[indices]
        
        # Compute statistics
        self.class_counts = self.labels.sum(axis=0)
        self.class_frequencies = self.class_counts / len(self.labels)
        
        print(f"PTB-XL {split}: {len(self.signals)} exams")
        print(f"  Sampling rate: {self.SAMPLING_RATE} Hz")
        print(f"  Record length: {self.RECORD_LENGTH}s ({self.N_SAMPLES} samples)")
        print(f"  Leads: {len(self.leads)}/{self.N_LEADS}")
        print(f"  Classes: {self.SUPERCLASSES}")
        print(f"  Class distribution: {dict(zip(self.SUPERCLASSES, self.class_counts.astype(int)))}")
        
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a single ECG record with labels.
        
        Returns:
            Dictionary containing:
            - 'signal': (n_leads, n_samples) raw ECG signal
            - 'patches': (3, patch_size) semantic patches (P, QRS, T)
            - 'labels': (5,) multi-label binary vector
            - 'symbolic_features': (n_features,) extracted symbolic features
            - 'exam_id': int, index of the exam
        """
        signal = self.signals[index]  # (12, 5000)
        label = self.labels[index]    # (5,)
        
        # Select specified leads
        signal = signal[self.leads]  # (n_leads, 5000)
        
        # Apply DSP filters (Bandpass + Notch)
        # Note: signal is (n_leads, n_samples)
        signal = apply_ecg_filters(signal, fs=self.SAMPLING_RATE)
        
        # Apply augmentation if in training mode
        # Transform should be an instance of ECGAugmentation from src.augmentation
        if self.transform is not None and self.split == 'train':
            # Apply to numpy array, transform returns numpy
            signal = self.transform(signal)
        
        # Extract semantic patches (P, QRS, T)
        # For now, we use a simple heuristic: divide signal into 3 equal parts
        # TODO: Replace with proper P-QRS-T segmentation using DSP
        patches = self._extract_semantic_patches(signal)
        
        # Extract symbolic features (HRV, QRS width, etc.)
        symbolic_features = self._extract_symbolic_features(signal)
        
        # Ensure patches are always 9 for consistency (replicate if needed)
        if patches.shape[0] == 3:
            # Fallback was used, replicate to 9 patches
            # patches is (3, 12, 64) -> want (9, 12, 64)
            patches = np.concatenate([patches] * 3, axis=0)
        
        # Convert to tensors
        result = {
            'signal': torch.from_numpy(signal).float(),
            'patches': torch.from_numpy(patches).float(),
            'labels': torch.from_numpy(label).float(),
            'symbolic_features': torch.from_numpy(symbolic_features).float(),
            'exam_id': index,
        }
        if self.patient_ids is not None:
            result['patient_id'] = torch.tensor(self.patient_ids[index], dtype=torch.long)
        
        # For SSL mode, add ordinal position prediction task
        if self.mode == 'ssl':
            # For SSL pretraining: shuffle patches from multiple heartbeats
            # Model must predict the original temporal position (0-8) of each patch
            
            from ecg_segmentation import create_temporal_shuffle_task
            
            # Patches are already extracted as (9, patch_size) from 3 beats
            # OR (3, patch_size) from fallback method
            # Convert from tensor to numpy if needed
            patches_np = patches if isinstance(patches, np.ndarray) else patches.numpy()
            
            # Check if we have 9 patches (semantic extraction) or 3 patches (fallback)
            if patches_np.shape[0] == 9:
                # Reshape to list of (P, QRS, T) tuples for create_temporal_shuffle_task
                patches_list = [
                    (patches_np[0], patches_np[1], patches_np[2]),  # Beat 1: P, QRS, T
                    (patches_np[3], patches_np[4], patches_np[5]),  # Beat 2: P, QRS, T
                    (patches_np[6], patches_np[7], patches_np[8]),  # Beat 3: P, QRS, T
                ]
                
                # Create shuffle task: returns (9, patch_size) shuffled patches and (9,) position labels
                shuffled_patches, position_labels = create_temporal_shuffle_task(
                    patches_list, n_patches_to_use=9
                )
            else:
                # Fallback: replicate the 3 patches to create 9 patches
                # This ensures compatibility when R-peak detection fails
                patches_replicated = np.tile(patches_np, (3, 1))  # (9, patch_size)
                patches_list = [
                    (patches_replicated[0], patches_replicated[1], patches_replicated[2]),
                    (patches_replicated[3], patches_replicated[4], patches_replicated[5]),
                    (patches_replicated[6], patches_replicated[7], patches_replicated[8]),
                ]
                shuffled_patches, position_labels = create_temporal_shuffle_task(
                    patches_list, n_patches_to_use=9
                )
            
            # Return shuffled patches and their original positions
            result['patches'] = torch.from_numpy(shuffled_patches).float()
            result['position_labels'] = torch.from_numpy(position_labels).long()
            
            # NOTE: Each patch must predict its original temporal position (0-8)
            # This is more learnable than predicting full permutation (9! = 362,880 classes)
        
        return result
    
    def _extract_semantic_patches(self, signal: np.ndarray) -> np.ndarray:
        """Extract P, QRS, and T wave patches from ECG signal using R-peak detection.
        
        Uses Pan-Tompkins algorithm to detect R-peaks, then extracts semantic patches
        (P-wave, QRS, T-wave) from multiple heartbeats for temporal reordering task.
        
        Args:
            signal: (n_leads, n_samples) ECG signal
            
        Returns:
            patches: (n_patches, n_leads, patch_size) temporal patches from multiple beats
                     For SSL mode: typically 9 patches from 3 consecutive beats
        """
        from src.ecg_segmentation import extract_multilead_heartbeat_patches
        
        n_leads, n_samples = signal.shape
        
        # Extract P, QRS, T patches from all heartbeats in the signal
        try:
            patches_list = extract_multilead_heartbeat_patches(
                signal,
                sampling_rate=self.SAMPLING_RATE,
                patch_size=self.patch_size,
                max_beats=5  # Extract up to 5 beats (we'll use 3 = 9 patches)
            )
            
            # Need at least 3 beats for the task
            if len(patches_list) < 3:
                # Fallback to simple division if not enough beats detected
                return self._extract_patches_fallback(signal)
            
            # Flatten first 3 beats into 9 patches: [P1, QRS1, T1, P2, QRS2, T2, P3, QRS3, T3]
            all_patches = []
            for i in range(min(3, len(patches_list))):
                p, qrs, t = patches_list[i]
                all_patches.extend([p, qrs, t])
            
            # Convert to array (9, n_leads, patch_size)
            patches = np.stack(all_patches[:9], axis=0).astype(np.float32)
            
            return patches
            
        except Exception as e:
            # Fallback to simple division if R-peak detection fails
            # import warnings
            # warnings.warn(f"R-peak detection failed, using fallback: {e}")
            return self._extract_patches_fallback(signal)
    
    def _extract_patches_fallback(self, signal: np.ndarray) -> np.ndarray:
        """Fallback method: simple division into 3 patches.
        
        Args:
            signal: (n_leads, n_samples) ECG signal
            
        Returns:
            patches: (3, n_leads, patch_size) patches from beginning, middle, end
        """
        n_leads, n_samples = signal.shape
        
        patch_starts = [
            0,                                    # Beginning
            n_samples // 2 - self.patch_size // 2,  # Middle
            n_samples - self.patch_size,          # End
        ]
        
        patches = []
        for start in patch_starts:
            end = start + self.patch_size
            if end > n_samples:
                patch = signal[:, start:]
                pad_width = self.patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, 0), (0, pad_width)), mode='edge')
            else:
                patch = signal[:, start:end]
            patches.append(patch)
        
        return np.stack(patches, axis=0).astype(np.float32)
    
    def _extract_symbolic_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract symbolic/rule-based features from ECG.
        
        Features extracted:
        - Heart rate variability (HRV) proxy: std of signal
        - QRS width proxy: fraction of signal above threshold
        - Signal energy: mean absolute amplitude
        - Peak-to-peak amplitude
        
        TODO: Implement proper feature extraction:
        - RR intervals (heart rate)
        - QRS duration
        - QT interval
        - ST segment deviation
        - PR interval
        
        Args:
            signal: (n_leads, n_samples) ECG signal
            
        Returns:
            features: (n_features,) symbolic feature vector
        """
        # Use lead II as reference
        reference_lead = signal[min(1, signal.shape[0] - 1)]
        
        # HRV proxy: standard deviation
        hrv_proxy = float(np.std(reference_lead))
        
        # QRS width proxy: fraction above threshold
        threshold = 0.3 * np.max(np.abs(reference_lead))
        qrs_width_proxy = float(np.mean(np.abs(reference_lead) > threshold))
        
        # Signal energy
        signal_energy = float(np.mean(np.abs(reference_lead)))
        
        # Peak-to-peak amplitude
        p2p_amplitude = float(np.max(reference_lead) - np.min(reference_lead))
        
        features = np.array([
            hrv_proxy,
            qrs_width_proxy,
            signal_energy,
            p2p_amplitude,
        ], dtype=np.float32)
        
        return features
    
    def _create_permutation(
        self, patches: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """Create a random permutation of patches for SSL task.
        
        Args:
            patches: (3, patch_size) original patches
            
        Returns:
            perm_index: int in [0, 5], index of the permutation (3! = 6 possibilities)
            permuted_patches: (3, patch_size) shuffled patches
        """
        import itertools
        import random
        
        # All possible permutations of 3 patches
        permutations = list(itertools.permutations(range(3)))
        
        # Select random permutation
        perm_index = random.randrange(len(permutations))
        perm = permutations[perm_index]
        
        # Apply permutation
        permuted_patches = patches[list(perm)]
        
        return perm_index, permuted_patches
    

    
    def get_class_weights(self, mode: str = 'inverse') -> torch.Tensor:
        """Compute class weights for imbalanced learning.
        
        Args:
            mode: Weight computation mode
                - 'inverse': N_total / (N_classes * N_pos)
                - 'sqrt': sqrt(N_neg / N_pos)
                - 'log': log(N_total / N_pos)
                
        Returns:
            weights: (num_classes,) class weights
        """
        n_total = len(self.labels)
        n_pos = self.class_counts
        n_neg = n_total - n_pos
        
        if mode == 'inverse':
            weights = n_total / (self.num_classes * n_pos)
        elif mode == 'sqrt':
            weights = np.sqrt(n_neg / n_pos)
        elif mode == 'log':
            weights = np.log1p(n_total / n_pos)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return torch.from_numpy(weights).float()


def load_ptbxl_split(
    data_path: str | Path,
    split: str = 'train',
    **kwargs: object,
) -> PTBXLDataset:
    """Factory function to load PTB-XL dataset.
    
    Args:
        data_path: Path to preprocessed PTB-XL data
        split: 'train', 'val', or 'test'
        **kwargs: Additional arguments for PTBXLDataset
        
    Returns:
        PTBXLDataset instance
    """
    return PTBXLDataset(data_path=data_path, split=split, **kwargs)
