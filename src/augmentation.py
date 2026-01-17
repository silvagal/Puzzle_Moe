"""
ECG data augmentation module.

Implements ECG-specific augmentations based on prior literature:
- Ribeiro et al. (2020): amplitude scaling + Gaussian noise
- Strodthoff et al. (2021): time warping + mixup
- Hong et al. (2020): baseline wander

Each augmentation simulates realistic variations observed in clinical ECGs.
"""

import numpy as np
import torch
import scipy.signal
from typing import Tuple


class ECGAugmentation:
    """ECG augmentations for 12-lead signals.

    Applied transforms:
    - Amplitude scaling: simulates gain variation across devices
    - Time warping: simulates heart-rate variability
    - Baseline wander: simulates respiration/motion artifacts
    - Gaussian noise: simulates electronic interference
    """

    def __init__(
        self,
        amplitude_scale_range: Tuple[float, float] = (0.8, 1.2),
        time_warp_range: Tuple[float, float] = (0.9, 1.1),
        baseline_wander_amplitude: float = 0.1,
        baseline_wander_freq_range: Tuple[float, float] = (0.2, 0.5),
        gaussian_noise_std: float = 0.05,
        prob: float = 0.5,
    ):
        """Initialize augmentation parameters.

        Args:
            amplitude_scale_range: Amplitude scaling range (min, max).
            time_warp_range: Time-warp factor range (min, max).
            baseline_wander_amplitude: Baseline-wander amplitude (0-1).
            baseline_wander_freq_range: Baseline-wander frequency range (Hz).
            gaussian_noise_std: Gaussian noise standard deviation.
            prob: Probability of applying each augmentation.
        """
        self.amplitude_scale_range = amplitude_scale_range
        self.time_warp_range = time_warp_range
        self.baseline_wander_amplitude = baseline_wander_amplitude
        self.baseline_wander_freq_range = baseline_wander_freq_range
        self.gaussian_noise_std = gaussian_noise_std
        self.prob = prob

        # Available augmentations
        self.augmentations = [
            self.amplitude_scaling,
            self.time_warping,
            self.baseline_wander,
            self.gaussian_noise,
        ]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations to an ECG signal.

        Args:
            signal: ECG signal with shape (n_leads, n_samples) or (n_samples,).

        Returns:
            Augmented signal with the same shape.
        """
        augmented = signal.copy()

        # Apply each augmentation with probability self.prob.
        for aug_fn in self.augmentations:
            if np.random.rand() < self.prob:
                augmented = aug_fn(augmented)

        return augmented

    def amplitude_scaling(self, signal: np.ndarray) -> np.ndarray:
        """Scale signal amplitude.

        Simulates gain variation across ECG devices or physiological differences.

        Args:
            signal: ECG signal.

        Returns:
            Amplitude-scaled signal.
        """
        scale = np.random.uniform(*self.amplitude_scale_range)
        return signal * scale

    def time_warping(self, signal: np.ndarray) -> np.ndarray:
        """Apply temporal warping.

        Simulates heart-rate variability while preserving signal quality via
        interpolation.

        Args:
            signal: ECG signal with shape (n_leads, n_samples) or (n_samples,).

        Returns:
            Time-warped signal.
        """
        warp_factor = np.random.uniform(*self.time_warp_range)

        # Determine the warped length.
        if signal.ndim == 1:
            n_samples = len(signal)
            new_length = int(n_samples * warp_factor)

            # Resample to the new length then back.
            warped = scipy.signal.resample(signal, new_length)
            result = scipy.signal.resample(warped, n_samples)
        else:
            n_leads, n_samples = signal.shape
            new_length = int(n_samples * warp_factor)

            # Apply warping per lead.
            warped = np.zeros((n_leads, new_length))
            for i in range(n_leads):
                warped[i] = scipy.signal.resample(signal[i], new_length)

            # Resample back to the original length.
            result = np.zeros((n_leads, n_samples))
            for i in range(n_leads):
                result[i] = scipy.signal.resample(warped[i], n_samples)

        return result

    def baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Add baseline wander.

        Simulates respiration or patient-motion artifacts that cause
        low-frequency drift in the ECG baseline.

        Args:
            signal: ECG signal with shape (n_leads, n_samples) or (n_samples,).

        Returns:
            Signal with baseline wander added.
        """
        if signal.ndim == 1:
            n_samples = len(signal)

            # Generate low-frequency baseline wander.
            freq = np.random.uniform(*self.baseline_wander_freq_range)
            t = np.linspace(0, n_samples / 500, n_samples)  # Assume 500 Hz.
            wander = self.baseline_wander_amplitude * np.sin(2 * np.pi * freq * t)

            return signal + wander

        n_leads, n_samples = signal.shape
        result = signal.copy()

        # Add baseline wander per lead (shared phase for coherent motion).
        freq = np.random.uniform(*self.baseline_wander_freq_range)
        t = np.linspace(0, n_samples / 500, n_samples)
        phase = np.random.uniform(0, 2 * np.pi)
        wander = self.baseline_wander_amplitude * np.sin(2 * np.pi * freq * t + phase)

        for i in range(n_leads):
            # Vary amplitude per lead.
            lead_amplitude = np.random.uniform(0.5, 1.5)
            result[i] += lead_amplitude * wander

        return result

    def gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise.

        Simulates electronic interference, quantization noise, or EMG noise.

        Args:
            signal: ECG signal.

        Returns:
            Signal with Gaussian noise added.
        """
        noise = np.random.normal(0, self.gaussian_noise_std, signal.shape)
        return signal + noise


class ECGMixup:
    """Mixup for ECG signals.

    Reference: Zhang et al. (2018) - "mixup: Beyond Empirical Risk Minimization"
    Adapted for ECG by Strodthoff et al. (2021).
    """

    def __init__(self, alpha: float = 0.4):
        """Initialize mixup coefficient.

        Args:
            alpha: Beta distribution parameter (higher = more mixing).
        """
        self.alpha = alpha

    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup between two examples.

        Args:
            x1, y1: First example and label.
            x2, y2: Second example and label.

        Returns:
            (mixed_x, mixed_y): Mixed example and label.
        """
        # Sample lambda from Beta distribution.
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix signals.
        mixed_x = lam * x1 + (1 - lam) * x2

        # Mix labels (soft labels for BCE loss).
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y


class ECGCutMix:
    """CutMix for ECG signals.

    Reference: Yun et al. (2019) - "CutMix: Regularization Strategy"
    Adapted for ECG time series.
    """

    def __init__(self, alpha: float = 1.0):
        """Initialize CutMix coefficient.

        Args:
            alpha: Beta distribution parameter.
        """
        self.alpha = alpha

    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix between two examples.

        Args:
            x1, y1: First example and label (shape: [n_leads, n_samples]).
            x2, y2: Second example and label.

        Returns:
            (mixed_x, mixed_y): Mixed example and label.
        """
        # Sample lambda from Beta distribution.
        lam = np.random.beta(self.alpha, self.alpha)

        # Determine cut length and position.
        n_samples = x1.size(-1)
        cut_len = int(n_samples * (1 - lam))
        cut_start = np.random.randint(0, n_samples - cut_len + 1)

        # Replace segment.
        mixed_x = x1.clone()
        mixed_x[..., cut_start:cut_start + cut_len] = x2[..., cut_start:cut_start + cut_len]

        # Adjust labels proportionally.
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y


def apply_mixup_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    alpha: float = 0.4,
    prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup to a batch with probability ``prob``.

    Args:
        batch_x: Batch of signals (batch_size, n_leads, n_samples).
        batch_y: Batch of labels (batch_size, n_classes).
        alpha: Mixup parameter.
        prob: Probability of applying mixup.

    Returns:
        (mixed_batch_x, mixed_batch_y)
    """
    if np.random.rand() > prob:
        return batch_x, batch_y

    batch_size = batch_x.size(0)

    # Shuffle indices to create pairs.
    indices = torch.randperm(batch_size)

    # Apply mixup.
    mixup = ECGMixup(alpha=alpha)
    mixed_x, mixed_y = mixup(batch_x, batch_y, batch_x[indices], batch_y[indices])

    return mixed_x, mixed_y


def apply_cutmix_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    alpha: float = 1.0,
    prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply CutMix to a batch with probability ``prob``.

    Args:
        batch_x: Batch of signals (batch_size, n_leads, n_samples).
        batch_y: Batch of labels (batch_size, n_classes).
        alpha: CutMix parameter.
        prob: Probability of applying CutMix.

    Returns:
        (mixed_batch_x, mixed_batch_y)
    """
    if np.random.rand() > prob:
        return batch_x, batch_y

    batch_size = batch_x.size(0)

    # Shuffle indices to create pairs.
    indices = torch.randperm(batch_size)

    # Apply CutMix.
    cutmix = ECGCutMix(alpha=alpha)
    mixed_x, mixed_y = cutmix(batch_x, batch_y, batch_x[indices], batch_y[indices])

    return mixed_x, mixed_y
