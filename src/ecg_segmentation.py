"""
Utility functions for ECG heartbeat segmentation.

This module provides functions to:
1. Detect R-peaks using Pan-Tompkins algorithm
2. Extract semantic patches (P-wave, QRS, T-wave) from heartbeats
3. Support for temporal patch reordering SSL task
"""
import numpy as np
from scipy import signal as sp_signal
from typing import Tuple, List, Optional


def pan_tompkins_detector(ecg_signal: np.ndarray, 
                          sampling_rate: int = 500) -> np.ndarray:
    """
    Simplified Pan-Tompkins R-peak detector.
    
    Reference: Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
    IEEE transactions on biomedical engineering, (3), 230-236.
    
    Args:
        ecg_signal: (n_samples,) ECG signal (single lead)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        r_peaks: (n_peaks,) indices of detected R-peaks
    """
    # 1. Bandpass filter (5-15 Hz)
    nyquist_freq = sampling_rate / 2
    low_cutoff = 5 / nyquist_freq
    high_cutoff = 15 / nyquist_freq
    
    b, a = sp_signal.butter(1, [low_cutoff, high_cutoff], btype='band')
    filtered = sp_signal.filtfilt(b, a, ecg_signal)
    
    # 2. Derivative filter (emphasizes QRS slope)
    derivative = np.diff(filtered)
    
    # 3. Squaring (amplifies high frequencies)
    squared = derivative ** 2
    
    # 4. Moving average integration (smooth the signal)
    window_size = int(0.150 * sampling_rate)  # 150ms window
    moving_avg = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # 5. Peak detection
    # Use scipy's find_peaks with adaptive threshold
    threshold = 0.5 * np.mean(moving_avg)
    distance = int(0.2 * sampling_rate)  # Minimum 200ms between peaks (max 300 bpm)
    
    peaks, properties = sp_signal.find_peaks(
        moving_avg,
        height=threshold,
        distance=distance
    )
    
    return peaks


def extract_heartbeat_segment(ecg_signal: np.ndarray,
                              r_peak: int,
                              sampling_rate: int = 500,
                              before_r: float = 0.25,
                              after_r: float = 0.45) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extract a complete heartbeat segment centered on R-peak.
    
    Args:
        ecg_signal: (n_samples,) ECG signal
        r_peak: Index of R-peak
        sampling_rate: Sampling rate in Hz
        before_r: Seconds before R-peak to include (default 250ms for P-wave)
        after_r: Seconds after R-peak to include (default 450ms for T-wave)
        
    Returns:
        segment: (segment_length,) heartbeat segment, or None if invalid
        valid: Boolean indicating if segment is valid
    """
    before_samples = int(before_r * sampling_rate)
    after_samples = int(after_r * sampling_rate)
    
    start = r_peak - before_samples
    end = r_peak + after_samples
    
    # Check boundaries
    if start < 0 or end > len(ecg_signal):
        return None, False
    
    segment = ecg_signal[start:end]
    
    # Quality check: reject flat or too noisy segments
    if np.std(segment) < 0.01:  # Too flat
        return None, False
    
    if np.std(segment) > 3.0:  # Too noisy (after z-score normalization)
        return None, False
    
    return segment, True


def extract_pqrst_patches(heartbeat: np.ndarray,
                         r_peak_relative: int,
                         sampling_rate: int = 500,
                         patch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract P, QRS, and T patches from a heartbeat segment.
    
    Timing estimates (from medical literature):
    - P-wave: 80-120ms, occurs 120-200ms before QRS
    - QRS complex: 80-100ms, centered on R-peak
    - T-wave: 120-160ms, occurs 200-400ms after R-peak start
    
    Args:
        heartbeat: (segment_length,) heartbeat segment with R-peak included
        r_peak_relative: Index of R-peak within the segment
        sampling_rate: Sampling rate in Hz
        patch_size: Target size for each patch (will resize/pad if needed)
        
    Returns:
        p_patch: (patch_size,) P-wave patch
        qrs_patch: (patch_size,) QRS complex patch
        t_patch: (patch_size,) T-wave patch
    """
    # Convert time intervals to samples
    # P-wave: 120-200ms before R (use 160ms midpoint ± 40ms)
    p_center = r_peak_relative - int(0.16 * sampling_rate)  # 160ms before R
    p_half_width = int(0.04 * sampling_rate)  # 40ms = 20 samples @ 500Hz
    
    # QRS: R-peak ± 60ms (QRS is typically 80-120ms total)
    qrs_half_width = int(0.06 * sampling_rate)  # 60ms = 30 samples
    
    # T-wave: 240-360ms after R (use 300ms midpoint ± 60ms)
    t_center = r_peak_relative + int(0.30 * sampling_rate)  # 300ms after R
    t_half_width = int(0.06 * sampling_rate)  # 60ms = 30 samples
    
    # Extract patches
    p_start = max(0, p_center - p_half_width)
    p_end = min(len(heartbeat), p_center + p_half_width)
    p_patch = heartbeat[p_start:p_end]
    
    qrs_start = max(0, r_peak_relative - qrs_half_width)
    qrs_end = min(len(heartbeat), r_peak_relative + qrs_half_width)
    qrs_patch = heartbeat[qrs_start:qrs_end]
    
    t_start = max(0, t_center - t_half_width)
    t_end = min(len(heartbeat), t_center + t_half_width)
    t_patch = heartbeat[t_start:t_end]
    
    # Resize or pad to target patch_size
    p_patch = resize_or_pad(p_patch, patch_size)
    qrs_patch = resize_or_pad(qrs_patch, patch_size)
    t_patch = resize_or_pad(t_patch, patch_size)
    
    return p_patch, qrs_patch, t_patch


def resize_or_pad(patch: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize or pad a patch to target size.
    
    Args:
        patch: (n,) input patch
        target_size: desired output size
        
    Returns:
        resized_patch: (target_size,) patch
    """
    current_size = len(patch)
    
    if current_size == target_size:
        return patch
    
    elif current_size < target_size:
        # Pad with zeros (or edge values)
        pad_left = (target_size - current_size) // 2
        pad_right = target_size - current_size - pad_left
        return np.pad(patch, (pad_left, pad_right), mode='edge')
    
    else:
        # Downsample using interpolation
        indices = np.linspace(0, current_size - 1, target_size)
        return np.interp(indices, np.arange(current_size), patch)


def extract_all_heartbeat_patches(ecg_signal: np.ndarray,
                                  sampling_rate: int = 500,
                                  patch_size: int = 64,
                                  max_beats: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract P, QRS, T patches from all heartbeats in a signal.
    
    Args:
        ecg_signal: (n_samples,) ECG signal (single lead)
        sampling_rate: Sampling rate in Hz
        patch_size: Size of each patch
        max_beats: Maximum number of beats to extract (None = all)
        
    Returns:
        patches_list: List of (p_patch, qrs_patch, t_patch) tuples
    """
    # Detect R-peaks
    r_peaks = pan_tompkins_detector(ecg_signal, sampling_rate)
    
    if max_beats is not None:
        r_peaks = r_peaks[:max_beats]
    
    patches_list = []
    before_samples = int(0.25 * sampling_rate)
    
    for r_peak in r_peaks:
        # Extract heartbeat segment
        segment, valid = extract_heartbeat_segment(ecg_signal, r_peak, sampling_rate)
        
        if not valid:
            continue
        
        # R-peak is at position 'before_samples' in the segment
        r_peak_relative = before_samples
        
        # Extract P, QRS, T patches
        p_patch, qrs_patch, t_patch = extract_pqrst_patches(
            segment, r_peak_relative, sampling_rate, patch_size
        )
        
        patches_list.append((p_patch, qrs_patch, t_patch))
    
    return patches_list


def create_temporal_shuffle_task(patches_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 n_patches_to_use: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a temporal patch reordering task (Ordinal Position Classification).
    
    Strategy:
    - Select first 3 heartbeats (9 patches total: P1,QRS1,T1, P2,QRS2,T2, P3,QRS3,T3)
    - Randomly permute all patches
    - For each patch, model predicts its ORIGINAL temporal position (0-8)
    
    This is a multi-class classification where each patch independently predicts
    its position in the temporal sequence.
    
    Args:
        patches_list: List of (p, qrs, t) patch tuples from heartbeats
        n_patches_to_use: Number of patches to use (default 9 = 3 beats × 3 patches)
        
    Returns:
        shuffled_patches: (n_patches, patch_size) shuffled patches
        position_labels: (n_patches,) original temporal position for each patch
        
    Example:
        Original sequence: [P1, QRS1, T1, P2, QRS2, T2, P3, QRS3, T3]
        Positions:         [ 0,    1,  2,  3,    4,  5,  6,    7,  8]
        
        After shuffle:     [T2, P1, QRS3, T1, QRS1, P3, QRS2, T3, P2]
        Target labels:     [ 5,  0,    7,  2,    1,  6,    4,  8,  3]
        
        Model sees shuffled patches and must predict position labels.
    """
    # Flatten patches from first few beats
    all_patches = []
    n_beats_needed = (n_patches_to_use + 2) // 3  # Round up
    
    for i in range(min(n_beats_needed, len(patches_list))):
        p, qrs, t = patches_list[i]
        all_patches.extend([p, qrs, t])
    
    # Convert to array and trim to exact size
    all_patches = np.array(all_patches[:n_patches_to_use])  # (n_patches, patch_size)
    n_patches = len(all_patches)
    
    # Create original position labels (0, 1, 2, ..., n_patches-1)
    original_positions = np.arange(n_patches)
    
    # Create random permutation
    shuffled_indices = np.random.permutation(n_patches)
    
    # Apply shuffle
    shuffled_patches = all_patches[shuffled_indices]  # (n_patches, patch_size)
    
    # For each shuffled patch, its label is its original position
    # shuffled_patches[i] came from position original_positions[shuffled_indices[i]]
    position_labels = original_positions[shuffled_indices]
    
    return shuffled_patches, position_labels


def get_pqrst_indices(r_peak: int,
                      sampling_rate: int = 500,
                      signal_length: int = 5000) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Get start/end indices for P, QRS, T patches relative to the full signal.
    
    Args:
        r_peak: Index of R-peak in the full signal
        sampling_rate: Sampling rate in Hz
        signal_length: Length of the full signal
        
    Returns:
        p_indices: (start, end)
        qrs_indices: (start, end)
        t_indices: (start, end)
    """
    # P-wave: 160ms before R ± 40ms
    p_center = r_peak - int(0.16 * sampling_rate)
    p_half_width = int(0.04 * sampling_rate)
    p_start = max(0, p_center - p_half_width)
    p_end = min(signal_length, p_center + p_half_width)
    
    # QRS: R-peak ± 60ms
    qrs_half_width = int(0.06 * sampling_rate)
    qrs_start = max(0, r_peak - qrs_half_width)
    qrs_end = min(signal_length, r_peak + qrs_half_width)
    
    # T-wave: 300ms after R ± 60ms
    t_center = r_peak + int(0.30 * sampling_rate)
    t_half_width = int(0.06 * sampling_rate)
    t_start = max(0, t_center - t_half_width)
    t_end = min(signal_length, t_center + t_half_width)
    
    return (p_start, p_end), (qrs_start, qrs_end), (t_start, t_end)


def extract_multilead_heartbeat_patches(signal: np.ndarray,
                                        sampling_rate: int = 500,
                                        patch_size: int = 64,
                                        max_beats: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract P, QRS, T patches from all heartbeats in a MULTI-LEAD signal.
    
    Args:
        signal: (n_leads, n_samples) ECG signal
        sampling_rate: Sampling rate in Hz
        patch_size: Size of each patch
        max_beats: Maximum number of beats to extract
        
    Returns:
        patches_list: List of (p_patch, qrs_patch, t_patch) tuples.
                      Each patch is (n_leads, patch_size).
    """
    n_leads, n_samples = signal.shape
    
    # Use lead II (index 1) or last lead for R-peak detection
    ref_lead_idx = min(1, n_leads - 1)
    ref_lead = signal[ref_lead_idx]
    
    # Detect R-peaks
    r_peaks = pan_tompkins_detector(ref_lead, sampling_rate)
    
    if max_beats is not None:
        r_peaks = r_peaks[:max_beats]
    
    patches_list = []
    
    for r_peak in r_peaks:
        # Get indices for P, QRS, T
        (p_s, p_e), (qrs_s, qrs_e), (t_s, t_e) = get_pqrst_indices(r_peak, sampling_rate, n_samples)
        
        # Extract slices for all leads
        # signal[:, start:end] -> (n_leads, length)
        p_raw = signal[:, p_s:p_e]
        qrs_raw = signal[:, qrs_s:qrs_e]
        t_raw = signal[:, t_s:t_e]
        
        # Resize/Pad for each lead
        # We can vectorize this or loop. Since n_leads=12 is small, loop is fine.
        # But resize_or_pad expects 1D.
        
        def process_multilead_patch(raw_patch):
            # raw_patch: (n_leads, length)
            processed = np.zeros((n_leads, patch_size), dtype=raw_patch.dtype)
            for i in range(n_leads):
                processed[i] = resize_or_pad(raw_patch[i], patch_size)
            return processed

        p_patch = process_multilead_patch(p_raw)
        qrs_patch = process_multilead_patch(qrs_raw)
        t_patch = process_multilead_patch(t_raw)
        
        patches_list.append((p_patch, qrs_patch, t_patch))
    
    return patches_list
