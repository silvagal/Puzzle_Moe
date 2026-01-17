import numpy as np
from scipy.signal import butter, sosfilt, iirnotch

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply filter along the last axis (time)
    y = sosfilt(sos, data, axis=-1)
    return y

def notch_filter(data, cutoff, fs, Q=30):
    nyq = 0.5 * fs
    freq = cutoff / nyq
    b, a = iirnotch(freq, Q)
    # Apply filter along the last axis (time)
    # Note: scipy's lfilter/filtfilt might be needed for b, a
    # But for simplicity and stability with sos, we can stick to bandpass mostly.
    # If we strictly need notch, we use lfilter.
    from scipy.signal import lfilter
    y = lfilter(b, a, data, axis=-1)
    return y

def apply_ecg_filters(signal, fs=500.0):
    """
    Apply standard ECG filters:
    1. Bandpass 0.5 - 50 Hz (Remove baseline wander and high freq noise)
    2. Notch 60 Hz (Remove power line interference - assuming 60Hz for now, PTB-XL is mostly 50Hz but 60Hz is common too. 
       Actually PTB-XL is European, so 50Hz is more likely. Let's apply 50Hz Notch.)
    """
    # Bandpass
    filtered = butter_bandpass_filter(signal, 0.5, 50.0, fs, order=3)
    
    # Notch 50Hz (Europe/PTB-XL)
    filtered = notch_filter(filtered, 50.0, fs)
    
    return filtered
