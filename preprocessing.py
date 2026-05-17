"""
Preprocessing module for ApneaGuard.
Handles bandpass filtering, normalization, and segmentation of ECG signals.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List
import config


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, 
                           fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to remove noise from ECG signal.
    
    Bandpass filtering removes:
    - Low-frequency baseline wander (< 0.5 Hz)
    - High-frequency noise and muscle artifacts (> 40 Hz)
    
    Args:
        data: Input ECG signal
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (higher = steeper transition)
        
    Returns:
        Filtered ECG signal
    """
    # Calculate normalized frequencies (Nyquist frequency = fs/2)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    # Returns filter coefficients b (numerator) and a (denominator)
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply the filter using forward-backward filtering
    # filtfilt applies the filter twice (forward and reverse) to eliminate phase shift
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def normalize_signal(data: np.ndarray, method: str = 'z-score') -> np.ndarray:
    """
    Normalize ECG signal to a standard range.
    
    Normalization helps the neural network train more effectively by:
    - Putting all features on the same scale
    - Reducing the effect of amplitude variations
    - Improving gradient flow during backpropagation
    
    Args:
        data: Input ECG signal
        method: Normalization method ('z-score' or 'min-max')
        
    Returns:
        Normalized ECG signal
    """
    if method == 'z-score':
        # Z-score normalization: (x - mean) / std
        # Results in mean=0, std=1
        mean = np.mean(data)
        std = np.std(data)
        
        # Avoid division by zero
        if std < 1e-10:
            return data - mean
        
        normalized = (data - mean) / std
        
    elif method == 'min-max':
        # Min-max normalization: (x - min) / (max - min)
        # Results in range [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Avoid division by zero
        if max_val - min_val < 1e-10:
            return data - min_val
        
        normalized = (data - min_val) / (max_val - min_val)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def segment_signal(ecg_signal: np.ndarray, labels: np.ndarray, 
                   window_size: int, overlap: int, 
                   sampling_rate: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Segment ECG signal into fixed-length windows with corresponding labels.
    
    The PhysioNet database has minute-by-minute annotations.
    We create shorter windows (e.g., 60 seconds) with overlap for training.
    
    Args:
        ecg_signal: Continuous ECG signal
        labels: Minute-by-minute annotations
        window_size: Size of each window in seconds
        overlap: Overlap between windows in seconds
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Tuple of (segments, segment_labels):
            - segments: List of ECG windows
            - segment_labels: List of corresponding labels
    """
    # Convert time parameters to samples
    window_samples = window_size * sampling_rate
    overlap_samples = overlap * sampling_rate
    step_samples = window_samples - overlap_samples
    
    segments = []
    segment_labels = []
    
    # Calculate total duration in minutes
    total_minutes = len(labels)
    
    # Iterate through the signal with sliding window
    for start_sample in range(0, len(ecg_signal) - window_samples + 1, step_samples):
        end_sample = start_sample + window_samples
        
        # Extract window
        window = ecg_signal[start_sample:end_sample]
        
        # Determine which minute(s) this window corresponds to
        # Convert sample indices to minutes
        start_minute = int(start_sample / (sampling_rate * 60))
        end_minute = int(end_sample / (sampling_rate * 60))
        
        # Make sure we don't exceed available labels
        if end_minute >= total_minutes:
            break
        
        # Assign label based on majority vote of minutes in this window
        # If most minutes in the window are apnea, label as apnea
        window_labels = labels[start_minute:end_minute+1]
        
        if len(window_labels) > 0:
            # Use majority vote (mean > 0.5 means more apnea than normal)
            label = 1 if np.mean(window_labels) > 0.5 else 0
            
            segments.append(window)
            segment_labels.append(label)
    
    return segments, segment_labels


def preprocess_record(ecg_signal: np.ndarray, labels: np.ndarray, 
                      sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for a single record.
    
    Steps:
    1. Bandpass filtering to remove noise
    2. Normalization to standardize amplitude
    3. Segmentation into fixed-length windows
    
    Args:
        ecg_signal: Raw ECG signal
        labels: Minute-by-minute annotations
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Tuple of (processed_segments, segment_labels):
            - processed_segments: numpy array of shape (num_segments, window_samples)
            - segment_labels: numpy array of shape (num_segments,)
    """
    # Step 1: Apply bandpass filter
    filtered_signal = butter_bandpass_filter(
        ecg_signal,
        lowcut=config.LOWCUT,
        highcut=config.HIGHCUT,
        fs=sampling_rate,
        order=config.FILTER_ORDER
    )
    
    # Step 2: Normalize the signal
    normalized_signal = normalize_signal(
        filtered_signal,
        method=config.NORMALIZATION_METHOD
    )
    
    # Step 3: Segment into windows
    segments, segment_labels = segment_signal(
        normalized_signal,
        labels,
        window_size=config.WINDOW_SIZE,
        overlap=config.WINDOW_OVERLAP,
        sampling_rate=sampling_rate
    )
    
    # Convert lists to numpy arrays
    segments_array = np.array(segments)
    labels_array = np.array(segment_labels)
    
    return segments_array, labels_array


def preprocess_dataset(signals: List, annotations: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all records in the dataset.
    
    Args:
        signals: List of (ecg_signal, metadata) tuples
        annotations: List of label arrays
        
    Returns:
        Tuple of (all_segments, all_labels):
            - all_segments: numpy array of all preprocessed windows
            - all_labels: numpy array of all corresponding labels
    """
    all_segments = []
    all_labels = []
    
    print(f"\nPreprocessing {len(signals)} records...")
    
    for i, ((ecg_signal, metadata), labels) in enumerate(zip(signals, annotations), 1):
        print(f"[{i}/{len(signals)}] Preprocessing record...")
        
        # Get sampling rate from metadata
        sampling_rate = int(metadata['sampling_rate'])
        
        # Preprocess this record
        segments, segment_labels = preprocess_record(
            ecg_signal, labels, sampling_rate
        )
        
        print(f"  Generated {len(segments)} segments")
        print(f"  Apnea segments: {np.sum(segment_labels == 1)} ({100*np.mean(segment_labels==1):.1f}%)")
        
        all_segments.append(segments)
        all_labels.append(segment_labels)
    
    # Concatenate all segments
    all_segments = np.vstack(all_segments)
    all_labels = np.concatenate(all_labels)
    
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"Total segments: {len(all_segments)}")
    print(f"Segment shape: {all_segments[0].shape}")
    print(f"Apnea segments: {np.sum(all_labels == 1)} ({100*np.mean(all_labels==1):.1f}%)")
    print(f"Normal segments: {np.sum(all_labels == 0)} ({100*np.mean(all_labels==0):.1f}%)")
    print(f"{'='*50}\n")
    
    return all_segments, all_labels


if __name__ == "__main__":
    """
    Test the preprocessing pipeline.
    """
    import data_loader
    
    print("Testing preprocessing pipeline...")
    
    # Load sample data
    signals, annotations = data_loader.load_dataset(config.DATA_DIR, max_records=2)
    
    # Preprocess
    segments, labels = preprocess_dataset(signals, annotations)
    
    print(f"\nFinal output shape:")
    print(f"  Segments: {segments.shape}")
    print(f"  Labels: {labels.shape}")
