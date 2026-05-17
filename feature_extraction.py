"""
Feature extraction module for ApneaGuard.
Extracts heart rate variability (HRV) and RR-interval features from ECG signals.
"""

import numpy as np
from scipy import signal
from typing import Dict, List
import config


def detect_r_peaks(ecg_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Detect R-peaks in ECG signal using a simple peak detection algorithm.
    
    R-peaks are the prominent spikes in ECG corresponding to ventricular depolarization.
    They are used to calculate heart rate and RR intervals.
    
    Args:
        ecg_signal: Preprocessed ECG signal
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Array of R-peak indices (sample positions)
    """
    # Find peaks in the signal
    # Set minimum distance between peaks based on physiological constraints
    # Minimum heart rate ~30 bpm -> max interval ~2 seconds
    min_distance = int(sampling_rate * 0.5)  # 0.5 seconds
    
    # Set height threshold as a fraction of signal range
    threshold = 0.3 * (np.max(ecg_signal) - np.min(ecg_signal)) + np.min(ecg_signal)
    
    # Find peaks above threshold with minimum distance
    peaks, _ = signal.find_peaks(
        ecg_signal,
        height=threshold,
        distance=min_distance
    )
    
    return peaks


def calculate_rr_intervals(r_peaks: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Calculate RR intervals from R-peak positions.
    
    RR interval = time between consecutive R-peaks (heartbeats)
    
    Args:
        r_peaks: Array of R-peak indices
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Array of RR intervals in milliseconds
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    # Calculate differences between consecutive peaks
    rr_intervals_samples = np.diff(r_peaks)
    
    # Convert from samples to milliseconds
    rr_intervals_ms = (rr_intervals_samples / sampling_rate) * 1000
    
    return rr_intervals_ms


def extract_hrv_features(ecg_signal: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """
    Extract heart rate variability (HRV) features from ECG signal.
    
    HRV measures variation in time intervals between heartbeats.
    Changes in HRV are associated with sleep apnea events.
    
    Features extracted:
    - mean_rr: Mean RR interval (ms)
    - sdnn: Standard deviation of RR intervals (ms)
    - rmssd: Root mean square of successive differences (ms)
    - pnn50: Percentage of successive RR differences > 50ms
    - mean_hr: Mean heart rate (bpm)
    - std_hr: Standard deviation of heart rate (bpm)
    
    Args:
        ecg_signal: Preprocessed ECG signal
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Dictionary of HRV features
    """
    features = {}
    
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_signal, sampling_rate)
    
    # Check if we have enough peaks
    if len(r_peaks) < 2:
        # Return default values if not enough peaks detected
        return {
            'mean_rr': 0.0,
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0,
            'mean_hr': 0.0,
            'std_hr': 0.0
        }
    
    # Calculate RR intervals
    rr_intervals = calculate_rr_intervals(r_peaks, sampling_rate)
    
    if len(rr_intervals) == 0:
        return {
            'mean_rr': 0.0,
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0,
            'mean_hr': 0.0,
            'std_hr': 0.0
        }
    
    # Feature 1: Mean RR interval
    features['mean_rr'] = np.mean(rr_intervals)
    
    # Feature 2: SDNN - Standard deviation of RR intervals
    # Reflects overall HRV
    features['sdnn'] = np.std(rr_intervals)
    
    # Feature 3: RMSSD - Root mean square of successive differences
    # Reflects short-term HRV (parasympathetic activity)
    if len(rr_intervals) > 1:
        successive_diffs = np.diff(rr_intervals)
        features['rmssd'] = np.sqrt(np.mean(successive_diffs ** 2))
    else:
        features['rmssd'] = 0.0
    
    # Feature 4: pNN50 - Percentage of successive RR differences > 50ms
    # Another measure of parasympathetic activity
    if len(rr_intervals) > 1:
        successive_diffs = np.abs(np.diff(rr_intervals))
        nn50 = np.sum(successive_diffs > 50)
        features['pnn50'] = 100 * nn50 / len(successive_diffs)
    else:
        features['pnn50'] = 0.0
    
    # Feature 5 & 6: Heart rate statistics
    # Convert RR intervals (ms) to heart rate (bpm)
    heart_rates = 60000 / rr_intervals  # 60000 ms per minute
    features['mean_hr'] = np.mean(heart_rates)
    features['std_hr'] = np.std(heart_rates)
    
    return features


def extract_features_from_segments(segments: np.ndarray, 
                                   sampling_rate: int) -> np.ndarray:
    """
    Extract HRV features from all ECG segments.
    
    Args:
        segments: Array of ECG segments, shape (num_segments, window_samples)
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Array of feature vectors, shape (num_segments, num_features)
    """
    feature_list = []
    
    print(f"\nExtracting HRV features from {len(segments)} segments...")
    
    for i, segment in enumerate(segments):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(segments)} segments")
        
        # Extract features for this segment
        features = extract_hrv_features(segment, sampling_rate)
        
        # Convert to array in consistent order
        feature_vector = [
            features['mean_rr'],
            features['sdnn'],
            features['rmssd'],
            features['pnn50'],
            features['mean_hr'],
            features['std_hr']
        ]
        
        feature_list.append(feature_vector)
    
    features_array = np.array(feature_list)
    
    print(f"\nFeature extraction complete!")
    print(f"Feature array shape: {features_array.shape}")
    
    return features_array


def combine_raw_and_features(raw_segments: np.ndarray, 
                            hrv_features: np.ndarray) -> np.ndarray:
    """
    Combine raw ECG signals with extracted HRV features.
    
    This can improve model performance by providing both:
    - Raw temporal patterns (learned by CNN)
    - Engineered statistical features (HRV metrics)
    
    Args:
        raw_segments: Raw ECG segments, shape (num_segments, window_samples)
        hrv_features: HRV features, shape (num_segments, num_features)
        
    Returns:
        Combined feature array
        For CNN models: returns raw_segments with features appended as metadata
        Note: Actual implementation depends on model architecture
    """
    # For now, we'll return both separately
    # The model can decide how to use them
    return raw_segments, hrv_features


if __name__ == "__main__":
    """
    Test the feature extraction pipeline.
    """
    import data_loader
    import preprocessing
    
    print("Testing feature extraction...")
    
    # Load and preprocess sample data
    signals, annotations = data_loader.load_dataset(config.DATA_DIR, max_records=1)
    segments, labels = preprocessing.preprocess_dataset(signals, annotations)
    
    # Extract features from first few segments
    test_segments = segments[:10]
    features = extract_features_from_segments(test_segments, config.SAMPLING_RATE)
    
    print(f"\nSample features (first segment):")
    feature_names = ['mean_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr', 'std_hr']
    for name, value in zip(feature_names, features[0]):
        print(f"  {name}: {value:.2f}")
