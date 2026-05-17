"""
Data loader module for ApneaGuard.
Handles loading ECG signals and annotations from PhysioNet Apnea-ECG Database.
"""

import os
import numpy as np
import wfdb
from typing import Tuple, List, Dict
import config


def get_record_list(data_dir: str) -> List[str]:
    """
    Get list of all available record names in the dataset directory.
    
    Args:
        data_dir: Path to the directory containing the dataset
        
    Returns:
        List of record names (without file extensions)
    """
    # Look for .hea (header) files which indicate available records
    records = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for file in os.listdir(data_dir):
        if file.endswith('.hea'):
            # Remove the .hea extension to get record name
            record_name = file[:-4]
            records.append(record_name)
    
    if len(records) == 0:
        raise ValueError(f"No record files (.hea) found in {data_dir}")
    
    print(f"Found {len(records)} records in {data_dir}")
    return sorted(records)


def load_ecg_signal(data_dir: str, record_name: str) -> Tuple[np.ndarray, Dict]:
    """
    Load ECG signal from a record file.
    
    The PhysioNet database uses the WFDB format:
    - .dat file contains the signal data
    - .hea file contains the header/metadata
    
    Args:
        data_dir: Path to the directory containing the dataset
        record_name: Name of the record (without extension)
        
    Returns:
        Tuple of (ecg_signal, metadata):
            - ecg_signal: numpy array of ECG values
            - metadata: dictionary with sampling rate and other info
    """
    # Construct full path to the record
    record_path = os.path.join(data_dir, record_name)
    
    try:
        # Read the WFDB record
        # This reads both .dat and .hea files
        record = wfdb.rdrecord(record_path)
        
        # Extract ECG signal (first channel if multiple channels exist)
        ecg_signal = record.p_signal[:, 0]  # Shape: (num_samples,)
        
        # Extract metadata
        metadata = {
            'sampling_rate': record.fs,
            'signal_name': record.sig_name[0],
            'units': record.units[0],
            'num_samples': len(ecg_signal),
            'duration_seconds': len(ecg_signal) / record.fs
        }
        
        print(f"Loaded ECG signal: {record_name}")
        print(f"  - Duration: {metadata['duration_seconds']:.1f} seconds")
        print(f"  - Sampling rate: {metadata['sampling_rate']} Hz")
        print(f"  - Number of samples: {metadata['num_samples']}")
        
        return ecg_signal, metadata
        
    except Exception as e:
        raise RuntimeError(f"Error loading record {record_name}: {str(e)}")


def load_annotations(data_dir: str, record_name: str) -> np.ndarray:
    """
    Load apnea annotations from annotation file.
    
    The PhysioNet database provides binary annotation files (.apn).
    We use wfdb.rdann to read them properly.
    
    Args:
        data_dir: Path to the directory containing the dataset
        record_name: Name of the record (without extension)
        
    Returns:
        numpy array of labels for each minute (0 = Normal, 1 = Apnea)
    """
    # Construct path to record (without extension)
    record_path = os.path.join(data_dir, record_name)
    
    try:
        # Use WFDB to read binary annotation file
        annotation = wfdb.rdann(record_path, 'apn')
        
        # Get the annotation symbols (should be 'N' or 'A')
        symbols = annotation.symbol
        
        # Convert to numerical labels
        labels = []
        for symbol in symbols:
            if symbol == 'N':
                labels.append(0)  # Normal
            elif symbol == 'A':
                labels.append(1)  # Apnea
            # Skip any other symbols
        
        labels = np.array(labels)
        
        print(f"Loaded annotations: {record_name}")
        print(f"  - Total minutes: {len(labels)}")
        print(f"  - Apnea minutes: {np.sum(labels == 1)} ({100*np.mean(labels==1):.1f}%)")
        print(f"  - Normal minutes: {np.sum(labels == 0)} ({100*np.mean(labels==0):.1f}%)")
        
        return labels
        
    except Exception as e:
        print(f"Error loading annotations for {record_name}: {str(e)}")
        return None

def load_dataset(data_dir: str, max_records: int = None) -> Tuple[List, List]:
    """
    Load all ECG signals and annotations from the dataset.
    
    Args:
        data_dir: Path to the directory containing the dataset
        max_records: Maximum number of records to load (None = all)
        
    Returns:
        Tuple of (signals, annotations):
            - signals: List of (ecg_signal, metadata) tuples
            - annotations: List of label arrays
    """
    # Get list of all available records
    record_list = get_record_list(data_dir)
    
    # Limit number of records if specified
    if max_records is not None:
        record_list = record_list[:max_records]
        print(f"Loading first {max_records} records...")
    
    signals = []
    annotations = []
    
    # Load each record
    for i, record_name in enumerate(record_list, 1):
        print(f"\n[{i}/{len(record_list)}] Processing {record_name}...")
        
        try:
            # Load ECG signal
            ecg_signal, metadata = load_ecg_signal(data_dir, record_name)
            
            # Load annotations
            labels = load_annotations(data_dir, record_name)
            
            # Only include records that have both signal and annotations
            if labels is not None:
                signals.append((ecg_signal, metadata))
                annotations.append(labels)
            else:
                print(f"  Skipping {record_name} (no annotations)")
                
        except Exception as e:
            print(f"  Error processing {record_name}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Successfully loaded {len(signals)} records")
    print(f"{'='*50}\n")
    
    return signals, annotations


def get_class_distribution(annotations: List[np.ndarray]) -> Dict:
    """
    Calculate class distribution across all records.
    
    Args:
        annotations: List of label arrays
        
    Returns:
        Dictionary with class counts and percentages
    """
    total_normal = 0
    total_apnea = 0
    
    for labels in annotations:
        total_normal += np.sum(labels == 0)
        total_apnea += np.sum(labels == 1)
    
    total = total_normal + total_apnea
    
    distribution = {
        'normal_count': total_normal,
        'apnea_count': total_apnea,
        'total_count': total,
        'normal_percentage': 100 * total_normal / total if total > 0 else 0,
        'apnea_percentage': 100 * total_apnea / total if total > 0 else 0
    }
    
    return distribution


if __name__ == "__main__":
    """
    Test the data loader with a sample dataset.
    """
    # Test with a small number of records
    print("Testing data loader...")
    
    data_dir = config.DATA_DIR
    
    # Load first 5 records for testing
    signals, annotations = load_dataset(data_dir, max_records=5)
    
    # Print class distribution
    dist = get_class_distribution(annotations)
    print(f"\nClass Distribution:")
    print(f"  Normal: {dist['normal_count']} ({dist['normal_percentage']:.1f}%)")
    print(f"  Apnea:  {dist['apnea_count']} ({dist['apnea_percentage']:.1f}%)")
